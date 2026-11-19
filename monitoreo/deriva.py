"""Cálculo de deriva de datos (PSI, Kolmogorov-Smirnov, Chi-cuadrado), calidad y deriva conceptual.

Implementa la metodología de la Semana 14 y la Sección 6 del PSet #4:
- PSI con 5 quintiles abiertos [-inf, q20, q40, q60, q80, inf] calibrados para lotes de N=46 ramas
  y piso de proporción mínima 0.0001 para evitar ln(0).
- Prueba de Kolmogorov-Smirnov (ks_2samp) para variables numéricas.
- Prueba de bondad de ajuste Chi-cuadrado (chisquare) para variables categóricas.
- Diagnóstico cruzado de Deriva de Datos vs. Deriva Conceptual frente al MAE de CV del PSet #3
  (MAE_CV = 13.9173 ± 1.1121 M USD).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp

from api.esquemas import (
    CODIGOS_INDUSTRIA_VALIDOS,
    SECTORES_ECONOMICOS_VALIDOS,
    SECTORES_MACRO_VALIDOS,
)
from src.pipeline import (
    COLUMNAS_MODELO,
    HeuristicaBCE,
    OBJETIVO,
    calcular_costo_negocio_musd,
    calcular_metricas,
)

MINIMO_PSI = 0.0001
UMBRAL_PSI_ALERTA = 0.25

# Referencia oficial de Validación Cruzada (5 pliegues TimeSeriesSplit) del PSet #3 (@champion v1)
MAE_CV_PROMEDIO = 13.917309816494924
MAE_CV_STD = 1.1121422947058854
UMBRAL_MAE_2STD = MAE_CV_PROMEDIO + 2.0 * MAE_CV_STD  # 16.141594 M USD
UMBRAL_WAPE_DEGRADACION_PCT = 4.50

FEATURES_NUMERICAS_MONITOREO = [
    "VENTAS_NOMINALES_USD",
    "NUM_DECLARANTES",
    "INDICE_PRECIO_REAL",
    "RATIO_REPRESENTATIVIDAD_HIST_12M",
    "proyeccion_cobertura_12m",
    "impulso_declarantes",
]

FEATURES_CATEGORICAS_MONITOREO = [
    "SECTOR_MACRO",
    "SECTOR_ECONOMICO",
]


def _cortes_quintiles_abiertos(serie: pd.Series) -> list[float]:
    valores = serie.dropna().to_numpy(dtype=float)
    cuantiles = np.quantile(valores, [0.2, 0.4, 0.6, 0.8])
    # Asegurar bordes estrictamente crecientes si hay empates
    bordes = sorted(set(float(x) for x in cuantiles))
    return [-np.inf] + bordes + [np.inf]


def proporciones_numericas(serie: pd.Series, bordes: list[float]) -> np.ndarray:
    valores = serie.dropna().to_numpy(dtype=float)
    if len(valores) == 0:
        return np.zeros(len(bordes) - 1, dtype=float)
    conteo, _ = np.histogram(valores, bins=bordes)
    total = conteo.sum()
    return (conteo / total) if total > 0 else np.zeros(len(bordes) - 1, dtype=float)


def proporciones_categoricas(serie: pd.Series, categorias: list[str]) -> np.ndarray:
    limpia = serie.dropna().astype(str)
    if len(limpia) == 0:
        return np.zeros(len(categorias), dtype=float)
    frec = limpia.value_counts(normalize=True).reindex(categorias, fill_value=0.0)
    return frec.to_numpy(dtype=float)


def calcular_psi(
    prop_referencia: np.ndarray | list[float],
    prop_actual: np.ndarray | list[float],
    minimo: float = MINIMO_PSI,
) -> float:
    """Calcula el Population Stability Index (PSI) con piso `minimo` en casillas vacías."""
    ref = np.clip(np.asarray(prop_referencia, dtype=float), minimo, None)
    act = np.clip(np.asarray(prop_actual, dtype=float), minimo, None)
    return float(np.sum((act - ref) * np.log(act / ref)))


def construir_referencia_entrenamiento(
    train_df: pd.DataFrame,
    predicciones_train: np.ndarray,
) -> dict[str, Any]:
    """Construye las distribuciones de referencia de entrenamiento para PSI, KS y Chi².

    Para `INDICE_PRECIO_REAL` (índice acumulado con tendencia secular 2016-2024), utiliza
    la ventana del último año de entrenamiento (2024) como línea base del nivel de precios
    vigente al cierre del entrenamiento.
    """
    df_ref = train_df.copy()
    df_ref["prediccion_ventas_musd"] = np.clip(np.asarray(predicciones_train, dtype=float), 0.0, None)
    df_ref_reciente = df_ref[df_ref["TIEMPO_KEY"] >= 202401].copy()
    if df_ref_reciente.empty:
        df_ref_reciente = df_ref

    numericas_ref: dict[str, dict[str, Any]] = {}
    for col in FEATURES_NUMERICAS_MONITOREO + ["prediccion_ventas_musd"]:
        fuente = df_ref_reciente[col] if col == "INDICE_PRECIO_REAL" else df_ref[col]
        bordes = _cortes_quintiles_abiertos(fuente)
        props = proporciones_numericas(fuente, bordes)
        numericas_ref[col] = {
            "bordes": bordes,
            "proporciones": props.tolist(),
            "muestra_ks": fuente.dropna().to_numpy(dtype=float),
        }

    categoricas_ref: dict[str, dict[str, Any]] = {}
    for col in FEATURES_CATEGORICAS_MONITOREO:
        cats = sorted(df_ref[col].dropna().astype(str).unique().tolist())
        props = proporciones_categoricas(df_ref[col], cats)
        categoricas_ref[col] = {
            "categorias": cats,
            "proporciones": props.tolist(),
        }

    # Baselines ajustados sobre entrenamiento para comparación de negocio en cada lote
    heuristica = HeuristicaBCE(columna="proyeccion_cobertura_12m").fit(df_ref, df_ref[OBJETIVO])
    mediana_dummy = float(np.nanmedian(df_ref[OBJETIVO].to_numpy(dtype=float)))

    return {
        "numericas": numericas_ref,
        "categoricas": categoricas_ref,
        "heuristica_bce": heuristica,
        "mediana_dummy_train": mediana_dummy,
    }


def evaluar_calidad_datos_lote(df_lote: pd.DataFrame) -> dict[str, Any]:
    """Calcula métricas de calidad de datos del lote (Sección 6.1: faltantes, fuera de rango, categorías no vistas)."""
    n_filas = max(len(df_lote), 1)
    faltantes_por_col = {
        col: round(float(df_lote[col].isna().sum() / n_filas * 100.0), 4)
        for col in COLUMNAS_MODELO
        if col in df_lote.columns
    }
    pct_faltantes_max = max(faltantes_por_col.values()) if faltantes_por_col else 0.0
    pct_faltantes_promedio = (
        round(float(np.mean(list(faltantes_por_col.values()))), 4) if faltantes_por_col else 0.0
    )

    fuera_rango = 0
    if "VENTAS_NOMINALES_USD" in df_lote.columns:
        fuera_rango += int((df_lote["VENTAS_NOMINALES_USD"] < 0).sum())
    if "NUM_DECLARANTES" in df_lote.columns:
        fuera_rango += int((df_lote["NUM_DECLARANTES"] < 0).sum())
    if "INDICE_PRECIO_REAL" in df_lote.columns:
        fuera_rango += int((df_lote["INDICE_PRECIO_REAL"] <= 0).sum())
    if "RATIO_REPRESENTATIVIDAD_HIST_12M" in df_lote.columns:
        fuera_rango += int(
            (
                (df_lote["RATIO_REPRESENTATIVIDAD_HIST_12M"] < 0.0)
                | (df_lote["RATIO_REPRESENTATIVIDAD_HIST_12M"] > 1.5)
            ).sum()
        )

    codigos = df_lote["CODIGO_INDUSTRIA"].astype(str).apply(lambda x: f"{int(x):03d}" if x.isdigit() else x)
    no_vistas = int((~codigos.isin(CODIGOS_INDUSTRIA_VALIDOS)).sum())
    if "SECTOR_MACRO" in df_lote.columns:
        no_vistas += int((~df_lote["SECTOR_MACRO"].astype(str).isin(SECTORES_MACRO_VALIDOS)).sum())
    if "SECTOR_ECONOMICO" in df_lote.columns:
        no_vistas += int((~df_lote["SECTOR_ECONOMICO"].astype(str).isin(SECTORES_ECONOMICOS_VALIDOS)).sum())

    return {
        "pct_faltantes_max": round(float(pct_faltantes_max), 4),
        "pct_faltantes_promedio": pct_faltantes_promedio,
        "valores_fuera_rango": int(fuera_rango),
        "categorias_no_vistas": int(no_vistas),
        "faltantes_por_feature": faltantes_por_col,
    }


def diagnosticar_tipo_deriva(
    hay_deriva_datos: bool,
    target_disponible: bool,
    mae_lote: float | None,
    wape_lote_pct: float | None,
) -> tuple[str, str]:
    """Completa la tabla de diagnóstico y acción de la Sección 6.2 del PSet #4."""
    if not target_disponible or mae_lote is None or wape_lote_pct is None:
        if hay_deriva_datos:
            return (
                "Deriva de datos (target pendiente)",
                "Auditar fuente SRI/precios y vigilar hasta recepción del cierre definitivo (+30d)",
            )
        return ("Estable (target pendiente)", "Operación normal; esperar cierre definitivo (+30d)")

    cae_desempeno = (mae_lote > UMBRAL_MAE_2STD) and (wape_lote_pct > UMBRAL_WAPE_DEGRADACION_PCT)

    if hay_deriva_datos and not cae_desempeno:
        return (
            "Deriva de datos sin impacto",
            "Documentar cambio en distribución (ej. estacionalidad) y vigilar; NO reentrenar",
        )
    if hay_deriva_datos and cae_desempeno:
        return (
            "Deriva de datos con impacto",
            "Auditar fuente SRI/precios y disparar reentrenamiento.yml sobre lotes etiquetados",
        )
    if (not hay_deriva_datos) and cae_desempeno:
        return (
            "Deriva conceptual",
            "Disparar reentrenamiento.yml ponderando ventana reciente y revisar juicio humano",
        )
    return ("Estable", "Mantener modelo @champion actual en producción")


def evaluar_lote_completo(
    lote_id: str,
    tiempo_key: int,
    fecha_prediccion: str,
    fecha_target_disponible: str,
    df_lote_features: pd.DataFrame,
    predicciones_musd: np.ndarray,
    decisiones: list[str],
    latencias_ms: list[float],
    errores_4xx: int,
    errores_5xx: int,
    version_modelo: str,
    run_id: str,
    modo_respaldo: bool,
    referencia: dict[str, Any],
    target_disponible: bool = True,
) -> dict[str, Any]:
    """Calcula todas las métricas de sistema, datos, deriva (PSI/KS/Chi²) y desempeño de un lote."""
    df = df_lote_features.copy()
    preds = np.clip(np.asarray(predicciones_musd, dtype=float), 0.0, None)
    df["prediccion_ventas_musd"] = preds
    n_solicitudes = len(df)

    # 1. Métricas de sistema
    lat_p50 = round(float(np.percentile(latencias_ms, 50)), 3) if latencias_ms else 0.0
    lat_p95 = round(float(np.percentile(latencias_ms, 95)), 3) if latencias_ms else 0.0
    tasa_4xx = round(float(errores_4xx / max(n_solicitudes + errores_4xx, 1)), 4)
    tasa_5xx = round(float(errores_5xx / max(n_solicitudes + errores_5xx, 1)), 4)

    # 2. Calidad de datos
    calidad = evaluar_calidad_datos_lote(df)

    # 3. Deriva de datos: PSI + KS en numéricas y PSI + Chi² en categóricas
    psi_features: dict[str, float] = {}
    ks_pvalues: dict[str, float] = {}
    for col, ref_info in referencia["numericas"].items():
        prop_act = proporciones_numericas(df[col], ref_info["bordes"])
        psi_features[col] = round(calcular_psi(ref_info["proporciones"], prop_act), 4)
        muestra_act = df[col].dropna().to_numpy(dtype=float)
        if len(muestra_act) > 0:
            _, pval = ks_2samp(ref_info["muestra_ks"], muestra_act)
            ks_pvalues[col] = round(float(pval), 6)
        else:
            ks_pvalues[col] = 1.0

    chi2_pvalues: dict[str, float] = {}
    for col, ref_info in referencia["categoricas"].items():
        prop_act = proporciones_categoricas(df[col], ref_info["categorias"])
        psi_features[col] = round(calcular_psi(ref_info["proporciones"], prop_act), 4)
        f_exp = np.clip(np.asarray(ref_info["proporciones"], dtype=float) * n_solicitudes, 1e-6, None)
        f_obs = prop_act * n_solicitudes
        # Normalizar suma exacta para scipy.stats.chisquare
        f_exp = f_exp * (f_obs.sum() / f_exp.sum())
        _, pval_c = chisquare(f_obs=f_obs, f_exp=f_exp)
        chi2_pvalues[col] = round(float(pval_c), 6)

    # Features nucleares vigiladas para alerta de deriva de entrada
    features_nucleares = [
        "proyeccion_cobertura_12m",
        "VENTAS_NOMINALES_USD",
        "INDICE_PRECIO_REAL",
        "NUM_DECLARANTES",
        "RATIO_REPRESENTATIVIDAD_HIST_12M",
    ]
    psi_max_nuclear = max(psi_features[c] for c in features_nucleares)
    feature_max_psi = max(features_nucleares, key=lambda c: psi_features[c])
    ks_min_pvalue = min(ks_pvalues[c] for c in features_nucleares)
    hay_deriva_datos = bool(psi_max_nuclear >= UMBRAL_PSI_ALERTA)

    # 4. Distribución de predicciones y tasas de decisión
    serie_dec = pd.Series(decisiones)
    tasa_expansion = round(float((serie_dec == "publicar_flash_expansion").mean()), 4)
    tasa_estable = round(float((serie_dec == "publicar_flash_estable").mean()), 4)
    tasa_contraccion = round(float((serie_dec == "alerta_contraccion_comite").mean()), 4)
    tasa_revision = round(float((serie_dec == "revisar_cobertura_manual").mean()), 4)

    # 5. Desempeño técnico y de negocio (solo cuando el target definitivo del lote ya está disponible)
    if target_disponible and OBJETIVO in df.columns and df[OBJETIVO].notna().all():
        y_real = df[OBJETIVO].to_numpy(dtype=float)
        m_mod = calcular_metricas(y_real, preds)
        mae_lote = round(float(m_mod["mae"]), 4)
        rmse_lote = round(float(m_mod["rmse"]), 4)
        wape_lote = round(float(m_mod["wape_pct"]), 4)
        costo_lote = round(float(m_mod["costo_negocio_musd"]), 4)
        z_mae_cv = round(float((mae_lote - MAE_CV_PROMEDIO) / MAE_CV_STD), 4)

        # Comparación contra línea base de negocio (HeuristicaBCE) y "No hacer nada" (Dummy mediana)
        preds_bce = referencia["heuristica_bce"].predict(df)
        m_bce = calcular_metricas(y_real, preds_bce)
        mae_bce = round(float(m_bce["mae"]), 4)
        wape_bce = round(float(m_bce["wape_pct"]), 4)
        costo_bce = round(float(m_bce["costo_negocio_musd"]), 4)

        preds_dummy = np.full_like(y_real, float(referencia["mediana_dummy_train"]))
        costo_dummy = round(float(calcular_costo_negocio_musd(y_real, preds_dummy)), 4)
        ahorro_vs_bce = round(float(costo_bce - costo_lote), 4)
    else:
        mae_lote = rmse_lote = wape_lote = costo_lote = z_mae_cv = None
        mae_bce = wape_bce = costo_bce = costo_dummy = ahorro_vs_bce = None

    diagnostico, accion_recomendada = diagnosticar_tipo_deriva(
        hay_deriva_datos=hay_deriva_datos,
        target_disponible=target_disponible,
        mae_lote=mae_lote,
        wape_lote_pct=wape_lote,
    )

    return {
        "lote_id": lote_id,
        "tiempo_key": int(tiempo_key),
        "fecha_prediccion": fecha_prediccion,
        "fecha_target_disponible": fecha_target_disponible,
        "target_disponible": bool(target_disponible),
        "version_modelo": version_modelo,
        "run_id": run_id,
        "modo_respaldo": bool(modo_respaldo),
        "volumen_solicitudes": int(n_solicitudes),
        "latencia_p50_ms": lat_p50,
        "latencia_p95_ms": lat_p95,
        "tasa_error_4xx": tasa_4xx,
        "tasa_error_5xx": tasa_5xx,
        "pct_faltantes_max": calidad["pct_faltantes_max"],
        "valores_fuera_rango": calidad["valores_fuera_rango"],
        "categorias_no_vistas": calidad["categorias_no_vistas"],
        "pred_media_musd": round(float(np.mean(preds)), 4),
        "pred_p50_musd": round(float(np.percentile(preds, 50)), 4),
        "pred_p95_musd": round(float(np.percentile(preds, 95)), 4),
        "tasa_expansion": tasa_expansion,
        "tasa_estable": tasa_estable,
        "tasa_contraccion": tasa_contraccion,
        "tasa_revision_manual": tasa_revision,
        "psi_ventas_nominales_usd": psi_features["VENTAS_NOMINALES_USD"],
        "psi_num_declarantes": psi_features["NUM_DECLARANTES"],
        "psi_indice_precio_real": psi_features["INDICE_PRECIO_REAL"],
        "psi_cobertura_hist_12m": psi_features["RATIO_REPRESENTATIVIDAD_HIST_12M"],
        "psi_proyeccion_cobertura_12m": psi_features["proyeccion_cobertura_12m"],
        "psi_impulso_declarantes": psi_features["impulso_declarantes"],
        "psi_sector_macro": psi_features["SECTOR_MACRO"],
        "psi_prediccion_ventas_musd": psi_features["prediccion_ventas_musd"],
        "psi_max_nuclear": round(float(psi_max_nuclear), 4),
        "feature_max_psi": feature_max_psi,
        "ks_pvalue_proyeccion": ks_pvalues["proyeccion_cobertura_12m"],
        "ks_pvalue_precios": ks_pvalues["INDICE_PRECIO_REAL"],
        "ks_min_pvalue": round(float(ks_min_pvalue), 6),
        "chi2_pvalue_sector_macro": chi2_pvalues["SECTOR_MACRO"],
        "mae_lote": mae_lote,
        "rmse_lote": rmse_lote,
        "wape_lote_pct": wape_lote,
        "costo_negocio_lote_musd": costo_lote,
        "z_score_mae_cv": z_mae_cv,
        "mae_heuristica_bce": mae_bce,
        "wape_heuristica_bce_pct": wape_bce,
        "costo_heuristica_bce_musd": costo_bce,
        "costo_dummy_no_hacer_nada_musd": costo_dummy,
        "ahorro_vs_heuristica_musd": ahorro_vs_bce,
        "hay_deriva_datos": hay_deriva_datos,
        "diagnostico_deriva": diagnostico,
        "accion_recomendada": accion_recomendada,
    }
