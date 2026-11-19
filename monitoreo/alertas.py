"""Motor de reglas de alerta y persistencia de tablas MONITOREO y ALERTAS en Snowflake (Sección 6.3).

Aplica la plantilla oficial de alertas de la Sección 6.3 del PSet #4:
- Métrica | Umbral | Ventana | Severidad | Quién la recibe | Acción
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from monitoreo.deriva import UMBRAL_MAE_2STD, UMBRAL_PSI_ALERTA, UMBRAL_WAPE_DEGRADACION_PCT

logger = logging.getLogger("monitoreo.alertas")

CATALOGO_REGLAS_ALERTA = [
    {
        "metrica": "psi_proyeccion_cobertura_12m",
        "umbral_num": UMBRAL_PSI_ALERTA,
        "umbral_texto": ">= 0.25",
        "ventana": "1 lote",
        "severidad": "Alta",
        "destinatario": "Equipo de Datos / SCNC",
        "accion": "Revisar fuente F104 SRI y evaluar impacto en desempeño (reentrenar solo si cae MAE/WAPE)",
    },
    {
        "metrica": "psi_ventas_nominales_usd",
        "umbral_num": UMBRAL_PSI_ALERTA,
        "umbral_texto": ">= 0.25",
        "ventana": "1 lote",
        "severidad": "Alta",
        "destinatario": "Equipo de Datos / SCNC",
        "accion": "Auditar cobertura de declaraciones tempranas día 15 en SRI",
    },
    {
        "metrica": "psi_indice_precio_real",
        "umbral_num": UMBRAL_PSI_ALERTA,
        "umbral_texto": ">= 0.25",
        "ventana": "1 lote",
        "severidad": "Media",
        "destinatario": "Analista de Precios / SCNC",
        "accion": "Verificar choque inflacionario en índice sectorial CNT-46 e IPC/IPP",
    },
    {
        "metrica": "mae_lote_degradacion",
        "umbral_num": round(UMBRAL_MAE_2STD, 2),
        "umbral_texto": f"> {UMBRAL_MAE_2STD:.2f} M USD (MAE_CV + 2*std y WAPE > {UMBRAL_WAPE_DEGRADACION_PCT:.1f}%)",
        "ventana": "1 lote crítico o 2 lotes consecutivos",
        "severidad": "Crítica",
        "destinatario": "Comité Técnico MLOps / SCNC",
        "accion": "Disparar flujo Kestra reentrenamiento.yml y comparar challenger vs champion",
    },
    {
        "metrica": "costo_negocio_vs_heuristica_bce",
        "umbral_num": 0.0,
        "umbral_texto": "Costo modelo > Costo HeuristicaBCE",
        "ventana": "2 lotes consecutivos",
        "severidad": "Crítica",
        "destinatario": "Subgerencia de Cuentas Nacionales (BCE)",
        "accion": "Evaluar reentrenamiento o conmutar interruptor a modo respaldo HeuristicaBCE",
    },
    {
        "metrica": "latencia_p95_ms_o_errores_5xx",
        "umbral_num": 200.0,
        "umbral_texto": "p95 > 200 ms o errores_5xx > 1%",
        "ventana": "1 lote",
        "severidad": "Alta",
        "destinatario": "Ingeniería MLOps",
        "accion": "Inspeccionar contenedor FastAPI y conexión a Snowflake/MLflow",
    },
]


def evaluar_alertas_lote(
    resumen_lote: dict[str, Any],
    historial_lotes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Evalúa las reglas de alerta sobre un lote y devuelve la lista de alertas disparadas."""
    historial = historial_lotes or []
    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lote_id = str(resumen_lote["lote_id"])
    tiempo_key = int(resumen_lote["tiempo_key"])
    alertas: list[dict[str, Any]] = []

    def _crear_alerta(
        metrica: str,
        valor: float,
        umbral_texto: str,
        ventana: str,
        severidad: str,
        destinatario: str,
        accion: str,
        dispara_reentrenamiento: bool = False,
    ) -> dict[str, Any]:
        alerta = {
            "id_alerta": uuid.uuid4().hex[:16],
            "timestamp_utc": ts_utc,
            "lote_id": lote_id,
            "tiempo_key": tiempo_key,
            "metrica": metrica,
            "valor_observado": round(float(valor), 4),
            "umbral": umbral_texto,
            "ventana": ventana,
            "severidad": severidad,
            "quien_la_recibe": destinatario,
            "accion": accion,
            "dispara_reentrenamiento": bool(dispara_reentrenamiento),
        }
        logger.warning(
            "[ALERTA %s] Lote=%s | %s=%.4f (umbral %s) -> %s",
            severidad.upper(),
            lote_id,
            metrica,
            valor,
            umbral_texto,
            accion,
        )
        return alerta

    # 1. Alertas de deriva de datos (PSI >= 0.25)
    psi_proy = float(resumen_lote.get("psi_proyeccion_cobertura_12m") or 0.0)
    if psi_proy >= UMBRAL_PSI_ALERTA:
        alertas.append(
            _crear_alerta(
                metrica="PSI de proyeccion_cobertura_12m",
                valor=psi_proy,
                umbral_texto=">= 0.25",
                ventana="1 lote",
                severidad="Alta",
                destinatario="Equipo de Datos / SCNC",
                accion="Revisar fuente F104 SRI y evaluar reentrenamiento si cae el desempeño",
                dispara_reentrenamiento=False,
            )
        )

    psi_ventas = float(resumen_lote.get("psi_ventas_nominales_usd") or 0.0)
    if psi_ventas >= UMBRAL_PSI_ALERTA:
        alertas.append(
            _crear_alerta(
                metrica="PSI de VENTAS_NOMINALES_USD",
                valor=psi_ventas,
                umbral_texto=">= 0.25",
                ventana="1 lote",
                severidad="Alta",
                destinatario="Equipo de Datos / SCNC",
                accion="Auditar cobertura del corte temprano día 15 en SRI",
                dispara_reentrenamiento=False,
            )
        )

    psi_precios = float(resumen_lote.get("psi_indice_precio_real") or 0.0)
    if psi_precios >= UMBRAL_PSI_ALERTA:
        alertas.append(
            _crear_alerta(
                metrica="PSI de INDICE_PRECIO_REAL",
                valor=psi_precios,
                umbral_texto=">= 0.25",
                ventana="1 lote",
                severidad="Media",
                destinatario="Analista de Precios / SCNC",
                accion="Verificar choque inflacionario en índice sectorial CNT-46",
                dispara_reentrenamiento=False,
            )
        )

    # 2. Alerta crítica de degradación de desempeño (solo si el target del lote ya llegó)
    if resumen_lote.get("target_disponible") and resumen_lote.get("mae_lote") is not None:
        mae_actual = float(resumen_lote["mae_lote"])
        wape_actual = float(resumen_lote.get("wape_lote_pct") or 0.0)
        mae_prev = (
            float(historial[-1]["mae_lote"])
            if historial and historial[-1].get("mae_lote") is not None
            else 0.0
        )

        # Se dispara por degradación severa (MAE > 16.14 M y WAPE > 4.5%) o por 2 lotes consecutivos > 16.14 M
        degradacion_critica = (mae_actual > UMBRAL_MAE_2STD and wape_actual > UMBRAL_WAPE_DEGRADACION_PCT) or (
            mae_actual > UMBRAL_MAE_2STD and mae_prev > UMBRAL_MAE_2STD
        )
        if degradacion_critica:
            alertas.append(
                _crear_alerta(
                    metrica="MAE del lote (degradación de desempeño)",
                    valor=mae_actual,
                    umbral_texto=f"> {UMBRAL_MAE_2STD:.2f} M USD (MAE_CV + 2*std)",
                    ventana="1 lote crítico o 2 lotes consecutivos",
                    severidad="Crítica",
                    destinatario="Comité Técnico MLOps / SCNC",
                    accion="Disparar flujo Kestra reentrenamiento.yml y evaluar challenger vs champion",
                    dispara_reentrenamiento=True,
                )
            )

        # 3. Alerta de costo de negocio frente a HeuristicaBCE en 2 lotes consecutivos
        costo_act = float(resumen_lote.get("costo_negocio_lote_musd") or 0.0)
        costo_bce_act = float(resumen_lote.get("costo_heuristica_bce_musd") or 0.0)
        if historial and historial[-1].get("costo_negocio_lote_musd") is not None:
            costo_prev = float(historial[-1]["costo_negocio_lote_musd"])
            costo_bce_prev = float(historial[-1].get("costo_heuristica_bce_musd") or 0.0)
            if (costo_act > costo_bce_act) and (costo_prev > costo_bce_prev):
                alertas.append(
                    _crear_alerta(
                        metrica="Costo de negocio vs HeuristicaBCE",
                        valor=costo_act,
                        umbral_texto=f"> {costo_bce_act:.2f} M USD (2 lotes)",
                        ventana="2 lotes consecutivos",
                        severidad="Crítica",
                        destinatario="Subgerencia de Cuentas Nacionales (BCE)",
                        accion="Evaluar reentrenamiento o activar modo respaldo HeuristicaBCE",
                        dispara_reentrenamiento=True,
                    )
                )

    # 4. Alerta operativa de sistema (latencia p95 o errores 5xx)
    lat_p95 = float(resumen_lote.get("latencia_p95_ms") or 0.0)
    err_5xx = float(resumen_lote.get("tasa_error_5xx") or 0.0)
    if lat_p95 > 200.0 or err_5xx > 0.01:
        alertas.append(
            _crear_alerta(
                metrica="Latencia p95 / Tasa errores 5xx",
                valor=lat_p95 if lat_p95 > 200.0 else err_5xx,
                umbral_texto="p95 > 200 ms o 5xx > 1%",
                ventana="1 lote",
                severidad="Alta",
                destinatario="Ingeniería MLOps",
                accion="Revisar contenedor API FastAPI y recursos de red/base de datos",
                dispara_reentrenamiento=False,
            )
        )

    return alertas


def guardar_monitoreo_y_alertas_snowflake(
    resumenes_lotes: list[dict[str, Any]],
    alertas_totales: list[dict[str, Any]],
    reemplazar: bool = True,
) -> None:
    """Crea y puebla las tablas PSET2_DB.ANALYTICS.MONITOREO y PSET2_DB.ANALYTICS.ALERTAS en Snowflake."""
    from src.datos import _conectar_snowflake

    conn = _conectar_snowflake()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS PSET2_DB.ANALYTICS.MONITOREO (
                LOTE_ID VARCHAR(64),
                TIEMPO_KEY INTEGER,
                FECHA_PREDICCION VARCHAR(32),
                FECHA_TARGET_DISPONIBLE VARCHAR(32),
                TARGET_DISPONIBLE BOOLEAN,
                VERSION_MODELO VARCHAR(64),
                RUN_ID VARCHAR(64),
                MODO_RESPALDO BOOLEAN,
                VOLUMEN_SOLICITUDES INTEGER,
                LATENCIA_P50_MS FLOAT,
                LATENCIA_P95_MS FLOAT,
                TASA_ERROR_4XX FLOAT,
                TASA_ERROR_5XX FLOAT,
                PCT_FALTANTES_MAX FLOAT,
                VALORES_FUERA_RANGO INTEGER,
                CATEGORIAS_NO_VISTAS INTEGER,
                PRED_MEDIA_MUSD FLOAT,
                PRED_P50_MUSD FLOAT,
                PRED_P95_MUSD FLOAT,
                TASA_EXPANSION FLOAT,
                TASA_ESTABLE FLOAT,
                TASA_CONTRACCION FLOAT,
                TASA_REVISION_MANUAL FLOAT,
                PSI_VENTAS_NOMINALES_USD FLOAT,
                PSI_NUM_DECLARANTES FLOAT,
                PSI_INDICE_PRECIO_REAL FLOAT,
                PSI_COBERTURA_HIST_12M FLOAT,
                PSI_PROYECCION_COBERTURA_12M FLOAT,
                PSI_IMPULSO_DECLARANTES FLOAT,
                PSI_SECTOR_MACRO FLOAT,
                PSI_PREDICCION_VENTAS_MUSD FLOAT,
                PSI_MAX_NUCLEAR FLOAT,
                FEATURE_MAX_PSI VARCHAR(64),
                KS_PVALUE_PROYECCION FLOAT,
                KS_PVALUE_PRECIOS FLOAT,
                KS_MIN_PVALUE FLOAT,
                CHI2_PVALUE_SECTOR_MACRO FLOAT,
                MAE_LOTE FLOAT,
                RMSE_LOTE FLOAT,
                WAPE_LOTE_PCT FLOAT,
                COSTO_NEGOCIO_LOTE_MUSD FLOAT,
                Z_SCORE_MAE_CV FLOAT,
                MAE_HEURISTICA_BCE FLOAT,
                WAPE_HEURISTICA_BCE_PCT FLOAT,
                COSTO_HEURISTICA_BCE_MUSD FLOAT,
                COSTO_DUMMY_NO_HACER_NADA_MUSD FLOAT,
                AHORRO_VS_HEURISTICA_MUSD FLOAT,
                HAY_DERIVA_DATOS BOOLEAN,
                DIAGNOSTICO_DERIVA VARCHAR(128),
                ACCION_RECOMENDADA VARCHAR(256),
                NUM_ALERTAS INTEGER,
                DISPARO_REENTRENAMIENTO BOOLEAN
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS PSET2_DB.ANALYTICS.ALERTAS (
                ID_ALERTA VARCHAR(64),
                TIMESTAMP_UTC VARCHAR(32),
                LOTE_ID VARCHAR(64),
                TIEMPO_KEY INTEGER,
                METRICA VARCHAR(128),
                VALOR_OBSERVADO FLOAT,
                UMBRAL VARCHAR(128),
                VENTA_EVALUACION VARCHAR(64),
                SEVERIDAD VARCHAR(32),
                QUIEN_LA_RECIBE VARCHAR(128),
                ACCION VARCHAR(256),
                DISPARA_REENTRENAMIENTO BOOLEAN
            )
            """
        )

        if reemplazar:
            cur.execute("TRUNCATE TABLE PSET2_DB.ANALYTICS.MONITOREO")
            cur.execute("TRUNCATE TABLE PSET2_DB.ANALYTICS.ALERTAS")

        if resumenes_lotes:
            cols_mon = [
                "lote_id",
                "tiempo_key",
                "fecha_prediccion",
                "fecha_target_disponible",
                "target_disponible",
                "version_modelo",
                "run_id",
                "modo_respaldo",
                "volumen_solicitudes",
                "latencia_p50_ms",
                "latencia_p95_ms",
                "tasa_error_4xx",
                "tasa_error_5xx",
                "pct_faltantes_max",
                "valores_fuera_rango",
                "categorias_no_vistas",
                "pred_media_musd",
                "pred_p50_musd",
                "pred_p95_musd",
                "tasa_expansion",
                "tasa_estable",
                "tasa_contraccion",
                "tasa_revision_manual",
                "psi_ventas_nominales_usd",
                "psi_num_declarantes",
                "psi_indice_precio_real",
                "psi_cobertura_hist_12m",
                "psi_proyeccion_cobertura_12m",
                "psi_impulso_declarantes",
                "psi_sector_macro",
                "psi_prediccion_ventas_musd",
                "psi_max_nuclear",
                "feature_max_psi",
                "ks_pvalue_proyeccion",
                "ks_pvalue_precios",
                "ks_min_pvalue",
                "chi2_pvalue_sector_macro",
                "mae_lote",
                "rmse_lote",
                "wape_lote_pct",
                "costo_negocio_lote_musd",
                "z_score_mae_cv",
                "mae_heuristica_bce",
                "wape_heuristica_bce_pct",
                "costo_heuristica_bce_musd",
                "costo_dummy_no_hacer_nada_musd",
                "ahorro_vs_heuristica_musd",
                "hay_deriva_datos",
                "diagnostico_deriva",
                "accion_recomendada",
                "num_alertas",
                "disparo_reentrenamiento",
            ]
            placeholders = ", ".join(["%s"] * len(cols_mon))
            sql_mon = f"INSERT INTO PSET2_DB.ANALYTICS.MONITOREO VALUES ({placeholders})"
            filas_mon = [tuple(r.get(c) for c in cols_mon) for r in resumenes_lotes]
            cur.executemany(sql_mon, filas_mon)

        if alertas_totales:
            sql_al = """
                INSERT INTO PSET2_DB.ANALYTICS.ALERTAS VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            filas_al = [
                (
                    a["id_alerta"],
                    a["timestamp_utc"],
                    a["lote_id"],
                    int(a["tiempo_key"]),
                    a["metrica"],
                    float(a["valor_observado"]),
                    a["umbral"],
                    a["ventana"],
                    a["severidad"],
                    a["quien_la_recibe"],
                    a["accion"],
                    bool(a["dispara_reentrenamiento"]),
                )
                for a in alertas_totales
            ]
            cur.executemany(sql_al, filas_al)

        conn.commit()
    finally:
        cur.close()
        conn.close()
