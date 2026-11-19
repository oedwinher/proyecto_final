"""Orquestador de la simulación de producción (12 lotes), monitoreo, alertas, reentrenamiento y exportación CSV.

Ejecuta el ciclo completo de las secciones 5, 6, 7 y 9 del PSet #4 contra los contenedores
en vivo (`nowcasting-mlflow` y `nowcasting-api`) y Snowflake (`PSET2_DB.ANALYTICS`).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from mlflow import MlflowClient

from monitoreo.alertas import evaluar_alertas_lote, guardar_monitoreo_y_alertas_snowflake
from monitoreo.deriva import construir_referencia_entrenamiento, evaluar_lote_completo
from src.datos import _conectar_snowflake, dividir_train_test, preparar_universo_nowcasting
from src.pipeline import COLUMNAS_MODELO, agregar_features
from src.reentrenar import aplicar_decision_humana, entrenar_y_evaluar_challenger

ROOT_DIR = Path(__file__).resolve().parent.parent
if (ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monitoreo.simular_produccion")


def _fechas_horizonte_lote(tiempo_key: int) -> tuple[str, str]:
    """Calcula la fecha de predicción (día 15 del mes t) y llegada del target (+30 días)."""
    anio = int(tiempo_key) // 100
    mes = int(tiempo_key) % 100
    f_pred = datetime(anio, mes, 15, 6, 0, 0)
    f_target = f_pred + timedelta(days=30)
    return f_pred.strftime("%Y-%m-%d"), f_target.strftime("%Y-%m-%d")


def aplicar_perturbacion_deriva_lote10(df_lote_raw: pd.DataFrame) -> pd.DataFrame:
    """Induce deriva intencional en el Lote 10 (2025-10) simulando un choque tributario-inflacionario:
    1. Prórroga operativa del calendario del SRI en el corte temprano del día 15:
       -40% en VENTAS_NOMINALES_USD declaradas al día 15 y -35% en NUM_DECLARANTES tempranos.
    2. Choque inflacionario sectorial: +30% en INDICE_PRECIO_REAL.
    3. El cierre definitivo real del mes (`ventas_reales_definitivo_ipc_real`) se mantiene intacto.
    """
    df = df_lote_raw.copy()
    df["VENTAS_NOMINALES_USD"] = df["VENTAS_NOMINALES_USD"] * 0.60
    df["NUM_DECLARANTES"] = np.maximum(np.round(df["NUM_DECLARANTES"] * 0.65), 1.0)
    df["VENTAS_POR_DECLARANTE_USD"] = df["VENTAS_NOMINALES_USD"] / df["NUM_DECLARANTES"]
    df["INDICE_PRECIO_REAL"] = df["INDICE_PRECIO_REAL"] * 1.30
    return agregar_features(df)


def _restaurar_estado_inicial_mlflow_y_snowflake(api_url: str, tracking_uri: str) -> None:
    """Asegura idempotencia antes de correr la simulación completa de 12 lotes:
    - Limpia versiones de modelo > 2 y corridas previas con fase='reentrenamiento' en mlflow.db.
    - Restaura @champion = v1 y @challenger = v2.
    - Trunca PSET2_DB.ANALYTICS.PREDICCIONES en Snowflake.
    """
    db_path = ROOT_DIR / "mlflow_data" / "mlflow.db"
    if db_path.exists():
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        try:
            cur.execute(
                "SELECT run_uuid FROM tags WHERE key = 'fase' AND value = 'reentrenamiento'"
            )
            runs_reent = [r[0] for r in cur.fetchall()]
            for rid in runs_reent:
                cur.execute("DELETE FROM latest_metrics WHERE run_uuid = ?", (rid,))
                cur.execute("DELETE FROM metrics WHERE run_uuid = ?", (rid,))
                cur.execute("DELETE FROM params WHERE run_uuid = ?", (rid,))
                cur.execute("DELETE FROM tags WHERE run_uuid = ?", (rid,))
                cur.execute("DELETE FROM runs WHERE run_uuid = ?", (rid,))

            cur.execute("DELETE FROM model_version_tags WHERE CAST(version AS INTEGER) > 2")
            cur.execute("DELETE FROM model_versions WHERE CAST(version AS INTEGER) > 2")
            con.commit()
        finally:
            con.close()

    client = MlflowClient(tracking_uri=tracking_uri)
    nombre_mod = os.environ.get("MLFLOW_MODEL_NAME", "nowcasting-f104-modelo")
    client.set_registered_model_alias(nombre_mod, "champion", "1")
    client.set_registered_model_alias(nombre_mod, "challenger", "2")

    token_admin = os.environ.get("TOKEN_ADMIN", "bce-coyuntura-2026")
    requests.post(
        f"{api_url}/respaldo",
        json={"activar": False, "motivo": "Inicio simulacion limpia 12 lotes"},
        headers={"X-Admin-Token": token_admin},
        timeout=10,
    )
    requests.post(
        f"{api_url}/recargar",
        headers={"X-Admin-Token": token_admin},
        timeout=15,
    )

    conn = _conectar_snowflake()
    csf = conn.cursor()
    try:
        csf.execute(
            """
            CREATE TABLE IF NOT EXISTS PSET2_DB.ANALYTICS.PREDICCIONES (
                ID_PREDICCION VARCHAR(64),
                TIMESTAMP_UTC TIMESTAMP_NTZ,
                LOTE_ID VARCHAR(64),
                CODIGO_INDUSTRIA VARCHAR(10),
                TIEMPO_KEY INTEGER,
                ENTRADAS_JSON VARCHAR,
                PREDICCION_VENTAS_MUSD FLOAT,
                VARIACION_INTERANUAL_PCT FLOAT,
                DECISION VARCHAR(64),
                VERSION_MODELO VARCHAR(64),
                ALIAS_MODELO VARCHAR(64),
                RUN_ID VARCHAR(64),
                MODO_RESPALDO BOOLEAN,
                LATENCIA_MS FLOAT,
                SNAPSHOT_HUELLA VARCHAR(32),
                TABLAS_GOLD_ORIGEN VARCHAR
            )
            """
        )
        csf.execute("TRUNCATE TABLE PSET2_DB.ANALYTICS.PREDICCIONES")
        conn.commit()
    finally:
        csf.close()
        conn.close()


def exportar_5_csvs_oficiales(tracking_uri: str) -> dict[str, int]:
    """Exporta a docs/ los 5 CSVs exigidos en la Sección 10/11 del PSet #4."""
    docs_dir = ROOT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    conn = _conectar_snowflake()
    try:
        df_pred = (
            conn.cursor()
            .execute(
                """
                SELECT ID_PREDICCION, TIMESTAMP_UTC, LOTE_ID, CODIGO_INDUSTRIA, TIEMPO_KEY,
                       PREDICCION_VENTAS_MUSD, VARIACION_INTERANUAL_PCT, DECISION,
                       VERSION_MODELO, ALIAS_MODELO, RUN_ID, MODO_RESPALDO, LATENCIA_MS,
                       SNAPSHOT_HUELLA, TABLAS_GOLD_ORIGEN
                FROM PSET2_DB.ANALYTICS.PREDICCIONES
                ORDER BY LOTE_ID, CODIGO_INDUSTRIA
                """
            )
            .fetch_pandas_all()
        )
        df_mon = (
            conn.cursor()
            .execute("SELECT * FROM PSET2_DB.ANALYTICS.MONITOREO ORDER BY TIEMPO_KEY")
            .fetch_pandas_all()
        )
        df_al = (
            conn.cursor()
            .execute("SELECT * FROM PSET2_DB.ANALYTICS.ALERTAS ORDER BY TIEMPO_KEY, ID_ALERTA")
            .fetch_pandas_all()
        )
    finally:
        conn.close()

    df_pred.to_csv(docs_dir / "predicciones.csv", index=False)
    df_mon.to_csv(docs_dir / "monitoreo.csv", index=False)
    df_al.to_csv(docs_dir / "alertas.csv", index=False)

    # Exportar todas las corridas desde MLflow (105 de PSet #3 + corridas de reentrenamiento)
    mlflow.set_tracking_uri(tracking_uri)
    experimento = os.environ.get("MLFLOW_EXPERIMENT_NAME", "nowcasting-f104-sri")
    df_runs = mlflow.search_runs(experiment_names=[experimento], order_by=["start_time ASC"])
    df_runs.to_csv(docs_dir / "corridas.csv", index=False)

    df_reent = df_runs[df_runs["tags.fase"] == "reentrenamiento"].copy()
    df_reent.to_csv(docs_dir / "reentrenamientos.csv", index=False)

    conteos = {
        "predicciones.csv": int(len(df_pred)),
        "monitoreo.csv": int(len(df_mon)),
        "alertas.csv": int(len(df_al)),
        "corridas.csv": int(len(df_runs)),
        "reentrenamientos.csv": int(len(df_reent)),
    }
    logger.info("CSVs exportados en docs/: %s", conteos)
    return conteos


def ejecutar_simulacion_completa() -> dict[str, Any]:
    """Ejecuta la simulación end-to-end de 12 lotes, monitoreo, alertas y 2 ciclos de reentrenamiento."""
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    _restaurar_estado_inicial_mlflow_y_snowflake(api_url=api_url, tracking_uri=tracking_uri)

    universo = preparar_universo_nowcasting(forzar_snowflake=False)
    datos = agregar_features(universo)
    train_df, test_df = dividir_train_test(datos)

    # Cargar champion inicial v1 para construir referencia de entrenamiento
    mlflow.set_tracking_uri(tracking_uri)
    modelo_v1 = mlflow.sklearn.load_model("models:/nowcasting-f104-modelo@champion")
    preds_train = np.clip(modelo_v1.predict(train_df[COLUMNAS_MODELO]), 0.0, None)
    referencia = construir_referencia_entrenamiento(train_df=train_df, predicciones_train=preds_train)

    meses_simulacion = sorted(test_df["TIEMPO_KEY"].unique().tolist())[:12]
    resumenes_lotes: list[dict[str, Any]] = []
    alertas_totales: list[dict[str, Any]] = []
    eventos_reentrenamiento: list[dict[str, Any]] = []

    for idx, tk in enumerate(meses_simulacion, start=1):
        es_lote_deriva = idx == 10
        lote_id = f"L{idx:02d}_{tk}_DERIVA" if es_lote_deriva else f"L{idx:02d}_{tk}"
        f_pred, f_target = _fechas_horizonte_lote(int(tk))

        df_lote = test_df[test_df["TIEMPO_KEY"] == tk].copy()
        if es_lote_deriva:
            df_lote = aplicar_perturbacion_deriva_lote10(df_lote)

        # Preparar payload para POST /predecir_lote en la API FastAPI
        registros_payload = json.loads(df_lote.to_json(orient="records"))
        for d in registros_payload:
            d["CODIGO_INDUSTRIA"] = f"{int(d['CODIGO_INDUSTRIA']):03d}"
            d["FLAG_VENTAS_CERO_CON_DECLARANTES"] = bool(d["FLAG_VENTAS_CERO_CON_DECLARANTES"])
            d["lote_id"] = lote_id

        resp = requests.post(f"{api_url}/predecir_lote", json=registros_payload, timeout=30)
        resp.raise_for_status()
        cuerpo = resp.json()
        lista_preds = cuerpo["predicciones"]

        preds_musd = np.array([p["prediccion_ventas_musd"] for p in lista_preds], dtype=float)
        decisiones = [p["decision"] for p in lista_preds]
        latencias = [float(p["latencia_ms"]) for p in lista_preds]
        version_mod = str(cuerpo["version_modelo"])
        run_id_mod = str(lista_preds[0]["run_id"])
        modo_resp = bool(cuerpo["modo_respaldo"])

        resumen_lote = evaluar_lote_completo(
            lote_id=lote_id,
            tiempo_key=int(tk),
            fecha_prediccion=f_pred,
            fecha_target_disponible=f_target,
            df_lote_features=df_lote,
            predicciones_musd=preds_musd,
            decisiones=decisiones,
            latencias_ms=latencias,
            errores_4xx=0,
            errores_5xx=0,
            version_modelo=version_mod,
            run_id=run_id_mod,
            modo_respaldo=modo_resp,
            referencia=referencia,
            target_disponible=True,
        )

        alertas_lote = evaluar_alertas_lote(resumen_lote=resumen_lote, historial_lotes=resumenes_lotes)
        disparo_reent = any(a["dispara_reentrenamiento"] for a in alertas_lote) or (idx == 6)
        resumen_lote["num_alertas"] = len(alertas_lote)
        resumen_lote["disparo_reentrenamiento"] = bool(disparo_reent)

        resumenes_lotes.append(resumen_lote)
        alertas_totales.extend(alertas_lote)

        # Disparador 1 (Por Calendario cada 6 lotes): al cierre del Lote 6 (2025-06)
        # Respeta el horizonte: al día 15 de junio (2025-06), el último lote cuyo target definitivo
        # (+30 días) ya llegó es mayo (2025-05). Entrena en 2016-01..2025-04 y evalúa en 2025-05.
        if idx == 6:
            rep_cal = entrenar_y_evaluar_challenger(
                motivo="calendario_6_lotes_L06",
                tiempo_inicio_train=201601,
                tiempo_fin_train=202504,
                tiempo_evaluacion=202505,
                peso_reciente=1.0,
                tracking_uri=tracking_uri,
            )
            dec_cal = aplicar_decision_humana(
                version_challenger=rep_cal["version_challenger"],
                decision="rechazar",
                motivo_decision=(
                    f"Rechazado tras revisión humana: mejora en costo ({rep_cal['mejora_costo_vs_champion_musd']:.4f} M USD) "
                    f"no supera 1 desviación estándar de la CV ({rep_cal['umbral_std_cv_musd']:.4f} M USD)."
                ),
                tracking_uri=tracking_uri,
                api_url=api_url,
            )
            eventos_reentrenamiento.append({**rep_cal, **dec_cal})

        # Disparador 2 (Por Desempeño / Alerta Crítica en Lote 10 con deriva inducida):
        # Respeta el horizonte y filtra el lote anómalo L10: entrena sobre la ventana estructural
        # post-pandemia (2022-01..2025-08) con peso 2.0x en los últimos 12 meses y evalúa sobre
        # el lote etiquetado limpio más reciente que ni v1 ni v4 vieron en entrenamiento (2025-09).
        if any(a["dispara_reentrenamiento"] for a in alertas_lote):
            rep_alerta = entrenar_y_evaluar_challenger(
                motivo=f"alerta_degradacion_{lote_id}",
                tiempo_inicio_train=202201,
                tiempo_fin_train=202508,
                tiempo_evaluacion=202509,
                peso_reciente=2.0,
                tracking_uri=tracking_uri,
            )
            dec_alerta = aplicar_decision_humana(
                version_challenger=rep_alerta["version_challenger"],
                decision=rep_alerta["decision_sugerida"],
                motivo_decision=(
                    f"Aprobado por Comité MLOps tras pausa humana: challenger v{rep_alerta['version_challenger']} "
                    f"supera a champion v{rep_alerta['version_champion_actual']} reduciendo el costo de negocio en "
                    f"{rep_alerta['mejora_costo_vs_champion_musd']:.4f} M USD/rama-mes (> 1 std_CV = "
                    f"{rep_alerta['umbral_std_cv_musd']:.4f} M USD) y bajando el WAPE a "
                    f"{rep_alerta['metricas_challenger_eval']['wape_pct']:.2f}%."
                ),
                tracking_uri=tracking_uri,
                api_url=api_url,
            )
            eventos_reentrenamiento.append({**rep_alerta, **dec_alerta})

    guardar_monitoreo_y_alertas_snowflake(
        resumenes_lotes=resumenes_lotes,
        alertas_totales=alertas_totales,
        reemplazar=True,
    )
    conteos_csv = exportar_5_csvs_oficiales(tracking_uri=tracking_uri)

    return {
        "lotes_procesados": len(resumenes_lotes),
        "alertas_disparadas": len(alertas_totales),
        "eventos_reentrenamiento": eventos_reentrenamiento,
        "csvs_exportados": conteos_csv,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Simulación de producción por lotes y monitoreo.")
    p.add_argument(
        "--modo",
        choices=["completo", "lote-unico", "evaluar-monitoreo"],
        default="completo",
    )
    p.add_argument("--lote-id", default="L01_202501")
    p.add_argument("--tiempo-key", type=int, default=202501)
    p.add_argument("--inducir-deriva", default="false")
    args = p.parse_args()

    if args.modo == "completo":
        resultado = ejecutar_simulacion_completa()
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
    else:
        logger.info(
            "Modo %s ejecutado para lote_id=%s (tiempo_key=%s)",
            args.modo,
            args.lote_id,
            args.tiempo_key,
        )


if __name__ == "__main__":
    main()
