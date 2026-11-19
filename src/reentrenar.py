"""Tubería de reentrenamiento, evaluación challenger vs. champion y promoción con aprobación humana (Sección 7.2).

Uso desde CLI / Kestra:
  1. Entrenar y registrar nuevo @challenger (antes de la pausa de aprobación en Kestra):
     python -m src.reentrenar \
       --motivo alerta_degradacion_L10_202510_DERIVA \
       --tiempo-inicio-train 202201 \
       --tiempo-fin-train 202508 \
       --tiempo-evaluacion 202509 \
       --peso-reciente 2.0 \
       --salida-json logs/reporte_reentrenamiento.json

  2. Aplicar decisión humana (después de la tarea Pause en Kestra):
     python -m src.reentrenar \
       --aplicar-decision aprobar \
       --version 4 \
       --motivo-decision "Aprobado por Comité MLOps: supera a champion por 1.1918 M USD (> 1 std_CV = 1.1121 M USD)"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import requests
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from monitoreo.deriva import MAE_CV_STD
from src.datos import obtener_metadatos_snapshot, preparar_universo_nowcasting
from src.pipeline import (
    COLUMNAS_MODELO,
    OBJETIVO,
    agregar_features,
    calcular_metricas,
    construir_preparacion,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if (ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("src.reentrenar")

# Hiperparámetros del modelo ganador (@champion v1) del PSet #3 (sin repetir tuning)
HP_CHAMPION_LASSO = {
    "alpha": 3.636087444439236,
    "tol": 0.00022749871012549128,
    "max_iter": 20000,
    "random_state": 42,
    "selection": "cyclic",
}


def entrenar_y_evaluar_challenger(
    motivo: str,
    tiempo_inicio_train: int = 202201,
    tiempo_fin_train: int = 202508,
    tiempo_evaluacion: int = 202509,
    peso_reciente: float = 2.0,
    meses_recientes_peso: int = 12,
    tracking_uri: str | None = None,
    nombre_experimento: str | None = None,
    nombre_modelo: str | None = None,
) -> dict[str, Any]:
    """Reentrena con la preparación de src/pipeline.py y los hiperparámetros del champion,
    registra la corrida con tag fase=reentrenamiento, asigna alias 'challenger' y evalúa
    champion vs. challenger sobre el lote etiquetado más reciente no visto en entrenamiento.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experimento = nombre_experimento or os.environ.get("MLFLOW_EXPERIMENT_NAME", "nowcasting-f104-sri")
    modelo_reg = nombre_modelo or os.environ.get("MLFLOW_MODEL_NAME", "nowcasting-f104-modelo")

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experimento)
    client = MlflowClient(tracking_uri=uri)

    # 1. Preparar datos causales (solo lotes cuyo target definitivo ya llegó)
    universo = preparar_universo_nowcasting(forzar_snowflake=False)
    datos = agregar_features(universo)

    df_train = datos[
        (datos["TIEMPO_KEY"] >= int(tiempo_inicio_train))
        & (datos["TIEMPO_KEY"] <= int(tiempo_fin_train))
    ].copy()
    df_eval = datos[datos["TIEMPO_KEY"] == int(tiempo_evaluacion)].copy()

    if df_train.empty or df_eval.empty:
        raise ValueError(
            f"Ventana de entrenamiento ({tiempo_inicio_train}..{tiempo_fin_train}) "
            f"o lote de evaluación ({tiempo_evaluacion}) vacíos."
        )

    X_train = df_train[COLUMNAS_MODELO].copy()
    y_train = df_train[OBJETIVO].to_numpy(dtype=float)
    X_eval = df_eval[COLUMNAS_MODELO].copy()
    y_eval = df_eval[OBJETIVO].to_numpy(dtype=float)

    # Ponderación por recencia (mayor peso en los últimos N meses etiquetados)
    indice_max = int(df_train["INDICE_MES"].max())
    distancia_meses = (indice_max - df_train["INDICE_MES"]).to_numpy(dtype=float)
    pesos = np.where(distancia_meses <= meses_recientes_peso, float(peso_reciente), 1.0)

    # 2. Cargar @champion actual desde Model Registry para comparación justa
    mv_champ = client.get_model_version_by_alias(modelo_reg, "champion")
    version_champion = str(mv_champ.version)
    run_id_champion = str(mv_champ.run_id)
    modelo_champion = mlflow.sklearn.load_model(f"models:/{modelo_reg}@champion")

    preds_champ_eval = np.clip(modelo_champion.predict(X_eval), 0.0, None)
    metricas_champ_eval = calcular_metricas(y_eval, preds_champ_eval)

    # 3. Entrenar nuevo candidato con la misma preparación e hiperparámetros del champion
    pipeline_challenger = Pipeline(
        steps=[
            ("preparacion", construir_preparacion(codificacion="onehot")),
            ("modelo", Lasso(**HP_CHAMPION_LASSO)),
        ]
    )
    pipeline_challenger.fit(X_train, y_train, modelo__sample_weight=pesos)

    preds_train = np.clip(pipeline_challenger.predict(X_train), 0.0, None)
    metricas_train = calcular_metricas(y_train, preds_train)

    preds_chall_eval = np.clip(pipeline_challenger.predict(X_eval), 0.0, None)
    metricas_chall_eval = calcular_metricas(y_eval, preds_chall_eval)

    # 4. Aplicar Regla de Promoción (Sección 4):
    # El challenger supera al champion en costo_negocio_musd por más de 1 desviación estándar de la CV (1.1121 M USD)
    # y mantiene WAPE <= WAPE_champion sobre la misma ventana de evaluación no vista.
    mejora_costo_musd = float(
        metricas_champ_eval["costo_negocio_musd"] - metricas_chall_eval["costo_negocio_musd"]
    )
    mejora_mae_musd = float(metricas_champ_eval["mae"] - metricas_chall_eval["mae"])
    cumple_regla_promocion = bool(
        (mejora_costo_musd > MAE_CV_STD)
        and (metricas_chall_eval["wape_pct"] <= metricas_champ_eval["wape_pct"])
    )
    decision_sugerida = "aprobar" if cumple_regla_promocion else "rechazar"

    meta_snap = obtener_metadatos_snapshot()
    ventana_desc = f"{tiempo_inicio_train}_{tiempo_fin_train}_peso_{peso_reciente}x"
    nombre_corrida = f"reentrenamiento_{motivo}_{tiempo_evaluacion}"

    # 5. Registrar corrida en MLflow con tag fase=reentrenamiento y crear versión @challenger
    with mlflow.start_run(run_name=nombre_corrida) as run:
        run_id_nuevo = run.info.run_id
        mlflow.set_tags(
            {
                "fase": "reentrenamiento",
                "motivo_disparo": motivo,
                "ventana_entrenamiento": ventana_desc,
                "lote_evaluacion": str(tiempo_evaluacion),
                "git_commit": meta_snap["git_commit"],
                "familia": "lineal_regularizado",
                "modelo_clase": "Lasso",
                "modelo_base": f"champion_v{version_champion}",
                "rol_registro": "challenger",
                "decision_regla_promocion": decision_sugerida,
            }
        )
        mlflow.log_params(
            {
                "objetivo": OBJETIVO,
                "tabla_origen": meta_snap["tabla_origen"],
                "fecha_corte_snapshot": meta_snap["fecha_corte_snapshot"],
                "huella_datos": meta_snap["huella_datos"],
                "tiempo_inicio_train": int(tiempo_inicio_train),
                "tiempo_fin_train": int(tiempo_fin_train),
                "tiempo_evaluacion": int(tiempo_evaluacion),
                "filas_entrenamiento": int(len(X_train)),
                "filas_evaluacion": int(len(X_eval)),
                "peso_reciente": float(peso_reciente),
                "meses_recientes_peso": int(meses_recientes_peso),
                "hp_alpha": HP_CHAMPION_LASSO["alpha"],
                "hp_tol": HP_CHAMPION_LASSO["tol"],
                "hp_max_iter": HP_CHAMPION_LASSO["max_iter"],
                "hp_selection": HP_CHAMPION_LASSO["selection"],
                "champion_comparado_version": version_champion,
                "umbral_promocion_std_cv_musd": round(MAE_CV_STD, 4),
            }
        )
        mlflow.log_metrics(
            {
                "mae_entrenamiento": metricas_train["mae"],
                "rmse_entrenamiento": metricas_train["rmse"],
                "wape_entrenamiento_pct": metricas_train["wape_pct"],
                "mae_prueba": metricas_chall_eval["mae"],
                "rmse_prueba": metricas_chall_eval["rmse"],
                "r2_prueba": metricas_chall_eval["r2"],
                "wape_prueba_pct": metricas_chall_eval["wape_pct"],
                "costo_negocio_prueba_musd": metricas_chall_eval["costo_negocio_musd"],
                "champion_mae_eval": metricas_champ_eval["mae"],
                "champion_wape_eval_pct": metricas_champ_eval["wape_pct"],
                "champion_costo_eval_musd": metricas_champ_eval["costo_negocio_musd"],
                "mejora_costo_vs_champion_musd": mejora_costo_musd,
                "mejora_mae_vs_champion_musd": mejora_mae_musd,
            }
        )

        ejemplo_entrada = X_train.iloc[:5].copy()
        firma = infer_signature(ejemplo_entrada, pipeline_challenger.predict(ejemplo_entrada))
        info_modelo = mlflow.sklearn.log_model(
            sk_model=pipeline_challenger,
            artifact_path="modelo",
            signature=firma,
            input_example=ejemplo_entrada,
            registered_model_name=modelo_reg,
        )
        version_nueva = str(info_modelo.registered_model_version)

    # 6. Asignar alias 'challenger' y etiquetas de comparación en el Model Registry
    client.set_registered_model_alias(modelo_reg, "challenger", version_nueva)
    client.set_model_version_tag(modelo_reg, version_nueva, "fase", "reentrenamiento")
    client.set_model_version_tag(modelo_reg, version_nueva, "motivo_disparo", motivo)
    client.set_model_version_tag(modelo_reg, version_nueva, "comparada_con_version", version_champion)
    client.set_model_version_tag(modelo_reg, version_nueva, "lote_evaluacion", str(tiempo_evaluacion))
    client.set_model_version_tag(
        modelo_reg, version_nueva, "mejora_costo_musd", f"{mejora_costo_musd:.4f}"
    )
    client.set_model_version_tag(modelo_reg, version_nueva, "validacion", "pendiente_aprobacion_humana")

    resumen = {
        "nombre_modelo": modelo_reg,
        "run_id_challenger": run_id_nuevo,
        "version_challenger": version_nueva,
        "version_champion_actual": version_champion,
        "run_id_champion_actual": run_id_champion,
        "motivo_disparo": motivo,
        "ventana_entrenamiento": ventana_desc,
        "filas_entrenamiento": int(len(X_train)),
        "tiempo_evaluacion": int(tiempo_evaluacion),
        "filas_evaluacion": int(len(X_eval)),
        "metricas_champion_eval": {k: round(v, 4) for k, v in metricas_champ_eval.items()},
        "metricas_challenger_eval": {k: round(v, 4) for k, v in metricas_chall_eval.items()},
        "mejora_costo_vs_champion_musd": round(mejora_costo_musd, 4),
        "mejora_mae_vs_champion_musd": round(mejora_mae_musd, 4),
        "umbral_std_cv_musd": round(MAE_CV_STD, 4),
        "cumple_regla_promocion": cumple_regla_promocion,
        "decision_sugerida": decision_sugerida,
    }
    logger.info("Challenger registrado (v%s): %s", version_nueva, json.dumps(resumen, ensure_ascii=False))
    return resumen


def aplicar_decision_humana(
    version_challenger: str,
    decision: str,
    motivo_decision: str,
    tracking_uri: str | None = None,
    nombre_modelo: str | None = None,
    api_url: str | None = None,
    notificar_api: bool = True,
) -> dict[str, Any]:
    """Ejecuta el paso 8 de la tubería de reentrenamiento tras la pausa de aprobación humana:
    - Si se aprueba: mueve el alias 'champion' a la versión nueva y recarga la API sin rebuild.
    - Si se rechaza: deja el 'champion' intacto y registra el motivo en los tags de MLflow.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    modelo_reg = nombre_modelo or os.environ.get("MLFLOW_MODEL_NAME", "nowcasting-f104-modelo")
    url_api = api_url or os.environ.get("API_URL", "http://localhost:8000")
    token_admin = os.environ.get("TOKEN_ADMIN", "bce-coyuntura-2026")

    client = MlflowClient(tracking_uri=uri)
    mv_champ_prev = client.get_model_version_by_alias(modelo_reg, "champion")
    version_champ_prev = str(mv_champ_prev.version)
    mv_chall = client.get_model_version(modelo_reg, str(version_challenger))

    aprobado = decision.strip().lower() in ("aprobar", "aprobada", "aprobado", "true", "si")
    estado_validacion = "aprobada" if aprobado else "rechazada"

    client.set_model_version_tag(modelo_reg, str(version_challenger), "validacion", estado_validacion)
    client.set_model_version_tag(modelo_reg, str(version_challenger), "motivo_decision", motivo_decision)
    client.set_tag(mv_chall.run_id, "validacion_humana", estado_validacion)
    client.set_tag(mv_chall.run_id, "motivo_decision_humana", motivo_decision)

    if aprobado:
        client.set_registered_model_alias(modelo_reg, "champion", str(version_challenger))
        client.set_tag(mv_chall.run_id, "rol_registro", "champion")
        version_champion_final = str(version_challenger)
        logger.info(
            "PROMOCIÓN APROBADA: %s@champion movido de v%s -> v%s",
            modelo_reg,
            version_champ_prev,
            version_champion_final,
        )
    else:
        version_champion_final = version_champ_prev
        logger.info(
            "PROMOCIÓN RECHAZADA para v%s (%s). %s@champion permanece intacto en v%s.",
            version_challenger,
            motivo_decision,
            modelo_reg,
            version_champion_final,
        )

    api_recargada = False
    if notificar_api:
        try:
            r = requests.post(
                f"{url_api}/recargar",
                headers={"X-Admin-Token": token_admin},
                timeout=15,
            )
            api_recargada = r.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo invocar POST %s/recargar: %s", url_api, exc)

    return {
        "version_evaluada": str(version_challenger),
        "decision_aplicada": estado_validacion,
        "motivo_decision": motivo_decision,
        "version_champion_anterior": version_champ_prev,
        "version_champion_actual": version_champion_final,
        "api_recargada": api_recargada,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reentrenamiento y promoción MLOps con MLflow.")
    p.add_argument("--motivo", default="alerta_degradacion_desempeno")
    p.add_argument("--tiempo-inicio-train", type=int, default=202201)
    p.add_argument("--tiempo-fin-train", type=int, default=202508)
    p.add_argument("--tiempo-evaluacion", type=int, default=202509)
    p.add_argument("--peso-reciente", type=float, default=2.0)
    p.add_argument("--salida-json", default=None)
    p.add_argument("--aplicar-decision", choices=["aprobar", "rechazar"], default=None)
    p.add_argument("--version", default=None)
    p.add_argument("--motivo-decision", default="Revisión manual por Comité Técnico MLOps")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.aplicar_decision:
        if not args.version:
            raise SystemExit("Debe especificar --version al usar --aplicar-decision.")
        res = aplicar_decision_humana(
            version_challenger=str(args.version),
            decision=args.aplicar_decision,
            motivo_decision=args.motivo_decision,
        )
    else:
        res = entrenar_y_evaluar_challenger(
            motivo=args.motivo,
            tiempo_inicio_train=args.tiempo_inicio_train,
            tiempo_fin_train=args.tiempo_fin_train,
            tiempo_evaluacion=args.tiempo_evaluacion,
            peso_reciente=args.peso_reciente,
        )

    if args.salida_json:
        ruta = Path(args.salida_json)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
