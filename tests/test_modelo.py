"""Pruebas automáticas de reproducibilidad del modelo y ausencia de fuga (Sección 3.2)."""
from __future__ import annotations

import json

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.datos import COLUMNAS_FUGA
from src.pipeline import COLUMNAS_MODELO, OBJETIVO, agregar_features, calcular_metricas
from tests.conftest import RUTA_FIXTURE_MODELO, RUTA_METRICAS_REF, RUTA_MUESTRA_HOLDOUT


def test_modelo_reproduce_metrica_mlflow_en_muestra_fija() -> None:
    """El modelo exportado reproduce el MAE registrado en la muestra fija dentro de 1e-4 M USD."""
    referencia = json.loads(RUTA_METRICAS_REF.read_text(encoding="utf-8"))
    muestra_df = pd.read_csv(RUTA_MUESTRA_HOLDOUT)
    muestra_df["CODIGO_INDUSTRIA"] = muestra_df["CODIGO_INDUSTRIA"].apply(lambda x: f"{int(x):03d}")

    modelo = mlflow.sklearn.load_model(str(RUTA_FIXTURE_MODELO))
    df_features = agregar_features(muestra_df)
    y_real = df_features[OBJETIVO].to_numpy(dtype=float)
    y_pred = np.clip(modelo.predict(df_features[COLUMNAS_MODELO]), 0.0, None)

    metricas = calcular_metricas(y_real, y_pred)
    tolerancia = float(referencia["tolerancia_mae_musd"])

    assert abs(metricas["mae"] - referencia["mae_muestra_fija"]) <= tolerancia, (
        f"MAE obtenido ({metricas['mae']:.6f}) difiere de la referencia "
        f"({referencia['mae_muestra_fija']:.6f}) por más de {tolerancia}"
    )
    assert metricas["wape_pct"] < 5.0, "El WAPE en la muestra hold-out debe ser inferior al 5%."


def test_consistencia_inferencia_api_vs_pipeline_directo(cliente_api: TestClient) -> None:
    """Verifica cero sesgo entrenamiento-servicio entre el pipeline directo y POST /predecir_lote."""
    muestra_df = pd.read_csv(RUTA_MUESTRA_HOLDOUT).iloc[:10].copy()
    muestra_df["CODIGO_INDUSTRIA"] = muestra_df["CODIGO_INDUSTRIA"].apply(lambda x: f"{int(x):03d}")
    muestra_df["FLAG_VENTAS_CERO_CON_DECLARANTES"] = muestra_df[
        "FLAG_VENTAS_CERO_CON_DECLARANTES"
    ].astype(bool)

    modelo = mlflow.sklearn.load_model(str(RUTA_FIXTURE_MODELO))
    df_feat = agregar_features(muestra_df)
    preds_directas = np.clip(modelo.predict(df_feat[COLUMNAS_MODELO]), 0.0, None)

    payload_lote = muestra_df.to_dict(orient="records")
    resp = cliente_api.post("/predecir_lote", json=payload_lote)
    assert resp.status_code == 200
    preds_api = [item["prediccion_ventas_musd"] for item in resp.json()["predicciones"]]

    np.testing.assert_allclose(preds_api, preds_directas, rtol=1e-4, atol=1e-3)


def test_ausencia_de_columnas_de_fuga_en_modelo() -> None:
    """Garantiza que ninguna columna prohibida post-día 15 ni el target entren al modelo."""
    for col_fuga in COLUMNAS_FUGA:
        assert col_fuga not in COLUMNAS_MODELO, f"Fuga detectada: {col_fuga} en COLUMNAS_MODELO"
    assert OBJETIVO not in COLUMNAS_MODELO
