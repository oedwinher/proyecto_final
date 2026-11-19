"""Configuración compartida de pytest y fixtures desacoplados de MLflow y Snowflake para CI."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RUTA_FIXTURE_MODELO = ROOT_DIR / "tests" / "fixtures" / "modelo_champion"
RUTA_MUESTRA_HOLDOUT = ROOT_DIR / "tests" / "fixtures" / "muestra_holdout.csv"
RUTA_METRICAS_REF = ROOT_DIR / "tests" / "fixtures" / "metricas_referencia.json"

# Configurar entorno de pruebas antes de importar api.main
os.environ["MODEL_LOCAL_PATH"] = str(RUTA_FIXTURE_MODELO)
os.environ["GUARDAR_SNOWFLAKE"] = "false"
os.environ["MODO_RESPALDO"] = "false"
os.environ["TOKEN_ADMIN"] = "token-prueba-ci"

import api.main as api_main  # noqa: E402


@pytest.fixture
def fila_entrada_valida() -> dict:
    """Devuelve un diccionario válido basado en la primera fila real de muestra_holdout.csv."""
    df = pd.read_csv(RUTA_MUESTRA_HOLDOUT)
    fila = df.iloc[0].to_dict()
    fila["CODIGO_INDUSTRIA"] = f"{int(fila['CODIGO_INDUSTRIA']):03d}"
    fila["FLAG_VENTAS_CERO_CON_DECLARANTES"] = bool(fila["FLAG_VENTAS_CERO_CON_DECLARANTES"])
    fila["lote_id"] = "2025-01"
    return fila


@pytest.fixture
def cliente_api(tmp_path: Path) -> TestClient:
    """Inicializa TestClient con el fixture local del modelo @champion y log JSONL temporal."""
    os.environ["MODEL_LOCAL_PATH"] = str(RUTA_FIXTURE_MODELO)
    os.environ["GUARDAR_SNOWFLAKE"] = "false"
    os.environ["MODO_RESPALDO"] = "false"
    os.environ["TOKEN_ADMIN"] = "token-prueba-ci"
    os.environ["PREDICCIONES_LOG_PATH"] = str(tmp_path / "predicciones_test.jsonl")

    api_main.gestor_modelo.interruptor_manual = False
    api_main.gestor_modelo.cargar()
    api_main.metricas_sistema.reiniciar()
    return TestClient(api_main.app)
