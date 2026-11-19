"""Pruebas de comportamiento de dominio macroeconómico y modo de respaldo/fallback (Sección 3.3 y 3.4)."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main


def test_comportamiento_monotonicidad_ventas_tempranas(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Invariante de dominio 1: mayores ventas nominales tempranas (ceteris paribus) elevan la predicción real."""
    base = dict(fila_entrada_valida)
    incrementada = dict(fila_entrada_valida)
    incrementada["VENTAS_NOMINALES_USD"] = float(base["VENTAS_NOMINALES_USD"]) * 1.30
    if incrementada.get("VENTAS_POR_DECLARANTE_USD") is not None:
        incrementada["VENTAS_POR_DECLARANTE_USD"] = float(base["VENTAS_POR_DECLARANTE_USD"]) * 1.30

    r_base = cliente_api.post("/predecir", json=base).json()
    r_inc = cliente_api.post("/predecir", json=incrementada).json()

    assert r_inc["prediccion_ventas_musd"] > r_base["prediccion_ventas_musd"], (
        f"Se esperaba monotonicidad creciente ante +30% en ventas tempranas: "
        f"{r_base['prediccion_ventas_musd']} -> {r_inc['prediccion_ventas_musd']}"
    )


def test_comportamiento_deflactor_indice_precios(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Invariante de dominio 2: un mayor índice de precios sectorial (ceteris paribus) reduce el volumen real predicho."""
    precio_normal = dict(fila_entrada_valida)
    precio_alto = dict(fila_entrada_valida)
    precio_alto["INDICE_PRECIO_REAL"] = float(precio_normal["INDICE_PRECIO_REAL"]) * 1.25

    r_normal = cliente_api.post("/predecir", json=precio_normal).json()
    r_alto = cliente_api.post("/predecir", json=precio_alto).json()

    assert r_alto["prediccion_ventas_musd"] < r_normal["prediccion_ventas_musd"], (
        f"Un aumento del 25% en el deflactor de precios debe reducir las ventas reales estimadas: "
        f"{r_normal['prediccion_ventas_musd']} vs {r_alto['prediccion_ventas_musd']}"
    )


def test_comportamiento_reglas_decision_negocio_pset1(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Invariante de dominio 3: coherencia entre la variación interanual/cobertura y la acción operativa PSet #1."""
    pred_base = cliente_api.post("/predecir", json=fila_entrada_valida).json()["prediccion_ventas_musd"]

    # Caso A: Contracción severa frente a hace 12 meses (ventas_definitivo_real_lag12_musd mucho mayor)
    caso_contraccion = dict(fila_entrada_valida)
    caso_contraccion["ventas_definitivo_real_lag12_musd"] = pred_base * 1.25
    r_cont = cliente_api.post("/predecir", json=caso_contraccion).json()
    assert r_cont["decision"] == "alerta_contraccion_comite"
    assert r_cont["variacion_interanual_pct"] <= -5.0

    # Caso B: Expansión sólida frente a hace 12 meses (ventas_definitivo_real_lag12_musd menor)
    caso_expansion = dict(fila_entrada_valida)
    caso_expansion["ventas_definitivo_real_lag12_musd"] = max(pred_base * 0.85, 0.1)
    r_exp = cliente_api.post("/predecir", json=caso_expansion).json()
    assert r_exp["decision"] == "publicar_flash_expansion"
    assert r_exp["variacion_interanual_pct"] >= 2.0

    # Caso C: Anomalía de cobertura temprana (< 20%) dispara revisión humana manual
    caso_baja_cobertura = dict(fila_entrada_valida)
    caso_baja_cobertura["RATIO_REPRESENTATIVIDAD_HIST_12M"] = 0.12
    r_rev = cliente_api.post("/predecir", json=caso_baja_cobertura).json()
    assert r_rev["decision"] == "revisar_cobertura_manual"


def test_respaldo_automatico_si_modelo_no_carga(
    tmp_path: Path,
    fila_entrada_valida: dict,
) -> None:
    """Sección 3.4: si el modelo primario falla al cargar, el servicio responde con HeuristicaBCE e indica modo_respaldo=True."""
    os.environ["MODEL_LOCAL_PATH"] = str(tmp_path / "modelo_inexistente")
    os.environ["PREDICCIONES_LOG_PATH"] = str(tmp_path / "pred_fallback.jsonl")
    api_main.gestor_modelo.interruptor_manual = False
    api_main.gestor_modelo.cargar()

    cliente = TestClient(api_main.app)
    r_salud = cliente.get("/salud").json()
    assert r_salud["estado"] == "degradado_respaldo"
    assert r_salud["modelo_cargado"] is False
    assert r_salud["modo_respaldo"] is True
    assert r_salud["version_modelo"] == "respaldo_heuristica_bce"

    r_pred = cliente.post("/predecir", json=fila_entrada_valida)
    assert r_pred.status_code == 200
    datos_pred = r_pred.json()
    assert datos_pred["modo_respaldo"] is True
    assert datos_pred["version_modelo"] == "respaldo_heuristica_bce"
    assert datos_pred["prediccion_ventas_musd"] > 0.0


def test_interruptor_manual_respaldo_endpoint_protegido(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Sección 8.4: el endpoint protegido /respaldo conmuta en vivo entre el modelo y la regla de negocio."""
    # Sin token administrativo debe rechazar con 403
    r_403 = cliente_api.post("/respaldo", json={"activar": True, "motivo": "Prueba sin token"})
    assert r_403.status_code == 403

    # Activar modo respaldo con token válido
    headers = {"X-Admin-Token": "token-prueba-ci"}
    r_on = cliente_api.post(
        "/respaldo",
        json={"activar": True, "motivo": "Simulacion mantenimiento programado"},
        headers=headers,
    )
    assert r_on.status_code == 200
    assert r_on.json()["modo_respaldo"] is True
    assert r_on.json()["version"] == "respaldo_heuristica_bce"

    pred_respaldo = cliente_api.post("/predecir", json=fila_entrada_valida).json()
    assert pred_respaldo["modo_respaldo"] is True
    assert pred_respaldo["version_modelo"] == "respaldo_heuristica_bce"

    # Desactivar modo respaldo y verificar retorno inmediato al modelo @champion v1
    r_off = cliente_api.post(
        "/respaldo",
        json={"activar": False, "motivo": "Restauracion modelo primario"},
        headers=headers,
    )
    assert r_off.status_code == 200
    assert r_off.json()["modo_respaldo"] is False
    assert r_off.json()["version"] == "1"

    pred_primario = cliente_api.post("/predecir", json=fila_entrada_valida).json()
    assert pred_primario["modo_respaldo"] is False
    assert pred_primario["version_modelo"] == "1"
