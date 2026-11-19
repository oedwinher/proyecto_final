"""Pruebas automáticas de contrato de entrada/salida y métricas de sistema (Sección 3.1)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient


def test_salud_y_modelo_esquema(cliente_api: TestClient) -> None:
    """Verifica que /salud y /modelo respondan 200 con la versión, run_id y linaje de la OBT."""
    r_salud = cliente_api.get("/salud")
    assert r_salud.status_code == 200
    datos_salud = r_salud.json()
    assert datos_salud["estado"] == "ok"
    assert datos_salud["modelo_cargado"] is True
    assert datos_salud["modo_respaldo"] is False
    assert datos_salud["version_modelo"] == "1"
    assert datos_salud["run_id"] == "b4bd242a500b42eea087a6cc9918097a"

    r_modelo = cliente_api.get("/modelo")
    assert r_modelo.status_code == 200
    datos_modelo = r_modelo.json()
    assert datos_modelo["nombre"] == "nowcasting-f104-modelo"
    assert datos_modelo["version"] == "1"
    assert datos_modelo["alias"] == "champion"
    assert datos_modelo["snapshot_huella"] == "e4cd092d7514"
    assert "PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS" in datos_modelo["tablas_gold_origen"]


def test_predecir_entrada_valida_200_y_auditoria(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Una entrada válida devuelve 200 con el esquema completo y registra la predicción."""
    resp = cliente_api.post("/predecir", json=fila_entrada_valida)
    assert resp.status_code == 200
    cuerpo = resp.json()

    claves_esperadas = {
        "id_prediccion",
        "timestamp_utc",
        "lote_id",
        "codigo_industria",
        "tiempo_key",
        "prediccion_ventas_musd",
        "variacion_interanual_pct",
        "decision",
        "version_modelo",
        "alias_modelo",
        "run_id",
        "modo_respaldo",
        "latencia_ms",
    }
    assert claves_esperadas.issubset(cuerpo.keys())
    assert cuerpo["prediccion_ventas_musd"] > 0.0
    assert cuerpo["decision"] in (
        "publicar_flash_expansion",
        "publicar_flash_estable",
        "alerta_contraccion_comite",
        "revisar_cobertura_manual",
    )
    assert cuerpo["version_modelo"] == "1"
    assert cuerpo["modo_respaldo"] is False

    # Comprobar trazabilidad en el log JSONL
    ruta_log = Path(os.environ["PREDICCIONES_LOG_PATH"])
    assert ruta_log.exists()
    lineas = [json.loads(line) for line in ruta_log.read_text(encoding="utf-8").splitlines()]
    assert len(lineas) == 1
    assert lineas[0]["id_prediccion"] == cuerpo["id_prediccion"]
    assert lineas[0]["version_modelo"] == "1"


def test_predecir_campo_faltante_devuelve_422(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Omitir un campo obligatorio devuelve 422 indicando explícitamente el campo faltante."""
    payload = dict(fila_entrada_valida)
    payload.pop("VENTAS_NOMINALES_USD")

    resp = cliente_api.post("/predecir", json=payload)
    assert resp.status_code == 422
    texto_error = json.dumps(resp.json())
    assert "VENTAS_NOMINALES_USD" in texto_error


def test_predecir_tipo_y_rango_incorrecto_devuelve_422(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Un tipo incorrecto o valor fuera de rango físico devuelve 422 indicando el campo."""
    payload_tipo = dict(fila_entrada_valida)
    payload_tipo["VENTAS_NOMINALES_USD"] = "texto_invalido"
    r1 = cliente_api.post("/predecir", json=payload_tipo)
    assert r1.status_code == 422
    assert "VENTAS_NOMINALES_USD" in json.dumps(r1.json())

    payload_rango = dict(fila_entrada_valida)
    payload_rango["INDICE_PRECIO_REAL"] = -15.0
    r2 = cliente_api.post("/predecir", json=payload_rango)
    assert r2.status_code == 422
    assert "INDICE_PRECIO_REAL" in json.dumps(r2.json())


def test_predecir_categoria_desconocida_devuelve_422(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """Un código CIIU fuera de '001'..'046' o un sector desconocido devuelve 422."""
    payload_ciiu = dict(fila_entrada_valida)
    payload_ciiu["CODIGO_INDUSTRIA"] = "999"
    r1 = cliente_api.post("/predecir", json=payload_ciiu)
    assert r1.status_code == 422
    assert "CODIGO_INDUSTRIA" in json.dumps(r1.json())

    payload_sector = dict(fila_entrada_valida)
    payload_sector["SECTOR_MACRO"] = "Criptomineria Espacial"
    r2 = cliente_api.post("/predecir", json=payload_sector)
    assert r2.status_code == 422
    assert "SECTOR_MACRO" in json.dumps(r2.json())


def test_metricas_middleware_registra_latencias_y_errores_422(
    cliente_api: TestClient,
    fila_entrada_valida: dict,
) -> None:
    """El middleware HTTP contabiliza solicitudes exitosas, errores 422 y latencias p50/p95."""
    cliente_api.post("/predecir", json=fila_entrada_valida)
    cliente_api.post("/predecir", json={"CODIGO_INDUSTRIA": "999"})

    r_met = cliente_api.get("/metricas")
    assert r_met.status_code == 200
    met = r_met.json()
    assert met["peticiones_totales"] >= 2
    assert met["peticiones_prediccion"] == 1
    assert met["errores_4xx"] == 1
    assert met["errores_5xx"] == 0
    assert met["latencia_p50_ms"] is not None and met["latencia_p50_ms"] > 0.0
    assert met["latencia_p95_ms"] >= met["latencia_p50_ms"]
