"""Servicio de predicción FastAPI para Nowcasting de Ventas Sectoriales (F104 -> IMAEC).

Cumple los requisitos de las secciones 2, 3, 4, 6 y 8 del PSet #4:
- Carga dinámica desde MLflow Model Registry por alias (`models:/nowcasting-f104-modelo@champion`)
  o desde un directorio local de fixture (`MODEL_LOCAL_PATH`) en CI/pruebas.
- Reutiliza `agregar_features`, `COLUMNAS_MODELO` y `HeuristicaBCE` desde `src.pipeline`.
- Respaldo automático si el modelo primario no carga e interruptor manual (`POST /respaldo`).
- Recarga en caliente sin reconstruir imagen Docker (`POST /recargar`) para rollback de versión.
- Middleware HTTP que mide latencias p50/p95 y tasas de error 4xx/5xx (`GET /metricas`).
- Registro auditable en JSONL local y en `PSET2_DB.ANALYTICS.PREDICCIONES` (Snowflake).
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request

from api.esquemas import (
    DecisionOperativa,
    EntradaPrediccion,
    RespuestaLotePrediccion,
    RespuestaMetricas,
    RespuestaModelo,
    RespuestaPrediccion,
    RespuestaSalud,
    SolicitudRespaldo,
)
from src.pipeline import COLUMNAS_MODELO, HeuristicaBCE, agregar_features

ROOT_DIR = Path(__file__).resolve().parent.parent
if (ROOT_DIR / ".env").exists():
    load_dotenv(ROOT_DIR / ".env", override=False)

logger = logging.getLogger("api.nowcasting")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Metadatos de trazabilidad hacia PSet #2 y PSet #3 (Sección 10.6)
SNAPSHOT_HUELLA = "e4cd092d7514"
FECHA_CORTE_SNAPSHOT = "2026-06-01"
RUN_ID_HEURISTICA_BCE = "8d013da463b2416f8bc1f507a5400818"
TABLAS_GOLD_ORIGEN = [
    "PSET2_DB.GOLD.FCT_VENTAS_MENSUALES",
    "PSET2_DB.GOLD.DIM_INDUSTRIA",
    "PSET2_DB.GOLD.DIM_TIEMPO",
    "PSET2_DB.GOLD.DIM_CORTE",
    "PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS",
    "PSET2_DB.ANALYTICS.PRECIOS_RAMA_CNT46",
]


class MetricasSistema:
    """Acumulador en memoria de métricas de sistema y distribución de decisiones."""

    def __init__(self) -> None:
        self.reiniciar()

    def reiniciar(self) -> None:
        self.latencias_ms: list[float] = []
        self.codigos_http: list[int] = []
        self.peticiones_prediccion: int = 0
        self.predicciones_respaldo: int = 0
        self.conteo_decisiones: dict[str, int] = {
            "publicar_flash_expansion": 0,
            "publicar_flash_estable": 0,
            "alerta_contraccion_comite": 0,
            "revisar_cobertura_manual": 0,
        }

    def registrar_http(self, latencia_ms: float, status_code: int, es_ruta_prediccion: bool) -> None:
        self.codigos_http.append(int(status_code))
        if es_ruta_prediccion:
            self.latencias_ms.append(float(latencia_ms))

    def registrar_prediccion(self, decision: str, modo_respaldo: bool) -> None:
        self.peticiones_prediccion += 1
        if modo_respaldo:
            self.predicciones_respaldo += 1
        self.conteo_decisiones[decision] = self.conteo_decisiones.get(decision, 0) + 1

    def resumen(self) -> RespuestaMetricas:
        total_http = len(self.codigos_http)
        err_4xx = sum(1 for c in self.codigos_http if 400 <= c < 500)
        err_5xx = sum(1 for c in self.codigos_http if c >= 500)

        if self.latencias_ms:
            p50 = round(float(np.percentile(self.latencias_ms, 50)), 3)
            p95 = round(float(np.percentile(self.latencias_ms, 95)), 3)
        else:
            p50 = None
            p95 = None

        total_pred = self.peticiones_prediccion
        tasa_decisiones = {
            k: round(v / total_pred, 4) if total_pred > 0 else 0.0
            for k, v in self.conteo_decisiones.items()
        }

        return RespuestaMetricas(
            peticiones_totales=total_http,
            peticiones_prediccion=total_pred,
            errores_4xx=err_4xx,
            errores_5xx=err_5xx,
            tasa_error_4xx=round(err_4xx / total_http, 4) if total_http > 0 else 0.0,
            tasa_error_5xx=round(err_5xx / total_http, 4) if total_http > 0 else 0.0,
            latencia_p50_ms=p50,
            latencia_p95_ms=p95,
            predicciones_respaldo=self.predicciones_respaldo,
            tasa_respaldo=round(self.predicciones_respaldo / total_pred, 4) if total_pred > 0 else 0.0,
            distribucion_decisiones=dict(self.conteo_decisiones),
            tasa_decisiones=tasa_decisiones,
        )


class GestorModelo:
    """Gestiona el modelo primario de MLflow y la conmutación con la línea base de respaldo."""

    def __init__(self) -> None:
        self.nombre_modelo: str = os.environ.get("MLFLOW_MODEL_NAME", "nowcasting-f104-modelo")
        self.alias_configurado: str = os.environ.get("MLFLOW_MODEL_ALIAS", "champion")
        self.modelo_primario: Any | None = None
        self.version_primaria: str = "desconocida"
        self.run_id_primario: str = "desconocido"
        self.mlflow_conectado: bool = False

        # Línea base del PSet #3 (HeuristicaBCE) pre-ajustada con la mediana histórica de entrenamiento
        self.modelo_respaldo = HeuristicaBCE(columna="proyeccion_cobertura_12m")
        self.modelo_respaldo.fallback_mediana_ = 213.45

        self.interruptor_manual: bool = os.environ.get("MODO_RESPALDO", "false").lower() in (
            "true",
            "1",
            "si",
            "yes",
        )
        self.fallo_carga_primario: bool = False
        self.motivo_respaldo: str | None = (
            "Activado por variable de entorno MODO_RESPALDO" if self.interruptor_manual else None
        )

    @property
    def en_modo_respaldo(self) -> bool:
        return bool(self.interruptor_manual or self.fallo_carga_primario or self.modelo_primario is None)

    @property
    def version_efectiva(self) -> str:
        if self.en_modo_respaldo:
            return "respaldo_heuristica_bce"
        return self.version_primaria

    @property
    def alias_efectivo(self) -> str:
        if self.en_modo_respaldo:
            return "fallback_baseline"
        return self.alias_configurado

    @property
    def run_id_efectivo(self) -> str:
        if self.en_modo_respaldo:
            return RUN_ID_HEURISTICA_BCE
        return self.run_id_primario

    def cargar(self) -> None:
        """Carga el modelo desde MODEL_LOCAL_PATH (si se define) o desde MLflow Model Registry por alias."""
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        self.nombre_modelo = os.environ.get("MLFLOW_MODEL_NAME", "nowcasting-f104-modelo")
        self.alias_configurado = os.environ.get("MLFLOW_MODEL_ALIAS", "champion")
        ruta_local = os.environ.get("MODEL_LOCAL_PATH", "").strip()

        if ruta_local:
            path_obj = Path(ruta_local)
            if not path_obj.is_absolute():
                path_obj = ROOT_DIR / path_obj
            if path_obj.exists():
                try:
                    self.modelo_primario = mlflow.sklearn.load_model(str(path_obj))
                    self.version_primaria = os.environ.get("MODEL_LOCAL_VERSION", "1")
                    self.run_id_primario = self._extraer_run_id_local(path_obj)
                    self.mlflow_conectado = False
                    self.fallo_carga_primario = False
                    if not self.interruptor_manual:
                        self.motivo_respaldo = None
                    logger.info(
                        "Modelo cargado desde fixture local %s (version=%s, run_id=%s)",
                        path_obj,
                        self.version_primaria,
                        self.run_id_primario,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    self._activar_respaldo_por_fallo(f"Error al cargar fixture local '{ruta_local}': {exc}")
                    return
            else:
                self._activar_respaldo_por_fallo(f"Ruta MODEL_LOCAL_PATH inexistente: '{ruta_local}'")
                return

        # Carga estándar desde servidor MLflow por alias (@champion)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)
            mv = client.get_model_version_by_alias(self.nombre_modelo, self.alias_configurado)
            model_uri = f"models:/{self.nombre_modelo}@{self.alias_configurado}"
            self.modelo_primario = mlflow.sklearn.load_model(model_uri)
            self.version_primaria = str(mv.version)
            self.run_id_primario = str(mv.run_id)
            self.mlflow_conectado = True
            self.fallo_carga_primario = False
            if not self.interruptor_manual:
                self.motivo_respaldo = None
            logger.info(
                "Modelo cargado desde MLflow %s -> version=%s, run_id=%s",
                model_uri,
                self.version_primaria,
                self.run_id_primario,
            )
        except Exception as exc:  # noqa: BLE001
            self.mlflow_conectado = False
            self._activar_respaldo_por_fallo(
                f"No se pudo cargar models:/{self.nombre_modelo}@{self.alias_configurado} desde {tracking_uri}: {exc}"
            )

    def _activar_respaldo_por_fallo(self, motivo: str) -> None:
        logger.warning("Activando respaldo automático (HeuristicaBCE): %s", motivo)
        self.modelo_primario = None
        self.fallo_carga_primario = True
        self.motivo_respaldo = motivo

    @staticmethod
    def _extraer_run_id_local(path_obj: Path) -> str:
        mlmodel_file = path_obj / "MLmodel"
        if mlmodel_file.exists():
            for line in mlmodel_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("run_id:"):
                    return line.split(":", 1)[1].strip()
        return "b4bd242a500b42eea087a6cc9918097a"

    def conmutar_respaldo_manual(self, activar: bool, motivo: str) -> None:
        self.interruptor_manual = bool(activar)
        if activar:
            self.motivo_respaldo = motivo
        else:
            if self.modelo_primario is None:
                self.cargar()
            if not self.fallo_carga_primario:
                self.motivo_respaldo = None

    def predecir_df(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[COLUMNAS_MODELO]
        if self.en_modo_respaldo:
            preds = self.modelo_respaldo.predict(X)
        else:
            preds = self.modelo_primario.predict(X)
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)


gestor_modelo = GestorModelo()
gestor_modelo.cargar()
metricas_sistema = MetricasSistema()

app = FastAPI(
    title="API de Nowcasting Sectorial F104 (SRI -> IMAEC BCE)",
    description=(
        "Servicio de inferencia MLOps (PSet #4) que estima las ventas reales definitivas "
        "de las 46 ramas CIIU a partir del corte temprano del día 15 y emite la decisión "
        "operativa para el Comité de Coyuntura del Banco Central del Ecuador."
    ),
    version="1.0.0",
)


@app.middleware("http")
async def middleware_metricas_http(request: Request, call_next):
    """Mide latencia en milisegundos y registra códigos HTTP 2xx, 4xx y 5xx."""
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    latencia_ms = (time.perf_counter() - inicio) * 1000.0
    es_prediccion = request.url.path in ("/predecir", "/predecir_lote")
    metricas_sistema.registrar_http(
        latencia_ms=latencia_ms,
        status_code=respuesta.status_code,
        es_ruta_prediccion=es_prediccion,
    )
    return respuesta


def _verificar_token_admin(x_admin_token: str | None) -> None:
    esperado = os.environ.get("TOKEN_ADMIN", "bce-coyuntura-2026")
    if not x_admin_token or x_admin_token != esperado:
        raise HTTPException(
            status_code=403,
            detail="Token administrativo inválido o ausente en header X-Admin-Token.",
        )


def _preparar_dataframe_entrada(entradas: list[EntradaPrediccion]) -> pd.DataFrame:
    """Convierte una lista de EntradaPrediccion en DataFrame y aplica agregar_features de src/pipeline.py."""
    filas = []
    for item in entradas:
        d = item.model_dump()
        if d.get("VENTAS_POR_DECLARANTE_USD") is None:
            num_decl = float(d.get("NUM_DECLARANTES") or 0.0)
            ventas_nom = float(d.get("VENTAS_NOMINALES_USD") or 0.0)
            d["VENTAS_POR_DECLARANTE_USD"] = (ventas_nom / num_decl) if num_decl > 0 else 0.0
        filas.append(d)
    df_raw = pd.DataFrame(filas)
    return agregar_features(df_raw)


def _evaluar_decision_pset1(
    entrada: EntradaPrediccion,
    prediccion_musd: float,
) -> tuple[DecisionOperativa, float | None]:
    """Aplica la regla de decisión operativa definida en el PSet #1 para el Comité de Coyuntura BCE."""
    flag_cero = bool(entrada.FLAG_VENTAS_CERO_CON_DECLARANTES)
    cobertura = float(entrada.RATIO_REPRESENTATIVIDAD_HIST_12M)

    # Cálculo de variación interanual real proyectada (%)
    base_lag12 = entrada.ventas_definitivo_real_lag12_musd
    if base_lag12 is None or base_lag12 <= 0:
        if entrada.ventas_temprano_real_lag12_musd and entrada.ventas_temprano_real_lag12_musd > 0:
            base_lag12 = float(entrada.ventas_temprano_real_lag12_musd) / max(cobertura, 0.05)

    if base_lag12 is not None and base_lag12 > 0:
        variacion_yoy = round(((prediccion_musd / float(base_lag12)) - 1.0) * 100.0, 4)
    else:
        variacion_yoy = None

    # 1. Juicio humano / bloqueo por anomalía de cobertura del corte temprano
    if flag_cero or cobertura < 0.20:
        return "revisar_cobertura_manual", variacion_yoy

    # 2. Clasificación macroeconómica según variación interanual proyectada
    if variacion_yoy is not None and variacion_yoy <= -5.0:
        return "alerta_contraccion_comite", variacion_yoy
    if variacion_yoy is not None and variacion_yoy >= 2.0:
        return "publicar_flash_expansion", variacion_yoy
    return "publicar_flash_estable", variacion_yoy


def _ruta_log_predicciones() -> Path:
    ruta = os.environ.get("PREDICCIONES_LOG_PATH", "logs/predicciones.jsonl")
    path_obj = Path(ruta)
    if not path_obj.is_absolute():
        path_obj = ROOT_DIR / path_obj
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def _guardar_registros_predicciones(
    registros_Auditoria: list[dict[str, Any]],
) -> None:
    """Persiste las predicciones en archivo JSONL local y, si está habilitado, en Snowflake."""
    ruta_log = _ruta_log_predicciones()
    with open(ruta_log, "a", encoding="utf-8") as archivo:
        for reg in registros_Auditoria:
            archivo.write(json.dumps(reg, ensure_ascii=False) + "\n")

    guardar_sf = os.environ.get("GUARDAR_SNOWFLAKE", "false").lower() in ("true", "1", "si", "yes")
    if not guardar_sf:
        return

    if not os.environ.get("SNOWFLAKE_ACCOUNT") or not os.environ.get("SNOWFLAKE_USER"):
        return

    try:
        from src.datos import _conectar_snowflake

        conn = _conectar_snowflake()
        cur = conn.cursor()
        try:
            cur.execute(
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
            sql = """
                INSERT INTO PSET2_DB.ANALYTICS.PREDICCIONES (
                    ID_PREDICCION, TIMESTAMP_UTC, LOTE_ID, CODIGO_INDUSTRIA, TIEMPO_KEY,
                    ENTRADAS_JSON, PREDICCION_VENTAS_MUSD, VARIACION_INTERANUAL_PCT, DECISION,
                    VERSION_MODELO, ALIAS_MODELO, RUN_ID, MODO_RESPALDO, LATENCIA_MS,
                    SNAPSHOT_HUELLA, TABLAS_GOLD_ORIGEN
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            filas_sf = [
                (
                    r["id_prediccion"],
                    r["timestamp_utc"],
                    r["lote_id"],
                    r["codigo_industria"],
                    int(r["tiempo_key"]),
                    json.dumps(r["entradas"], ensure_ascii=False),
                    float(r["prediccion_ventas_musd"]),
                    float(r["variacion_interanual_pct"]) if r["variacion_interanual_pct"] is not None else None,
                    r["decision"],
                    r["version_modelo"],
                    r["alias_modelo"],
                    r["run_id"],
                    bool(r["modo_respaldo"]),
                    float(r["latencia_ms"]),
                    r["snapshot_huella"],
                    json.dumps(r["tablas_gold_origen"], ensure_ascii=False),
                )
                for r in registros_Auditoria
            ]
            cur.executemany(sql, filas_sf)
        finally:
            cur.close()
            conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("No se pudo registrar en Snowflake PSET2_DB.ANALYTICS.PREDICCIONES: %s", exc)


@app.get("/salud", response_model=RespuestaSalud)
def obtener_salud() -> RespuestaSalud:
    """Devuelve el estado operativo del servicio y si el modelo primario o el respaldo están activos."""
    en_respaldo = gestor_modelo.en_modo_respaldo
    return RespuestaSalud(
        estado="degradado_respaldo" if en_respaldo else "ok",
        modelo_cargado=gestor_modelo.modelo_primario is not None,
        modo_respaldo=en_respaldo,
        version_modelo=gestor_modelo.version_efectiva,
        alias_modelo=gestor_modelo.alias_efectivo,
        run_id=gestor_modelo.run_id_efectivo,
        mlflow_conectado=gestor_modelo.mlflow_conectado,
    )


@app.get("/modelo", response_model=RespuestaModelo)
def obtener_modelo() -> RespuestaModelo:
    """Devuelve nombre, versión, alias, run_id y linaje hasta la OBT y tablas Gold."""
    return RespuestaModelo(
        nombre=gestor_modelo.nombre_modelo,
        version=gestor_modelo.version_efectiva,
        alias=gestor_modelo.alias_efectivo,
        run_id=gestor_modelo.run_id_efectivo,
        modo_respaldo=gestor_modelo.en_modo_respaldo,
        motivo_respaldo=gestor_modelo.motivo_respaldo,
        snapshot_huella=SNAPSHOT_HUELLA,
        fecha_corte_snapshot=FECHA_CORTE_SNAPSHOT,
        tablas_gold_origen=TABLAS_GOLD_ORIGEN,
    )


@app.get("/metricas", response_model=RespuestaMetricas)
def obtener_metricas() -> RespuestaMetricas:
    """Expone latencias p50/p95, volumen, tasas de error 4xx/5xx y distribución de decisiones."""
    return metricas_sistema.resumen()


@app.post("/predecir", response_model=RespuestaPrediccion)
def predecir(entrada: EntradaPrediccion) -> RespuestaPrediccion:
    """Genera una predicción validada para una rama CIIU, aplica la decisión del PSet #1 y la audita."""
    t0 = time.perf_counter()
    df_feat = _preparar_dataframe_entrada([entrada])
    pred_musd = round(float(gestor_modelo.predecir_df(df_feat)[0]), 4)
    decision, var_yoy = _evaluar_decision_pset1(entrada, pred_musd)
    latencia_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    id_pred = uuid.uuid4().hex
    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    en_respaldo = gestor_modelo.en_modo_respaldo
    version_mod = gestor_modelo.version_efectiva
    alias_mod = gestor_modelo.alias_efectivo
    run_id_mod = gestor_modelo.run_id_efectivo

    metricas_sistema.registrar_prediccion(decision=decision, modo_respaldo=en_respaldo)

    registro = {
        "id_prediccion": id_pred,
        "timestamp_utc": ts_utc,
        "lote_id": entrada.lote_id,
        "codigo_industria": entrada.CODIGO_INDUSTRIA,
        "tiempo_key": entrada.TIEMPO_KEY,
        "entradas": entrada.model_dump(mode="json"),
        "prediccion_ventas_musd": pred_musd,
        "variacion_interanual_pct": var_yoy,
        "decision": decision,
        "version_modelo": version_mod,
        "alias_modelo": alias_mod,
        "run_id": run_id_mod,
        "modo_respaldo": en_respaldo,
        "latencia_ms": latencia_ms,
        "snapshot_huella": SNAPSHOT_HUELLA,
        "tablas_gold_origen": TABLAS_GOLD_ORIGEN,
    }
    _guardar_registros_predicciones([registro])

    return RespuestaPrediccion(
        id_prediccion=id_pred,
        timestamp_utc=ts_utc,
        lote_id=entrada.lote_id,
        codigo_industria=entrada.CODIGO_INDUSTRIA,
        tiempo_key=entrada.TIEMPO_KEY,
        prediccion_ventas_musd=pred_musd,
        variacion_interanual_pct=var_yoy,
        decision=decision,
        version_modelo=version_mod,
        alias_modelo=alias_mod,
        run_id=run_id_mod,
        modo_respaldo=en_respaldo,
        latencia_ms=latencia_ms,
    )


@app.post("/predecir_lote", response_model=RespuestaLotePrediccion)
def predecir_lote(entradas: list[EntradaPrediccion]) -> RespuestaLotePrediccion:
    """Puntúa un lote de ramas CIIU en una sola llamada e inserta en bloque en Snowflake."""
    if not entradas:
        raise HTTPException(status_code=422, detail="La lista de entradas para /predecir_lote no puede estar vacía.")

    t0 = time.perf_counter()
    df_feat = _preparar_dataframe_entrada(entradas)
    preds = gestor_modelo.predecir_df(df_feat)
    latencia_total_ms = (time.perf_counter() - t0) * 1000.0
    latencia_unitaria_ms = round(latencia_total_ms / len(entradas), 3)

    ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    en_respaldo = gestor_modelo.en_modo_respaldo
    version_mod = gestor_modelo.version_efectiva
    alias_mod = gestor_modelo.alias_efectivo
    run_id_mod = gestor_modelo.run_id_efectivo

    respuestas: list[RespuestaPrediccion] = []
    registros: list[dict[str, Any]] = []

    for entrada, valor_pred in zip(entradas, preds):
        pred_musd = round(float(valor_pred), 4)
        decision, var_yoy = _evaluar_decision_pset1(entrada, pred_musd)
        id_pred = uuid.uuid4().hex
        metricas_sistema.registrar_prediccion(decision=decision, modo_respaldo=en_respaldo)

        resp = RespuestaPrediccion(
            id_prediccion=id_pred,
            timestamp_utc=ts_utc,
            lote_id=entrada.lote_id,
            codigo_industria=entrada.CODIGO_INDUSTRIA,
            tiempo_key=entrada.TIEMPO_KEY,
            prediccion_ventas_musd=pred_musd,
            variacion_interanual_pct=var_yoy,
            decision=decision,
            version_modelo=version_mod,
            alias_modelo=alias_mod,
            run_id=run_id_mod,
            modo_respaldo=en_respaldo,
            latencia_ms=latencia_unitaria_ms,
        )
        respuestas.append(resp)
        registros.append(
            {
                **resp.model_dump(),
                "entradas": entrada.model_dump(mode="json"),
                "snapshot_huella": SNAPSHOT_HUELLA,
                "tablas_gold_origen": TABLAS_GOLD_ORIGEN,
            }
        )

    _guardar_registros_predicciones(registros)
    return RespuestaLotePrediccion(
        lote_id=entradas[0].lote_id,
        total_predicciones=len(respuestas),
        version_modelo=version_mod,
        modo_respaldo=en_respaldo,
        predicciones=respuestas,
    )


@app.post("/respaldo", response_model=RespuestaModelo)
def configurar_respaldo(
    solicitud: SolicitudRespaldo,
    x_admin_token: str | None = Header(default=None),
) -> RespuestaModelo:
    """Interruptor protegido (Sección 8.4) para conmutar entre el modelo MLflow y HeuristicaBCE."""
    _verificar_token_admin(x_admin_token)
    gestor_modelo.conmutar_respaldo_manual(activar=solicitud.activar, motivo=solicitud.motivo)
    return obtener_modelo()


@app.post("/recargar", response_model=RespuestaModelo)
def recargar_modelo(
    x_admin_token: str | None = Header(default=None),
) -> RespuestaModelo:
    """Recarga en caliente la versión asociada al alias @champion en MLflow (Rollback / Promoción sin rebuild)."""
    _verificar_token_admin(x_admin_token)
    gestor_modelo.cargar()
    return obtener_modelo()
