"""Esquemas Pydantic v2 para validación de contrato de entrada y salida de la API.

Valida tipos, rangos físicos y categorías oficiales de las 46 ramas CIIU
del Formulario 104 (SRI) alineados con la firma del modelo en MLflow.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CODIGOS_INDUSTRIA_VALIDOS: set[str] = {f"{i:03d}" for i in range(1, 47)}

SECTORES_MACRO_VALIDOS: set[str] = {
    "Agricultura, ganaderia, silvicultura y pesca",
    "Comercio",
    "Construccion",
    "Electricidad y agua",
    "Manufactura",
    "Petroleo y minas",
    "Servicios empresariales y otros servicios",
    "Servicios sociales y administracion publica",
}

SECTORES_ECONOMICOS_VALIDOS: set[str] = {
    "Primario",
    "Secundario",
    "Terciario",
}

DecisionOperativa = Literal[
    "publicar_flash_expansion",
    "publicar_flash_estable",
    "alerta_contraccion_comite",
    "revisar_cobertura_manual",
]


class EntradaPrediccion(BaseModel):
    """Contrato de entrada en el corte temprano (día 15 del mes t) para una rama CIIU."""

    model_config = ConfigDict(extra="ignore")

    CODIGO_INDUSTRIA: str = Field(
        ...,
        description="Código de rama de actividad CNT-46 ('001' a '046').",
        examples=["011"],
    )
    SECTOR_MACRO: str = Field(
        ...,
        description="Macrosector económico oficial del catálogo CNT-46.",
        examples=["Manufactura"],
    )
    SECTOR_ECONOMICO: str = Field(
        ...,
        description="Sector económico agregado ('Primario', 'Secundario', 'Terciario').",
        examples=["Secundario"],
    )
    ANIO: int = Field(..., ge=2015, le=2035, description="Año fiscal de la declaración.")
    MES: int = Field(..., ge=1, le=12, description="Mes fiscal (1 a 12).")
    TRIMESTRE: int = Field(..., ge=1, le=4, description="Trimestre calendario (1 a 4).")
    SEMESTRE: int | None = Field(default=None, ge=1, le=2, description="Semestre calendario (1 o 2).")
    TIEMPO_KEY: int = Field(..., ge=201501, le=203512, description="Clave temporal AAAAMM.")
    INDICE_MES: int | None = Field(default=None, ge=1, description="Índice secuencial de mes.")

    # Variables observadas en el corte temprano (día 15 del mes t)
    VENTAS_NOMINALES_USD: float = Field(
        ..., ge=0.0, description="Ventas nominales declaradas al corte temprano (USD)."
    )
    NUM_DECLARANTES: float = Field(
        ..., ge=0.0, description="Número de contribuyentes declarantes al día 15."
    )
    VENTAS_POR_DECLARANTE_USD: float | None = Field(
        default=None, ge=0.0, description="Ventas nominales por declarante (USD)."
    )
    INDICE_PRECIO_REAL: float = Field(
        ..., gt=0.0, description="Índice de precios real sectorial CNT-46 (base 100)."
    )
    RATIO_REPRESENTATIVIDAD_HIST_12M: float = Field(
        ...,
        ge=0.0,
        le=1.5,
        description="Cobertura histórica promedio móvil de los últimos 12 meses cerrados [t-12, t-1].",
    )
    FLAG_VENTAS_CERO_CON_DECLARANTES: bool | int = Field(
        default=False,
        description="Bandera de anomalía: ventas cero con declarantes positivos.",
    )

    # Rezagos estrictamente causales conocidos al día 15 del mes t
    indice_precio_real_lag1: float | None = Field(default=None, gt=0.0)
    indice_precio_real_lag13: float | None = Field(default=None, gt=0.0)
    ventas_temprano_real_lag1_musd: float | None = Field(default=None, ge=0.0)
    ventas_temprano_real_lag12_musd: float | None = Field(default=None, ge=0.0)
    ventas_definitivo_real_lag2_musd: float | None = Field(default=None, ge=0.0)
    ventas_definitivo_real_lag12_musd: float | None = Field(default=None, ge=0.0)
    media_movil_declarantes_12m_lag1: float | None = Field(default=None, ge=0.0)

    lote_id: str | None = Field(
        default=None,
        description="Identificador del lote o período en simulación de producción.",
    )

    @field_validator("CODIGO_INDUSTRIA", mode="before")
    @classmethod
    def validar_codigo_industria(cls, valor: object) -> str:
        if not isinstance(valor, (str, int)):
            raise ValueError("CODIGO_INDUSTRIA debe ser una cadena o entero válido ('001' a '046').")
        texto = str(valor).strip()
        if texto.isdigit() and 1 <= int(texto) <= 46 and len(texto) <= 3:
            texto = f"{int(texto):03d}"
        if texto not in CODIGOS_INDUSTRIA_VALIDOS:
            raise ValueError(
                f"CODIGO_INDUSTRIA '{valor}' inválido. Debe pertenecer al catálogo '001'..'046'."
            )
        return texto

    @field_validator("SECTOR_MACRO")
    @classmethod
    def validar_sector_macro(cls, valor: str) -> str:
        texto = str(valor).strip()
        if texto not in SECTORES_MACRO_VALIDOS:
            raise ValueError(
                f"SECTOR_MACRO '{valor}' desconocido. Valores permitidos: {sorted(SECTORES_MACRO_VALIDOS)}"
            )
        return texto

    @field_validator("SECTOR_ECONOMICO")
    @classmethod
    def validar_sector_economico(cls, valor: str) -> str:
        texto = str(valor).strip()
        if texto not in SECTORES_ECONOMICOS_VALIDOS:
            raise ValueError(
                f"SECTOR_ECONOMICO '{valor}' desconocido. Valores permitidos: {sorted(SECTORES_ECONOMICOS_VALIDOS)}"
            )
        return texto


class RespuestaPrediccion(BaseModel):
    """Respuesta auditable de predicción individual de nowcasting."""

    id_prediccion: str
    timestamp_utc: str
    lote_id: str | None = None
    codigo_industria: str
    tiempo_key: int
    prediccion_ventas_musd: float
    variacion_interanual_pct: float | None
    decision: DecisionOperativa
    version_modelo: str
    alias_modelo: str
    run_id: str
    modo_respaldo: bool
    latencia_ms: float


class RespuestaLotePrediccion(BaseModel):
    """Respuesta para puntuación por lote."""

    lote_id: str | None
    total_predicciones: int
    version_modelo: str
    modo_respaldo: bool
    predicciones: list[RespuestaPrediccion]


class RespuestaSalud(BaseModel):
    """Estado de salud del servicio de inferencia."""

    estado: str
    modelo_cargado: bool
    modo_respaldo: bool
    version_modelo: str
    alias_modelo: str
    run_id: str
    mlflow_conectado: bool


class RespuestaModelo(BaseModel):
    """Metadatos y linaje completo del modelo activo en producción."""

    nombre: str
    version: str
    alias: str
    run_id: str
    modo_respaldo: bool
    motivo_respaldo: str | None = None
    snapshot_huella: str
    fecha_corte_snapshot: str
    tablas_gold_origen: list[str]


class SolicitudRespaldo(BaseModel):
    """Petición para activar o desactivar el interruptor manual de respaldo (fallback)."""

    activar: bool
    motivo: str = "Conmutacion manual por operador"


class RespuestaMetricas(BaseModel):
    """Métricas operativas de sistema y distribución de decisiones en memoria."""

    peticiones_totales: int
    peticiones_prediccion: int
    errores_4xx: int
    errores_5xx: int
    tasa_error_4xx: float
    tasa_error_5xx: float
    latencia_p50_ms: float | None
    latencia_p95_ms: float | None
    predicciones_respaldo: int
    tasa_respaldo: float
    distribucion_decisiones: dict[str, int]
    tasa_decisiones: dict[str, float]
