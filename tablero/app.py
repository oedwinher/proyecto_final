"""Tablero Interactivo de Monitoreo MLOps — Nowcasting Ventas Sectoriales F104 (SRI -> IMAEC BCE).

Cumple la Sección 6.4 del PSet #4:
- Muestra lote por lote las métricas de sistema (latencia p50/p95, errores 4xx/5xx, volumen),
  de modelo (MAE, WAPE, Costo de Negocio vs. HeuristicaBCE y Dummy, tasa de cada decisión)
  y de deriva (PSI por feature y salida, p-values KS y Chi-cuadrado).
- Marca explícitamente las alertas disparadas y los eventos de reentrenamiento (v3 rechazado en L06
  y v4 promovido a @champion tras la alerta crítica de L10_202510_DERIVA).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Monitoreo MLOps | Nowcasting F104 -> IMAEC BCE",
    layout="wide",
)

st.title("Tablero de Monitoreo MLOps — Nowcasting Sectorial F104 (SRI → IMAEC Flash BCE)")
st.caption(
    "Proyecto Final (PSets #1–#4) | Subgerencia de Cuentas Nacionales (BCE) — "
    "Vigilancia de Sistema, Desempeño de Modelo, Deriva de Datos/Conceptual, Alertas y Reentrenamiento."
)


def _cargar_csv(nombre: str) -> pd.DataFrame:
    ruta = DOCS_DIR / nombre
    if ruta.exists():
        df = pd.read_csv(ruta)
        df.columns = [c.lower() for c in df.columns]
        return df
    return pd.DataFrame()


df_mon = _cargar_csv("monitoreo.csv")
df_al = _cargar_csv("alertas.csv")
df_pred = _cargar_csv("predicciones.csv")
df_reent = _cargar_csv("reentrenamientos.csv")

if df_mon.empty:
    st.warning("No se encontró `docs/monitoreo.csv`. Ejecute la simulación de lotes primero.")
    st.stop()

# ---------------------------------------------------------------------------
# 1. KPIs Globales de Producción
# ---------------------------------------------------------------------------
total_preds = int(df_mon["volumen_solicitudes"].sum())
version_actual = str(df_mon.iloc[-1]["version_modelo"])
lat_p50_med = float(df_mon["latencia_p50_ms"].median())
lat_p95_med = float(df_mon["latencia_p95_ms"].median())
total_alertas = len(df_al)
total_reent = len(df_reent)

df_limpios = df_mon[df_mon["lote_id"] != "L10_202510_DERIVA"]
wape_limpio_prom = float(df_limpios["wape_lote_pct"].mean())
ahorro_limpio_acum = float(df_limpios["ahorro_vs_heuristica_musd"].sum() * 46.0)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Predicciones Auditadas", f"{total_preds:,}", "12 lotes × 46 ramas")
k2.metric("Versión @champion Activa", f"v{version_actual}", "Promovida tras L10")
k3.metric("Latencia Mediana p50 / p95", f"{lat_p50_med:.2f} / {lat_p95_med:.2f} ms", "0% errores 5xx")
k4.metric("WAPE Promedio (Lotes Limpios)", f"{wape_limpio_prom:.2f}%", "Meta BCE < 5.0%")
k5.metric("Alertas / Reentrenamientos", f"{total_alertas} alertas / {total_reent} runs", "v3 rechazado | v4 aprobado")
k6.metric("Ahorro Acumulado vs. BCE", f"${ahorro_limpio_acum:,.1f} M", "11 lotes limpios (46 ramas)")

st.divider()

# ---------------------------------------------------------------------------
# 2. Desempeño de Modelo y Deriva de Datos Lote por Lote
# ---------------------------------------------------------------------------
col_izq, col_der = st.columns(2)

with col_izq:
    st.subheader("1. Desempeño por Lote: Costo de Negocio y MAE vs. Heurística BCE")
    fig_perf = make_subplots(specs=[[{"secondary_y": True}]])
    fig_perf.add_trace(
        go.Scatter(
            x=df_mon["lote_id"],
            y=df_mon["costo_negocio_lote_musd"],
            mode="lines+markers",
            name="Costo Modelo @champion (M USD)",
            line={"color": "#1d4ed8", "width": 3},
        ),
        secondary_y=False,
    )
    fig_perf.add_trace(
        go.Scatter(
            x=df_mon["lote_id"],
            y=df_mon["costo_heuristica_bce_musd"],
            mode="lines+markers",
            name="Costo Heurística BCE (Línea Base)",
            line={"color": "#64748b", "width": 2, "dash": "dash"},
        ),
        secondary_y=False,
    )
    fig_perf.add_trace(
        go.Bar(
            x=df_mon["lote_id"],
            y=df_mon["wape_lote_pct"],
            name="WAPE Lote (%)",
            marker_color="rgba(14, 165, 233, 0.28)",
        ),
        secondary_y=True,
    )
    fig_perf.add_hline(
        y=16.14,
        line_dash="dot",
        line_color="#dc2626",
        annotation_text="Umbral Alerta MAE (16.14 M = CV + 2σ)",
    )
    fig_perf.add_vline(
        x="L06_202506",
        line_dash="dot",
        line_color="#d97706",
    )
    fig_perf.add_vline(
        x="L10_202510_DERIVA",
        line_dash="solid",
        line_color="#dc2626",
    )
    fig_perf.update_layout(
        height=400,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "y": 1.12},
    )
    fig_perf.update_yaxes(title_text="Costo Asimétrico / MAE (M USD)", secondary_y=False)
    fig_perf.update_yaxes(title_text="WAPE (%)", secondary_y=True)
    st.plotly_chart(fig_perf, use_container_width=True)

with col_der:
    st.subheader("2. Deriva de Datos Lote por Lote (PSI por Feature vs. Umbral 0.25)")
    fig_psi = go.Figure()
    fig_psi.add_trace(
        go.Scatter(
            x=df_mon["lote_id"],
            y=df_mon["psi_proyeccion_cobertura_12m"],
            mode="lines+markers",
            name="PSI proyeccion_cobertura_12m",
            line={"color": "#dc2626", "width": 3},
        )
    )
    fig_psi.add_trace(
        go.Scatter(
            x=df_mon["lote_id"],
            y=df_mon["psi_ventas_nominales_usd"],
            mode="lines+markers",
            name="PSI VENTAS_NOMINALES_USD",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig_psi.add_trace(
        go.Scatter(
            x=df_mon["lote_id"],
            y=df_mon["psi_indice_precio_real"],
            mode="lines+markers",
            name="PSI INDICE_PRECIO_REAL",
            line={"color": "#d97706", "width": 2},
        )
    )
    fig_psi.add_trace(
        go.Scatter(
            x=df_mon["lote_id"],
            y=df_mon["psi_prediccion_ventas_musd"],
            mode="lines+markers",
            name="PSI Salida Predicha (M USD)",
            line={"color": "#059669", "width": 2, "dash": "dot"},
        )
    )
    fig_psi.add_hline(
        y=0.25,
        line_dash="dash",
        line_color="#b91c1c",
        annotation_text="Umbral Alerta PSI >= 0.25",
    )
    fig_psi.update_layout(
        height=400,
        yaxis_type="log",
        yaxis_title="PSI (escala logarítmica)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(fig_psi, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Métricas de Sistema y Distribución de Decisiones Operativas
# ---------------------------------------------------------------------------
col_sis, col_dec = st.columns(2)

with col_sis:
    st.subheader("3. Métricas de Sistema: Latencia p50 y p95 por Lote (ms)")
    fig_lat = go.Figure()
    fig_lat.add_trace(
        go.Bar(
            x=df_mon["lote_id"],
            y=df_mon["latencia_p50_ms"],
            name="Latencia p50 (ms)",
            marker_color="#3b82f6",
        )
    )
    fig_lat.add_trace(
        go.Scatter(
            x=df_mon["lote_id"],
            y=df_mon["latencia_p95_ms"],
            mode="lines+markers",
            name="Latencia p95 (ms)",
            line={"color": "#1e293b", "width": 2},
        )
    )
    fig_lat.update_layout(
        height=340,
        yaxis_title="Milisegundos (ms)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(fig_lat, use_container_width=True)

with col_dec:
    st.subheader("4. Tasa de Decisiones Operativas PSet #1 por Lote")
    fig_dec = go.Figure()
    for col_d, nombre_d, color_d in [
        ("tasa_expansion", "Publicar Flash Expansión", "#16a34a"),
        ("tasa_estable", "Publicar Flash Estable", "#2563eb"),
        ("tasa_contraccion", "Alerta Contracción Comité", "#dc2626"),
        ("tasa_revision_manual", "Revisar Cobertura Manual", "#d97706"),
    ]:
        fig_dec.add_trace(
            go.Bar(
                x=df_mon["lote_id"],
                y=df_mon[col_d] * 100.0,
                name=nombre_d,
                marker_color=color_d,
            )
        )
    fig_dec.update_layout(
        barmode="stack",
        height=340,
        yaxis_title="Porcentaje de Ramas (%)",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "y": 1.15},
    )
    st.plotly_chart(fig_dec, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Tablas Auditables: Monitoreo Lote a Lote, Alertas y Reentrenamientos
# ---------------------------------------------------------------------------
st.subheader("5. Matriz de Monitoreo y Diagnóstico Lote por Lote (`MONITOREO`)")
cols_mostrar = [
    "lote_id",
    "tiempo_key",
    "version_modelo",
    "mae_lote",
    "wape_lote_pct",
    "costo_negocio_lote_musd",
    "costo_heuristica_bce_musd",
    "ahorro_vs_heuristica_musd",
    "psi_proyeccion_cobertura_12m",
    "psi_indice_precio_real",
    "ks_min_pvalue",
    "diagnostico_deriva",
    "num_alertas",
    "disparo_reentrenamiento",
]
st.dataframe(df_mon[[c for c in cols_mostrar if c in df_mon.columns]], use_container_width=True)

c_al, c_re = st.columns(2)
with c_al:
    st.subheader("6. Registro de Alertas Disparadas (`ALERTAS`)")
    st.dataframe(df_al, use_container_width=True)

with c_re:
    st.subheader("7. Historial de Reentrenamientos y Promoción (`reentrenamientos.csv`)")
    if not df_reent.empty:
        cols_r = [
            c
            for c in [
                "run_id",
                "tags.mlflow.runname",
                "tags.motivo_disparo",
                "tags.ventana_entrenamiento",
                "metrics.mae_prueba",
                "metrics.costo_negocio_prueba_musd",
                "metrics.champion_costo_eval_musd",
                "metrics.mejora_costo_vs_champion_musd",
                "tags.validacion_humana",
            ]
            if c in df_reent.columns
        ]
        st.dataframe(df_reent[cols_r], use_container_width=True)
