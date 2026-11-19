"""Tablero de Monitoreo MLOps - Nowcasting Ventas Sectoriales F104 (SRI -> IMAEC BCE).

Visualiza métricas de sistema, desempeño del modelo por lote, deriva de datos (PSI / KS / Chi²),
alertas disparadas y eventos de reentrenamiento.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Monitoreo MLOps | Nowcasting F104 -> IMAEC",
    layout="wide",
)

st.title("Tablero de Monitoreo MLOps — Nowcasting Sectorial F104 (SRI -> IMAEC BCE)")
st.caption(
    "Proyecto Final (PSet #4): Servicio de inferencia, métricas de sistema, desempeño lote a lote, "
    "deriva de datos/conceptual, alertas y ciclo de reentrenamiento."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado del Servicio de Predicción (`/salud` y `/modelo`)")
    try:
        r_salud = requests.get(f"{API_URL}/salud", timeout=3).json()
        r_modelo = requests.get(f"{API_URL}/modelo", timeout=3).json()
        st.json({"salud": r_salud, "modelo": r_modelo})
    except Exception as exc:  # noqa: BLE001
        st.warning(f"API no alcanzable en {API_URL}: {exc}")

with col2:
    st.subheader("Métricas Operativas de Sistema (`/metricas`)")
    try:
        r_metricas = requests.get(f"{API_URL}/metricas", timeout=3).json()
        st.json(r_metricas)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Sin conexión en vivo a {API_URL}/metricas ({exc}).")

st.divider()

ruta_monitoreo = ROOT_DIR / "docs" / "monitoreo.csv"
ruta_alertas = ROOT_DIR / "docs" / "alertas.csv"
ruta_predicciones = ROOT_DIR / "docs" / "predicciones.csv"

if ruta_monitoreo.exists():
    st.subheader("Evolución Lote por Lote (Simulación de Producción y Monitoreo)")
    df_mon = pd.read_csv(ruta_monitoreo)
    st.dataframe(df_mon, use_container_width=True)
    if "lote_id" in df_mon.columns and "mae_lote" in df_mon.columns:
        fig = px.line(
            df_mon,
            x="lote_id",
            y="mae_lote",
            markers=True,
            title="MAE por Lote vs. Referencia CV PSet #3",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("La tabla `docs/monitoreo.csv` se poblará al ejecutar la simulación de lotes en la Fase 3.")

if ruta_alertas.exists():
    st.subheader("Registro de Alertas Operativas y de Deriva")
    st.dataframe(pd.read_csv(ruta_alertas), use_container_width=True)

if ruta_predicciones.exists():
    st.subheader("Muestra de Predicciones Auditadas (`PREDICCIONES`)")
    st.dataframe(pd.read_csv(ruta_predicciones).head(50), use_container_width=True)
