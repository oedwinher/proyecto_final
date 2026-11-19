# Proyecto Final — Nowcasting de Ventas Sectoriales (F104 SRI $\rightarrow$ IMAEC Flash BCE)

Repositorio unificado end-to-end (PSets #1, #2, #3 y #4) para el sistema de **Nowcasting de Ventas Reales Sectoriales** de las 46 ramas de actividad CIIU (Formulario 104 del SRI) que alimenta la estimación adelantada del **IMAEC Flash** del Banco Central del Ecuador (BCE).

## Arquitectura y Estructura del Repositorio

```text
proyecto_final/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── kestra/
│   ├── ingesta.yml
│   ├── produccion_lotes.yml
│   ├── monitoreo.yml
│   └── reentrenamiento.yml
├── dbt/
├── spark/
├── src/
│   ├── datos.py
│   ├── pipeline.py
│   └── reentrenar.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_baseline_candidatos.ipynb
│   ├── 04_tuning.ipynb
│   ├── 05_seleccion.ipynb
│   ├── 06_simulacion_produccion.ipynb
│   └── 07_analisis_monitoreo.ipynb
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── esquemas.py
├── monitoreo/
│   ├── deriva.py
│   └── alertas.py
├── tablero/
│   └── app.py
├── tests/
│   ├── fixtures/
│   ├── test_contrato.py
│   ├── test_modelo.py
│   └── test_comportamiento.py
└── docs/
    └── historial/
```

## Cómo Levantar el Servicio y Ejecutar las Pruebas

1. **Configurar variables de entorno:**
   ```bash
   cp .env.example .env
   ```
2. **Ejecutar la suite de pruebas automáticas (Contrato, Modelo, Comportamiento y Respaldo):**
   ```bash
   pytest tests -v
   ```
3. **Levantar la infraestructura unificada con Docker Compose:**
   ```bash
   docker compose up -d
   ```
   - **Kestra UI:** `http://localhost:8080`
   - **Spark Master UI:** `http://localhost:8081`
   - **MLflow Tracking & Registry:** `http://localhost:5000`
   - **API de Inferencia FastAPI (`/docs`):** `http://localhost:8000/docs`
   - **Tablero de Monitoreo Streamlit:** `http://localhost:8501`
