# Proyecto Final — Nowcasting de Ventas Sectoriales (F104 SRI $\rightarrow$ IMAEC Flash BCE)

Sistema MLOps *end-to-end* reproducible (integración de los **PSets #1, #2, #3 y #4**) para la estimación adelantada (*nowcasting*) de las ventas reales definitivas de las **46 ramas de actividad CIIU** a partir del corte preliminar del **día 15 del Formulario 104 (SRI)**, insumo directo para el **Indicador Mensual de Actividad Económica (IMAEC Flash)** de la Subgerencia de Cuentas Nacionales del **Banco Central del Ecuador (BCE)**.

> **Video de Demostración de la Infraestructura en Vivo ($\le 7$ min):** [https://1drv.ms/v/c/46f169dd809cf7ba/IQDWVUj04EjqSa4Pj7UjB6h2AdQ3XnFBagHyg8-DNrh2S3A?e=rJx7WK](https://1drv.ms/v/c/46f169dd809cf7ba/IQDWVUj04EjqSa4Pj7UjB6h2AdQ3XnFBagHyg8-DNrh2S3A?e=rJx7WK)

---

## 1. Arquitectura y Estructura Oficial del Repositorio

```text
proyecto_final/
├── docker-compose.yml                  # Orquestación unificada: Postgres (Kestra+MLflow), Kestra, Spark, MLflow, API, Tablero, dbt
├── .env.example                        # Plantilla documentada de variables de entorno (Snowflake, Kestra, MLflow SQLite/Postgres, API)
├── requirements.txt                    # Dependencias fijadas y probadas en Python 3.12 (scikit-learn==1.6.1, mlflow==2.17.2)
├── README.md                           # Guía reproducible desde cero (Fuente -> Raw -> Bronze/Silver/Gold -> OBT -> MLOps)
├── infra/
│   └── postgres-init/
│       └── create_mlflow_db.sql        # Inicialización automática de base de datos PostgreSQL compartida para MLflow
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Integración continua: pytest (14 pruebas) + docker build en cada push/PR
│       └── cd.yml                      # Despliegue continuo: publicación de imagen en GHCR (sha + latest) en merge a main
├── kestra/
│   ├── ingesta.yml                     # Pipeline Datos: Extracción F104 (URL SRI o semilla) -> RAW -> dbt (53 tests) -> Spark OBT
│   ├── produccion_lotes.yml            # Simulación de producción lote a lote respetando el horizonte de +30 días
│   ├── monitoreo.yml                   # Cálculo de métricas de sistema/modelo/datos, deriva (PSI/KS/Chi²) y alertas
│   ├── reentrenamiento.yml             # Reentrenamiento con MLflow, comparación challenger vs champion y tarea Pause humana
│   └── scripts/
│       └── load_raw_f104.py            # Script Python de carga append-only hacia PSET2_DB.RAW.RAW_F104_VENTAS
├── dbt/                                # Proyecto dbt Core (Bronze -> Silver -> Gold Star Schema + 53 pruebas de datos)
├── spark/
│   └── build_obt.py                    # Job PySpark distribuido que construye PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS
├── src/
│   ├── datos.py                        # Conexión Snowflake, cruce con PRECIOS_RAMA_CNT46 y rezagos causales sin fuga
│   ├── pipeline.py                     # Ingeniería de features día 15, ColumnTransformer, HeuristicaBCE y métricas BCE
│   └── reentrenar.py                   # Tubería CLI de reentrenamiento, regla de promoción (> 1 std_CV) y aprobación humana
├── notebooks/
│   ├── 01_eda.ipynb                    # EDA, auditoría de acceso (99.04% F104, rama 020) y contraste IPC simulado vs real
│   ├── 02_features.ipynb               # Features causales del día 15 y prueba de ablación (-68.88% MAE en Ridge)
│   ├── 03_baseline_candidatos.ipynb    # Evaluación en TimeSeriesSplit (5 pliegues) de 2 baselines y 8 candidatos (5 familias)
│   ├── 04_tuning.ipynb                 # Optimización con Optuna (90 trials: Lasso, Ridge, RandomForest) + exploración LLM
│   ├── 05_seleccion.ipynb              # Evaluación final en hold-out (2025-01..2026-06) y registro @champion v1 / @challenger v2
│   ├── 06_simulacion_produccion.ipynb  # Auditoría de los 12 lotes (552 predicciones), trazabilidad y reentrenamientos v3/v4
│   └── 07_analisis_monitoreo.ipynb     # Deriva PSI/KS/Chi², ética por subgrupos, criterios de retirada y resultado de negocio
├── api/
│   ├── Dockerfile                      # Contenedor ligero Python 3.12-slim publicado en ghcr.io/oedwinher/proyecto_final-api
│   ├── main.py                         # Servicio FastAPI (/salud, /modelo, /predecir, /predecir_lote, /metricas, /respaldo, /recargar)
│   └── esquemas.py                     # Contratos Pydantic v2 (46 ramas CIIU, rangos físicos y 4 decisiones operativas)
├── monitoreo/
│   ├── deriva.py                       # Cálculo de PSI (5 quintiles abiertos), KS, Chi², calidad de datos y matriz de diagnóstico
│   ├── alertas.py                      # Reglas de alerta (Sección 6.3) y persistencia en MONITOREO y ALERTAS de Snowflake
│   └── simular_produccion.py           # Ejecutor end-to-end de los 12 lotes, deriva inducida en L10, reentrenamientos y CSVs
├── tablero/
│   └── app.py                          # Tablero interactivo Streamlit + Plotly (puerto 8501)
├── tests/
│   ├── fixtures/                       # Modelo @champion exportado y muestra fija de 100 filas hold-out para CI sin MLflow
│   ├── test_contrato.py                # Pruebas de contrato HTTP 200 / 422, auditoría JSONL y middleware de métricas
│   ├── test_modelo.py                  # Pruebas de reproducibilidad exacta de MAE (±1e-4), cero sesgo API y cero columnas de fuga
│   └── test_comportamiento.py          # Pruebas de invariantes macroeconómicos, respaldo automático e interruptor /respaldo
└── docs/
    ├── arquitectura.png                # Diagrama de arquitectura end-to-end (Fuente -> Snowflake -> MLflow -> API -> Monitoreo)
    ├── modelo_dimensional.png          # Diagrama Star Schema Gold (FCT_VENTAS_MENSUALES + 3 dimensiones) y derivación OBT
    ├── cambios.md                      # Registro detallado de correcciones de PSets #1-#3 y decisiones actualizadas
    ├── corridas.csv                    # 107 corridas exportadas de MLflow (105 de PSet #3 + 2 de reentrenamiento)
    ├── predicciones.csv                # 552 predicciones auditadas de los 12 lotes (PSET2_DB.ANALYTICS.PREDICCIONES)
    ├── monitoreo.csv                   # 12 filas de métricas lote a lote (PSET2_DB.ANALYTICS.MONITOREO)
    ├── alertas.csv                     # 5 alertas registradas en la simulación (PSET2_DB.ANALYTICS.ALERTAS)
    ├── reentrenamientos.csv            # 2 corridas de reentrenamiento (v3 rechazada por calendario, v4 aprobada por alerta)
    ├── PSet4_memo_Ortega_Pasmay.pdf    # Documento técnico final del proyecto completo (<= 12 páginas)
    ├── PSet4_presentacion_Ortega_Pasmay.pdf # Presentación ejecutiva para el patrocinador BCE (12 diapositivas 16:9)
    └── historial/
        ├── PSet1_memo_Ortega_Pasmay.pdf
        ├── PSet1_pitch_Ortega_Pasmay.pdf
        ├── PSet2_memo_Ortega_Pasmay.pdf
        └── PSet3_memo_Ortega_Pasmay.pdf
```

---

## 2. Configuración Inicial y Conexión a Snowflake

1. **Clonar el repositorio y crear el archivo `.env`:**
   ```bash
   git clone https://github.com/oedwinher/proyecto_final.git
   cd proyecto_final
   cp .env.example .env
   ```
2. **Completar las credenciales de Snowflake en `.env`:**
   ```ini
   SNOWFLAKE_ACCOUNT=tu_organizacion-tu_cuenta
   SNOWFLAKE_USER=TU_USUARIO
   SNOWFLAKE_PASSWORD=TU_PASSWORD
   SNOWFLAKE_ROLE=ACCOUNTADMIN
   SNOWFLAKE_WAREHOUSE=COMPUTE_WH
   SNOWFLAKE_DATABASE=PSET2_DB
   SNOWFLAKE_SCHEMA=ANALYTICS
   MLFLOW_TRACKING_URI=http://localhost:5000
   MLFLOW_EXPERIMENT_NAME=nowcasting-f104-sri
   MLFLOW_MODEL_NAME=nowcasting-f104-modelo
   MLFLOW_MODEL_ALIAS=champion
   # Por defecto carga el snapshot local reproducible (107 corridas); para backend compartido PostgreSQL:
   # MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://kestra:k3str4@postgres:5432/mlflow
   MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
   SRI_SOURCE_URL=
   API_PORT=8000
   TABLERO_PORT=8501
   MODO_RESPALDO=false
   TOKEN_ADMIN=bce-coyuntura-2026
   GUARDAR_SNOWFLAKE=true
   ```
3. **Crear el entorno virtual local (Python 3.12) para pruebas y notebooks:**
   ```bash
   python -m venv .venv
   # En Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # En Linux / macOS:
   # source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 3. Orden de Ejecución Reproducible End-to-End (Sección 10)

### Paso 1: Levantar toda la infraestructura con un solo comando
```bash
docker compose up -d
```
Servicios disponibles en la red `nowcasting_net`:
* **Kestra UI:** `http://localhost:8080` (`admin@kestra.io` / `Admin1234`)
* **Spark Master Web UI:** `http://localhost:8081` (Worker en `http://localhost:8082`)
* **MLflow Tracking & Model Registry:** `http://localhost:5000`
* **API FastAPI de Inferencia (`/docs`):** `http://localhost:8000/docs`
* **Tablero Interactivo Streamlit:** `http://localhost:8501`

### Paso 2: Fuente $\rightarrow$ Raw $\rightarrow$ dbt $\rightarrow$ Spark OBT (Orquestación Completa en Kestra)
Desplegar y ejecutar el flujo `kestra/ingesta.yml` (`namespace: ecuador.bce.coyuntura`, `id: ingesta`), que extrae el Formulario 104 desde un endpoint externo del SRI (`fuente_url` / `SRI_SOURCE_URL`) o desde `data/seed/f104_ventas_46_ramas.csv` como semilla reproducible con política de reintentos (`maxAttempts: 3`) y soporte de *backfill*, poblando `PSET2_DB.RAW.RAW_F104_VENTAS` (`25,822` filas) y encadenando automáticamente las tareas `transformar_y_validar_dbt` (`dbt build`) y `construir_obt_spark` (`spark-submit`):
```bash
curl -u admin@kestra.io:Admin1234 -X POST http://localhost:8080/api/v1/main/executions/ecuador.bce.coyuntura/ingesta
```

### Paso 3: Ejecución Manual Bajo Demanda de dbt Core (Raw $\rightarrow$ Bronze $\rightarrow$ Silver $\rightarrow$ Gold)
También es posible ejecutar de forma independiente los 6 modelos y las 53 pruebas de integridad de `dbt/` (`BRONZE.STG_F104_VENTAS` con `12,911` filas $\rightarrow$ `SILVER.INT_F104_VENTAS_LIMPIAS` con `12,911` filas $\rightarrow$ Star Schema en `GOLD`: `FCT_VENTAS_MENSUALES` con `12,635` filas de las 46 ramas oficiales tras filtrar las `276` filas sin clasificar `'000'`, `DIM_INDUSTRIA` con `46` filas, `DIM_TIEMPO` con `138` filas y `DIM_CORTE` con `2` filas):
```bash
docker compose --profile jobs run --rm dbt build
```

### Paso 4: Ejecución Manual Bajo Demanda de Apache Spark (Gold $\rightarrow$ OBT)
Enviar bajo demanda el job PySpark `spark/build_obt.py` al clúster `spark://spark-master:7077` para desnormalizar el Star Schema y construir `PSET2_DB.ANALYTICS.OBT_NOWCASTING_VENTAS` (`12,635` filas × `38` columnas):
```bash
docker compose --profile jobs run --rm spark-submit
```

### Paso 5: EDA, Ingeniería de Características, Experimentación y Registro del Modelo
Ejecutar secuencialmente con **Restart & Run All** los notebooks `01` a `05` (conectados a `http://localhost:5000` y Snowflake):
```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_features.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_baseline_candidatos.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/04_tuning.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/05_seleccion.ipynb
```
Resultado en MLflow Model Registry (`nowcasting-f104-modelo`):
* **Versión 1 (`@champion` inicial):** `tuning_lasso` (`run_id = b4bd242a500b42eea087a6cc9918097a`, $\text{MAE}_{\text{CV}} = 13.92 \pm 1.11\text{ M USD}$, $\text{MAE}_{\text{holdout}} = 14.79\text{ M USD}$, $\text{WAPE}_{\text{holdout}} = 3.27\%$, $\text{Costo}_{\text{holdout}} = 18.63\text{ M USD}$ vs. $19.85\text{ M USD}$ de `HeuristicaBCE`, $\text{Error de Revisión} = 0.614\text{ pp}$ y error agregado mensual $= 0.571\%$).
* **Versión 2 (`@challenger` inicial):** `tuning_ridge` (`run_id = 2ae1ddac80d24c5d92acc83b3f4f6b01`, $\text{MAE}_{\text{holdout}} = 17.06\text{ M USD}$, $\text{WAPE}_{\text{holdout}} = 3.77\%$, $\text{Costo}_{\text{holdout}} = 22.37\text{ M USD}$).

### Paso 6: Pruebas Automáticas (`pytest`) e Integración / Despliegue Continuo (CI/CD)
Ejecutar la suite de 14 pruebas automáticas (contrato `200`/`422`, reproducibilidad del modelo sobre `tests/fixtures/`, invariantes de comportamiento macroeconómico y modo de respaldo):
```bash
pytest tests -v
```

### Paso 7: Servicio de Predicción FastAPI, Rollback de Versión e Interruptor de Respaldo
* **Verificar salud y versión del modelo activo:**
  ```bash
  curl http://localhost:8000/salud
  curl http://localhost:8000/modelo
  ```
* **Solicitar una predicción válida (`HTTP 200`):**
  ```bash
  curl -X POST http://localhost:8000/predecir \
    -H "Content-Type: application/json" \
    -d '{
      "CODIGO_INDUSTRIA": "001",
      "SECTOR_MACRO": "Agricultura, ganaderia, silvicultura y pesca",
      "SECTOR_ECONOMICO": "Primario",
      "ANIO": 2025, "MES": 1, "TRIMESTRE": 1, "SEMESTRE": 1,
      "TIEMPO_KEY": 202501, "INDICE_MES": 121,
      "VENTAS_NOMINALES_USD": 95834120.0,
      "NUM_DECLARANTES": 1420,
      "INDICE_PRECIO_REAL": 111.45,
      "RATIO_REPRESENTATIVIDAD_HIST_12M": 0.618,
      "FLAG_VENTAS_CERO_CON_DECLARANTES": false,
      "ventas_definitivo_real_lag12_musd": 101.7986
    }'
  ```
* **Activar/desactivar el interruptor de respaldo (`HeuristicaBCE`, Sección 8.4):**
  ```bash
  curl -X POST http://localhost:8000/respaldo \
    -H "Content-Type: application/json" \
    -H "X-Admin-Token: bce-coyuntura-2026" \
    -d '{"activar": true, "motivo": "Prueba operativa de respaldo"}'
  ```
* **Recargar en caliente tras rollback o promoción de alias en MLflow (sin reconstruir la imagen):**
  ```bash
  curl -X POST http://localhost:8000/recargar -H "X-Admin-Token: bce-coyuntura-2026"
  ```

### Paso 8: Simulación de Producción (12 Lotes), Monitoreo, Alertas y Reentrenamiento
Ejecutar la simulación completa de los 12 lotes del hold-out (`L01_202501` a `L12_202512`, incluyendo `L10_202510_DERIVA` con deriva inducida), el monitoreo de deriva (`PSI`, `KS`, `Chi²`), el disparo de alertas, los dos ciclos de reentrenamiento (`v3` rechazado por calendario en `L06` y `v4` aprobado tras la alerta crítica de `L10`) y la exportación de los 5 CSVs oficiales a `docs/`:
```bash
python -m monitoreo.simular_produccion --modo completo
python -m nbconvert --to notebook --execute --inplace notebooks/06_simulacion_produccion.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/07_analisis_monitoreo.ipynb
```
Finalmente, abrir el tablero interactivo en `http://localhost:8501` para inspeccionar las métricas de sistema, modelo, deriva, alertas y reentrenamientos lote por lote.
