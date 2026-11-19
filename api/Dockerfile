FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias de runtime para API FastAPI, MLflow client, Snowflake y Tablero Streamlit
RUN pip install --upgrade pip && \
    pip install \
        "numpy>=1.26,<2.3" \
        "pandas>=2.2,<3.0" \
        "pyarrow>=16.0,<20.0" \
        "python-dotenv>=1.0.1" \
        "snowflake-connector-python[pandas]>=3.12" \
        "requests>=2.32.3" \
        "scipy>=1.13" \
        "plotly>=5.24" \
        "scikit-learn==1.6.1" \
        "mlflow==2.17.2" \
        "sqlalchemy>=2.0,<2.1" \
        "joblib==1.4.2" \
        "fastapi==0.115.4" \
        "uvicorn==0.32.0" \
        "pydantic==2.9.2" \
        "httpx==0.27.2" \
        "streamlit==1.40.1"

COPY src/ /app/src/
COPY api/ /app/api/
COPY tablero/ /app/tablero/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
