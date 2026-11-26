-- Inicialización automática de base de datos compartida en PostgreSQL 18 para MLflow
-- (Cierre de Observación de Retroalimentación PSet #3: Backend multiusuario PostgreSQL vs SQLite).
SELECT 'CREATE DATABASE mlflow OWNER kestra'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
