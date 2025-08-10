FROM python:3.10-slim

# Instalar dependencias
RUN pip install --no-cache-dir zenml[server] mlflow scikit-learn pandas numpy

# Puerto MLflow
EXPOSE 5000

# Comando por defecto: correr MLflow en modo servidor
CMD ["mlflow", "models", "serve", "-m", "/model", "-h", "0.0.0.0", "-p", "5000"]
