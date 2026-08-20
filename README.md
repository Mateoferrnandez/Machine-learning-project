  # 🛍️ Sales Predictor — End-to-End ML Project (MLOps)
 
**End-to-end** Machine Learning pipeline that predicts the `VENTA` variable (units sold) for clothing products based on historical data from sales campaigns, sales zones, price, size, and product subgroup. The project is built with **ZenML** (pipeline orchestration) and **MLflow** (experiment tracking and model deployment), following design patterns (Strategy, Factory) to keep the code modular and extensible.
 
---
 
## 🎯 Objective
 
Automate the full lifecycle of an ML model — from raw data ingestion to the deployment of an inference service — in a **reproducible, versioned, and monitorable** way, moving away from a "single notebook" approach and bringing the project closer to a production-grade standard.
 
---
 
## 🏗️ Project architecture
 
```
                 TRAINING PIPELINE (ml_pipeline)
┌───────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Data Ingestion │──▶│ Missing Values │──▶│ Feature Engineer.│──▶│ Outlier Detection │
└───────────────┘   └───────────────┘   └──────────────────┘   └──────────────────┘
                                                                          │
                                                                          ▼
┌───────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ MLflow Deploy  │◀──│ Model Evaluator│◀──│  Model Building   │◀──│  Data Splitter    │
└───────────────┘   └───────────────┘   └──────────────────┘   └──────────────────┘
        │
        ▼
              DEPLOYMENT / INFERENCE PIPELINE (continuous_deployment_pipeline)
┌────────────────────┐   ┌──────────────────────────┐   ┌───────────┐
│  Dynamic Importer   │──▶│ Prediction Service Loader│──▶│ Predictor │
└────────────────────┘   └──────────────────────────┘   └───────────┘
```
 
Each block in the diagram is a **ZenML step**, and each pipeline (`training_pipeline`, `deployment_pipeline`) is a **versioned DAG** that ZenML orchestrates and tracks automatically.
 
---
 
## 📂 Folder structure
 
```
├── analysis/                      # EDA (exploratory data analysis)
│   ├── basic_data_inspection.py   # Data types, dataset shape, basic statistics
│   ├── missing_value_analysis.py  # Identification and visualization of missing values
│   ├── univariate_analysis.py     # Distribution of a single variable
│   ├── bivariate_analysis.py      # Relationship between two variables
│   └── multivariate_analysis.py   # Correlations and multi-variable relationships
│
├── src/                           # Business logic (Strategy Pattern), agnostic to ZenML
│   ├── ingest_data.py             # Factory + ingestion strategies (CSV / ZIP)
│   ├── handle_missing_values.py   # Strategies: drop / mean / median / mode
│   ├── feature_engineering.py     # Strategies: log, scaling, one-hot encoding
│   ├── outlier_detection.py       # Strategies: Z-score / IQR
│   ├── data_splitter.py           # Train/test split strategy
│   ├── model_building.py          # Model building strategy (LinearRegression)
│   └── model_evaluator.py         # Evaluation strategy (MSE, R²)
│
├── steps/                         # ZenML @step wrappers around the src/ logic
│   ├── data_ingestion_step.py
│   ├── handle_missing_values_step.py
│   ├── feature_engineering_step.py
│   ├── outlier_detection_step.py
│   ├── data_splitter_step.py
│   ├── model_building_step.py     # Trains and tracks with MLflow autolog
│   ├── model_evaluator_step.py
│   ├── model_loader.py            # Loads the production model
│   ├── dynamic_importer.py        # Simulates new data for inference
│   ├── prediction_service_loader.py
│   └── predictor.py               # Calls the deployed MLflow service
│
├── pipelines/
│   ├── training_pipeline.py       # ml_pipeline: ingestion → ... → evaluation
│   └── deployment_pipeline.py     # continuous_deployment_pipeline + inference_pipeline
│
├── data/
│   └── Datosmodelo.csv            # Historical sales dataset
│
├── notebooks/
│   └── modelo.ipynb               # Exploration and initial prototyping
│
├── run_pipeline.py                # CLI: runs training
├── run_deployment.py              # CLI: runs continuous deployment + inference
├── docker-compose.yml             # Spins up the MLflow server
└── requirements.txt                # Project dependencies
```
 
---
 
## 🧠 Dataset
 
`Datosmodelo.csv` (semicolon-separated) contains historical sales data for women's clothing products across sales campaigns, with the following column groups:
 
| Group | Examples | Description |
|---|---|---|
| Identification | `CAMPANA`, `CODIGO_DEL_PRODUCTO` | Sales campaign and SKU |
| Target | `VENTA` | Units sold (variable to predict) |
| Sales by zone | `VENTA_ZONA_101` ... `VENTA_ZONA_119` | Sales broken down by geographic zone |
| Product | `NOMB_SUBGRUPO`, `TALLA`, `PRECIO_NAC` | Category, size, and national price |
| Sales force | `N° ASESORAS`, `N° ASESORAS_ZONA_xxx` | Number of sales advisors per zone |
 
---
 
## ⚙️ Tech stack
 
| Tool | Role in the project |
|---|---|
| **ZenML** | Pipeline orchestration, artifact and model versioning |
| **MLflow** | Experiment tracking (params, metrics, model) and model serving |
| **scikit-learn** | Preprocessing (`ColumnTransformer`, `Pipeline`) and model (`LinearRegression`) |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualization for the EDA |
| **Docker Compose** | Container for the MLflow tracking server |
| **Click** | Command-line interface to run the pipelines |
 
---
 
*Project built following the YouTube tutorial: [End-to-End Machine Learning Project – AI, MLOps](https://www.youtube.com/watch?v=o6vbe5G7xNo&t=9872s) — this kind of content has always helped me learn.*

