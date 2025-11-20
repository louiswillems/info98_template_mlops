# Databricks notebook source
# MAGIC # %pip install /Workspace/Shared/.bundle/prod/artifacts/.internal/template_mlops_wine_quality-0.1.1-py3-none-any.whl
# MAGIC %pip install loguru
# MAGIC %%restart_python
import mlflow
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig, Tags
from wine_quality.models.custom_model import CustomModel

# Select prod or dev profile
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

print(mlflow.get_tracking_uri())

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "example_de_branch"})

# COMMAND ----------
# Initialize model with the config path ou en local
# custom_model = CustomModel(
#     config=config, tags=tags, spark=spark, code_paths=[
#         "../dist/wine_quality-0.1.0-py3-none-any.whl"]
# )

# Seulement avec NOTEBOOK
custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=[
        "/Workspace/Shared/.bundle/prod/artifacts/.internal/template_mlops_wine_quality-0.1.1-py3-none-any.whl"]
)

# COMMAND ----------
# Ingesting and preparing features
custom_model.load_data()
custom_model.prepare_features()

# COMMAND ----------
# Train model
custom_model.train()

# COMMAND ----------
# Log the model (runs everything including MLflow logging)
custom_model.log_model()

# COMMAND ----------
# Searching for specific run_id
run_id = mlflow.search_runs(
    experiment_names=["/Shared/wine-quality-template"]).run_id[0]
# Loading custom model
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/ian-mlops-template-custom-wine-quality-model")

# COMMAND ----------
custom_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
custom_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
custom_model.register_model()

# COMMAND ----------
