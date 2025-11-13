# Databricks notebook source
# Pour Notebook Databricks
# MAGIC %pip install ../dist/wine_quality-0.0.1-py3-none-any.whl
# MAGIC %pip install loguru

# COMMAND ----------
# dbutils.library.restartPython()  # noqa: ERA001 Pour Notebook Databricks
# COMMAND ----------x

import mlflow
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig, Tags
from wine_quality.models.custom_model import CustomModel

# Select prod or dev profile
mlflow.set_tracking_uri("databricks://dev")
mlflow.set_registry_uri("databricks-uc://dev")

print(mlflow.get_tracking_uri())

config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "example_de_branch"})

# COMMAND ----------
# Initialize model with the config path
custom_model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=[
        "../dist/wine_quality-0.1.0-py3-none-any.whl"]
)

# COMMAND ----------
# Predict on the test set
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.wine_quality_template_test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = custom_model.load_latest_model_and_predict(X_test)

# COMMAND ----------
