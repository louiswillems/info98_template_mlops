# Databricks notebook source
# Testing on Databricks notebook
# MAGIC %pip install /Volumes/mlops_dev/louiswil/wine_quality_data/wine_quality-0.0.1-py3-none-any.whl

# COMMAND ----------

# dbutils.library.restartPython() # Databricks notebook  # noqa: ERA001

import pandas as pd
import yaml

# CHANGEMENT: Importer DatabricksSession au lieu de SparkSession
from databricks.connect import DatabricksSession
from loguru import logger

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor

# Load configuration
config = ProjectConfig.from_yaml(
    config_path="../project_config.yml", env="dev")

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# CHANGEMENT: Utiliser DatabricksSession.builder au lieu de SparkSession.builder
spark = DatabricksSession.builder.getOrCreate()

# Works both locally and in a Databricks environment
filepath = "../data/wine-quality-white-and-red.csv"
# filepath = "/Volumes/mlops_dev/louiswil/wine_quality_data/wine-quality-white-and-red.csv"  # noqa: ERA001 # Testing on Databricks notebook

# Load the data
pandas_df = pd.read_csv(filepath)

# Initialize DataProcessor
data_processor = DataProcessor(pandas_df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

data_processor.save_to_catalog(X_train, X_test)
logger.info(
    f"Les données ont été sauvegardées dans {config.catalog_name}.{config.schema_name}.wine_quality_template_train_set et {config.catalog_name}.{config.schema_name}.wine_quality_template_test_set")

# COMMAND ----------
