
# COMMAND ----------
# # %pip install /Workspace/Shared/.bundle/prod/artifacts/.internal/template_mlops_wine_quality-0.1.1-py3-none-any.whl
# COMMAND ----------
# %restart_python
# COMMAND ----------
# %pip list
# COMMAND ----------
# NE PAS UTILISER
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))
# COMMAND ----------
import argparse

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor

parser = argparse.ArgumentParser()

args = parser.parse_args()
root_path = args.root_path
CONFIG_PATH = f"{root_path}/files/project_config.yml"

# config = ProjectConfig.from_yaml(config_path="../project_config.yml") Local
config = ProjectConfig.from_yaml(config_path=CONFIG_PATH)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/rc_ian_uvc/wine-quality-white-and-red.csv",
    header=True,
    inferSchema=True,
).toPandas()

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess()

# Split the data
X_train, X_test = data_processor.split_data()

logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
