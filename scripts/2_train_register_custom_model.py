import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig, Tags
from wine_quality.models.custom_model import CustomModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"


config = ProjectConfig.from_yaml(config_path=config_path)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


# Initialize Custom model
custom_model = CustomModel(config=config, spark=spark, code_paths=[])
# custom_model = CustomModel(config=config, tags=tags, spark=spark, code_paths=[f"{args.root_path}/artifacts/.internal/wine_quality-0.0.1-py3-none-any.whl"])
logger.info("Model initialized.")


custom_model.load_data()
logger.info("Data loaded.")

custom_model.prepare_features()
logger.info("Features transformation completed.")

# Train + log the model (runs everything including MLflow logging)
custom_model.train()
logger.info("Model training completed.")

# Log the model to MLflow to get a run_id
custom_model.log_model()  # This sets self.run_id
logger.info("Model logging completed.")

# Evaluate model - Load test set from Delta table
spark = SparkSession.builder.getOrCreate()
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set").limit(100).toPandas()

model_improved = custom_model.model_improved(test_set=test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

if model_improved:
    # Register the model
    latest_version = custom_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)

else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)


custom_model.log_model()
logger.info("Model training completed.")

run_id = mlflow.search_runs(
    experiment_names=["/Shared/wine-quality-custom"]).run_id[0]
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-wine-quality-model")

# Retrieve dataset for the current run
custom_model.retrieve_current_run_dataset()

# Retrieve metadata for the current run
custom_model.retrieve_current_run_metadata()

# Register model
custom_model.register_model()
