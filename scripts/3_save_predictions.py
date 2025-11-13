import argparse

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig, Tags
from wine_quality.models.custom_model import CustomModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
# Add the missing arguments
parser.add_argument(
    "--git_sha",
    action="store",
    default="unknown",
    type=str,
    required=False,
)
parser.add_argument(
    "--branch",
    action="store",
    default="unknown",
    type=str,
    required=False,
)
parser.add_argument(
    "--job_run_id",
    action="store",
    default="unknown",
    type=str,
    required=False,
)

parser.add_argument(
    "--env",
    action="store",
    default="prd",
    type=str,
    help="Environment (dev, acc, prd)",
    required=False,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path)
spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize Custom model
custom_model = CustomModel(config=config, tags=tags, spark=spark, code_paths=[])
logger.info("Model initialized.")

# Predict on the test set - use Spark DataFrame first, then convert to pandas for prediction
test_set_spark = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
logger.info("Load data for prediction.")

X_test = test_set_spark.drop(config.target).toPandas()
logger.info("Drop target column.")

predictions_df = custom_model.load_latest_model_and_predict(X_test)
logger.info("Make predictions.")

# Save predictions to catalog
custom_model.save_predictions_to_catalog(X_test, predictions_df)
