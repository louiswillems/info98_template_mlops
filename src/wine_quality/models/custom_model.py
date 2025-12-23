from typing import List

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # noqa: N812
from pyspark.sql.functions import current_timestamp, lit, row_number, to_utc_timestamp
from pyspark.sql.window import Window
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from wine_quality.config import ProjectConfig, Tags


class WineModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model: object) -> None:
        self.model = model

    def predict(self, context, model_input: pd.DataFrame | np.ndarray) -> dict:  # noqa: ANN001
        predictions = self.model.predict(model_input)
        # looks like {"Prediction": 10000.0}
        return {"Prediction": predictions[0]}


class CustomModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: List[str]) -> None:
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self) -> None:
        """
        Load training and testing data from Delta tables.
        Splits data into:
        Features (X_train, X_test)
        Target (y_train, y_test)
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.wine_quality_template_train_set"
        )
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.wine_quality_template_test_set"
        ).toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """
        Encodes categorical features with OneHotEncoder (ignores unseen categories).
        Passes numerical features as-is (remainder='passthrough').
        Defines a pipeline combining:
            Features processing
            LightGBM regression model
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("regressor", GradientBoostingRegressor(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """
        Train the model.
        """
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """
        Log the model.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "GradientBoostingRegressor with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))

            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.wine_quality_template_train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=WineModelWrapper(self.pipeline),
                artifact_path="ian-mlops-template-custom-wine-quality-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
            )

    def register_model(self) -> None:
        """
        Register model in UC
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/ian-mlops-template-custom-wine-quality-model",
            name=f"{self.catalog_name}.{self.schema_name}.ian_mlops_template_custom_wine_quality_model",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.ian_mlops_template_custom_wine_quality_model",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> pd.DataFrame:
        """
        Retrieve MLflow run dataset.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        return dataset_source.load()
        logger.info("✅ Dataset source loaded.")

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """
        Retrieve MLflow run metadata.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        return metrics, params
        logger.info("✅ Dataset metadata loaded.")

    def model_improved(self, test_set: pd.DataFrame) -> bool:
        """
        Evaluate the model performance on the test set and compare with the latest registered model.
        Returns True if the current model performs better, False otherwise.
        """
        try:
            # Prepare test features
            X_test = test_set[self.num_features + self.cat_features]  # noqa: N806
            y_test = test_set[self.target]

            # Get predictions from the current model
            predictions_current = self.pipeline.predict(X_test)

            try:
                # Try to get predictions from the latest registered model
                predictions_latest = self.load_latest_model_and_predict(X_test)
                has_latest_model = True
            except Exception as e:
                # If there's no latest model or an error occurs
                logger.warning(f"No latest model available or error loading it: {str(e)}")
                has_latest_model = False

            logger.info("Predictions are ready.")

            # Create a DataFrame with the test set ID, actual values, and current predictions
            result_df = pd.DataFrame(
                {
                    "Id": test_set.index,
                    "Actual": y_test,
                    "prediction_current": predictions_current,
                }
            )

            # Add latest predictions if available
            if has_latest_model:
                result_df["prediction_latest"] = predictions_latest["prediction"]

            # Convert to Spark DataFrame
            result_spark_df = self.spark.createDataFrame(result_df)

            # Calculate the absolute error for current model
            result_spark_df = result_spark_df.withColumn(
                "error_current", F.abs(result_spark_df["Actual"] - result_spark_df["prediction_current"])
            )

            # Calculate MAE for current model
            mae_current = result_spark_df.agg(F.mean("error_current")).collect()[0][0]
            logger.info(f"📊 MAE for Current Model: {mae_current}")

            # If we have a latest model, compare the performance
            if has_latest_model:
                # Calculate the absolute error for latest model
                result_spark_df = result_spark_df.withColumn(
                    "error_latest", F.abs(result_spark_df["Actual"] - result_spark_df["prediction_latest"])
                )

                # Calculate MAE for latest model
                mae_latest = result_spark_df.agg(F.mean("error_latest")).collect()[0][0]
                logger.info(f"📊 MAE for Latest Model: {mae_latest}")

                # Compare models
                if mae_current < mae_latest:
                    logger.info("✅ Current Model performs better. Will register new model.")
                    return True
                else:
                    logger.info("⚠️ Current Model performs worse. Will keep the existing model.")
                    return False
            else:
                # If no latest model exists, always register the current model
                logger.info("✅ No previous model found. Will register current model.")
                return True

        except Exception as e:
            logger.error(f"Error in model_improved: {str(e)}")
            # For the first run, we want to register the model
            return True

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Load the latest model from MLflow (alias=latest-model) and make predictions.
        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = (
            f"models:/{self.catalog_name}.{self.schema_name}.ian_mlops_template_custom_wine_quality_model@latest-model"
        )
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model successfully loaded.")

        # Make predictions: None is context
        predictions = model.predict(input_data)

        # Gestion spécifique pour les prédictions de type dictionnaire
        if isinstance(predictions, dict):
            try:
                # Pour vérifier si les valeurs sont des scalaires ou des listes
                is_scalar_values = True
                for val in predictions.values():
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        is_scalar_values = False
                        break

                if is_scalar_values:
                    # Si ce sont des scalaires, créer un DataFrame d'une seule ligne
                    # et répliquer cette ligne pour chaque entrée d'origine
                    single_row_df = pd.DataFrame(predictions, index=[0])
                    predictions = pd.concat([single_row_df] * len(input_data), ignore_index=True)
                else:
                    # Sinon, conversion normale
                    predictions = pd.DataFrame(predictions)
            except Exception as e:
                logger.warning(f"Erreur lors de la conversion du dictionnaire: {e}")
                # Créer un DataFrame avec une colonne de prédiction par défaut
                predictions = pd.DataFrame({"prediction": [None] * len(input_data)})

        # Conversion générique pour les autres types (comme avant)
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions, columns=["prediction"])

        # S'assurer que le DataFrame a la bonne longueur
        if len(predictions) != len(input_data):
            logger.warning(
                f"Longueur des prédictions ({len(predictions)}) différente de celle des données d'entrée ({len(input_data)})"
            )
            # Ajuster la taille si nécessaire
            if len(predictions) == 1:
                # Répliquer la ligne unique
                predictions = pd.concat([predictions] * len(input_data), ignore_index=True)

        # Return predictions as a DataFrame
        return predictions

    def save_predictions_to_catalog(
        self, test_set: pd.DataFrame, predictions_df: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        """
        Save predictions to a table in the catalog.

        Args:
            test_set: The original test set
            predictions_df: DataFrame containing predictions
        """
        # Vérifier si predictions_df est vide
        is_empty = False

        if isinstance(predictions_df, pd.DataFrame):
            is_empty = predictions_df.empty
            logger.info(f"Le DataFrame pandas predictions_df est-il vide ? {is_empty}")

            # Afficher plus d'informations pour le débogage
            if is_empty:
                logger.warning("predictions_df est un DataFrame pandas vide")
            else:
                logger.info(
                    f"predictions_df contient {len(predictions_df)} lignes et {len(predictions_df.columns)} colonnes"
                )
                logger.info(f"Colonnes: {predictions_df.columns.tolist()}")
                logger.info(f"Premières lignes:\n{predictions_df.head(2)}")
        else:
            # Si c'est un DataFrame Spark
            try:
                count = predictions_df.count()
                is_empty = count == 0
                logger.info(f"Le DataFrame Spark predictions_df est-il vide ? {is_empty}")

                if is_empty:
                    logger.warning("predictions_df est un DataFrame Spark vide")
                else:
                    logger.info(f"predictions_df contient {count} lignes")
                    logger.info(f"Schéma: {predictions_df.schema}")
                    logger.info(f"Premières lignes:\n{predictions_df.limit(2).toPandas()}")
            except Exception as e:
                logger.error(f"Erreur lors de la vérification du DataFrame Spark: {e}")

        # Sortir de la fonction si le DataFrame est vide
        if is_empty:
            logger.warning("Impossible de sauvegarder des prédictions vides. Arrêt de la fonction.")
            return None

        # Continuer avec le reste de la fonction si le DataFrame n'est pas vide
        # S'assurer que test_set est un DataFrame Spark
        test_spark = self.spark.createDataFrame(test_set) if isinstance(test_set, pd.DataFrame) else test_set

        # S'assurer que predictions_df est un DataFrame Spark
        if isinstance(predictions_df, pd.DataFrame):
            pred_spark = self.spark.createDataFrame(predictions_df)
        else:
            pred_spark = predictions_df

        # Créer un index pour chaque DataFrame basé sur l'ordre des lignes
        windowSpec = Window.orderBy("dummy")  # noqa: N806
        test_spark_with_index = (
            test_spark.withColumn("dummy", lit(1)).withColumn("row_id", row_number().over(windowSpec)).drop("dummy")
        )

        pred_spark_with_index = (
            pred_spark.withColumn("dummy", lit(1)).withColumn("row_id", row_number().over(windowSpec)).drop("dummy")
        )

        # Joindre sur l'index de ligne
        combined_df = test_spark_with_index.join(pred_spark_with_index, on="row_id", how="left").drop("row_id")

        # Add timestamp
        combined_df_with_timestamp = combined_df.withColumn(
            "prediction_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Save to catalog
        combined_df_with_timestamp.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.prediction_table"
        )

        logger.info(f"✅ Predictions saved to {self.config.catalog_name}.{self.config.schema_name}.prediction_table")

        return combined_df_with_timestamp
