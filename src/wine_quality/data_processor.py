import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from wine_quality.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df"""
        # Handle missing values and convert data types as needed
        self.df["fixed_acidity"] = pd.to_numeric(self.df["fixed_acidity"], errors="coerce")

        self.df["quality"] = self.df["quality"].astype("int")

        median_no_of_previous_cancellations = self.df["alcohol"].median()

        self.df["alcohol"] = self.df["alcohol"].fillna(median_no_of_previous_cancellations)

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Fill missing values with mean or default values
        self.df = self.df.fillna(
            {
                "citric_acid": self.df["citric_acid"].mean(),
                "sulphates": 0,
            }
        )

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Extract target and relevant features
        target = self.config.target

        self.df["Id"] = range(1, len(self.df) + 1)
        relevant_columns = cat_features + num_features + [target] + ["Id"]
        self.df = self.df[relevant_columns]
        self.df["Id"] = self.df["Id"].astype("str")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )
        train_set_with_timestamp.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.wine_quality_template_train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.wine_quality_template_test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed (CDF) on the train and test tables."""
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.wine_quality_template_train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.wine_quality_template_test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
