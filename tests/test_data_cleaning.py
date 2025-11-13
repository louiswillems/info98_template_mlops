import pandas as pd
import pytest
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor


@pytest.fixture
def wine_data() -> pd.DataFrame:
    """Create sample wine data for testing."""
    return pd.DataFrame(
        {
            "fixed acidity": [7.0, 7.4, 7.8, 6.9, 7.2],
            "volatile acidity": [0.27, 0.24, 0.26, 0.30, 0.25],
            "type": ["white", "red", "white", "red", "white"],
            "alcohol": [8.8, 9.5, 10.0, 9.2, 9.8],
            "pH": [3.0, 3.1, 3.2, 3.05, 3.15],
            "density": [1.001, 0.998, 0.997, 1.002, 0.999],
            "quality": [6, 5, 7, 5, 6],
        }
    )


@pytest.fixture
def test_config() -> ProjectConfig:
    """Create a simple config for testing."""
    return ProjectConfig(
        catalog_name="test_catalog",
        schema_name="test_schema",
        target_column="quality",
        test_size=0.2,
        random_state=42,
        features_to_drop=["fixed acidity", "volatile acidity"],
        categorical_features=["type"],
        numerical_features=["alcohol", "pH", "density"],
        output_catalog="test_output",
        output_schema="test_output_schema",
    )


@pytest.fixture
def spark_session() -> SparkSession:
    """Create a spark session for testing."""
    return SparkSession.builder.getOrCreate()


def test_data_processor_init(wine_data: pd.DataFrame, test_config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test initializing the DataProcessor."""
    processor = DataProcessor(wine_data, test_config, spark_session)
    assert processor.df.equals(wine_data)
    assert processor.config == test_config
    assert processor.spark == spark_session


def test_preprocess(wine_data: pd.DataFrame, test_config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test preprocessing removes specified columns."""
    processor = DataProcessor(wine_data, test_config, spark_session)
    processor.preprocess()

    # Check that dropped features are gone
    for feature in test_config.features_to_drop:
        assert feature not in processor.df.columns


def test_split_data(wine_data: pd.DataFrame, test_config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test that data is split correctly."""
    processor = DataProcessor(wine_data, test_config, spark_session)
    processor.preprocess()
    X_train, X_test = processor.split_data()  # noqa: N806

    # Check that we have train and test sets
    assert len(X_train) > 0
    assert len(X_test) > 0

    # Check total rows match original data (minus any rows dropped in preprocess)
    assert len(X_train) + len(X_test) == len(processor.df)


def test_save_to_catalog(
    wine_data: pd.DataFrame, test_config: ProjectConfig, spark_session: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that save to catalog method is called correctly."""
    # Track if save method was called
    called = [False]

    # Mock the save method
    def mock_save(self: DataProcessor, train: pd.DataFrame, test: pd.DataFrame) -> None:
        called[0] = True

    processor = DataProcessor(wine_data, test_config, spark_session)
    monkeypatch.setattr(DataProcessor, "save_to_catalog", mock_save)

    processor.preprocess()
    X_train, X_test = processor.split_data()  # noqa: N806
    processor.save_to_catalog(X_train, X_test)

    assert called[0] is True
