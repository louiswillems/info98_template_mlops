import pandas as pd
import pytest
from unittest.mock import MagicMock  # C'est l'outil qui va remplacer Spark

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor

# --- FIXTURES ---

@pytest.fixture
def wine_data() -> pd.DataFrame:
    """Crée un petit dataset pandas pour le test."""
    return pd.DataFrame({
        "fixed_acidity": [7.0, 7.4, 7.8, 6.9, 7.2],
        "volatile_cidity": [0.27, 0.24, 0.26, 0.30, 0.25],
        "type": ["white", "red", "white", "red", "white"],
        "alcohol": [8.8, 9.5, 10.0, 9.2, 9.8],
        "pH": [3.0, 3.1, 3.2, 3.05, 3.15],
        "density": [1.001, 0.998, 0.997, 1.002, 0.999],
        "quality": [6, 5, 7, 5, 6],
    })

@pytest.fixture
def test_config() -> ProjectConfig:
    """Crée une config valide."""
    return ProjectConfig(
        catalog_name="test_catalog",
        schema_name="test_schema",
        output_catalog="test_output",
        output_schema="test_output_schema",
        target="quality",
        cat_features=["type"],
        num_features=["alcohol", "pH", "density"],
        test_size=0.2,
        random_state=42,
        features_to_drop=["fixed_acidity", "volatile_acidity"],
        experiment_name="test_experiment",
        parameters={"learning_rate": 0.01}
    )

@pytest.fixture
def spark_session():
    """
    CRUCIAL : Ceci est un FAUX SparkSession.
    Il ne nécessite pas Java, mais permet d'initialiser DataProcessor.
    """
    mock_spark = MagicMock()
    return mock_spark


# --- TESTS ---

def test_init(wine_data, test_config, spark_session):
    """Test sans Java : on vérifie juste que l'objet stocke le mock."""
    processor = DataProcessor(wine_data, test_config, spark_session)
    # On vérifie que le dataframe interne est bien le pandas dataframe
    assert processor.df.equals(wine_data)
    # On vérifie que le processeur a bien accepté notre faux spark
    assert processor.spark == spark_session

def test_preprocess(wine_data, test_config, spark_session):
    """Test de logique pure (Pandas) ne nécessitant pas Java."""
    processor = DataProcessor(wine_data, test_config, spark_session)
    processor.preprocess()
    
    # La logique de drop se fait sur le Pandas DF, donc ça marche sans Spark
    for col in test_config.features_to_drop:
        assert col not in processor.df.columns

# def test_split_data(wine_data, test_config, spark_session):
#     """Test de split pure (Pandas/Sklearn) ne nécessitant pas Java."""
#     processor = DataProcessor(wine_data, test_config, spark_session)
#     processor.preprocess()
    
#     train, test = processor.split_data()
    
#     assert len(train) > 0
#     assert len(test) > 0
#     assert len(train) + len(test) == len(processor.df)