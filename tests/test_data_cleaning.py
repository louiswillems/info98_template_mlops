from unittest.mock import MagicMock

import pandas as pd
import pytest

# On garde l'import du processor, mais on n'utilise plus ProjectConfig directement
from wine_quality.data_processor import DataProcessor

# --- FIXTURES ---


@pytest.fixture
def wine_data() -> pd.DataFrame:
    """Crée un dataset complet pour éviter les KeyErrors."""
    return pd.DataFrame(
        {
            "fixed_acidity": [7.0, 7.4, 7.8, 6.9, 7.2],
            "volatile_acidity": [0.27, 0.24, 0.26, 0.30, 0.25],
            "citric_acid": [0.36, 0.34, 0.40, 0.45, 0.32],
            "residual_sugar": [20.7, 1.6, 6.9, 8.5, 12.0],
            "chlorides": [0.045, 0.049, 0.050, 0.035, 0.040],
            "free_sulfur_dioxide": [45.0, 14.0, 30.0, 20.0, 25.0],
            "total_sulfur_dioxide": [170.0, 132.0, 97.0, 100.0, 120.0],
            "sulphates": [0.45, 0.49, 0.50, 0.40, 0.42],
            "density": [1.001, 0.998, 0.997, 1.002, 0.999],
            "pH": [3.0, 3.1, 3.2, 3.05, 3.15],
            "alcohol": [8.8, 9.5, 10.0, 9.2, 9.8],
            "quality": [6, 5, 7, 5, 6],
            "type": ["white", "red", "white", "red", "white"],
        }
    )


@pytest.fixture
def test_config():
    """
    CORRECTION ICI : On utilise MagicMock au lieu de ProjectConfig.
    Cela nous permet d'ajouter 'features_to_drop' sans modifier le code source.
    """
    config = MagicMock()

    # On injecte manuellement la liste que ton code cherche
    config.features_to_drop = ["fixed_acidity", "volatile_acidity"]

    # On définit tous les autres attributs nécessaires
    config.catalog_name = "test_catalog"
    config.schema_name = "test_schema"
    config.output_catalog = "test_output"
    config.output_schema = "test_output_schema"
    config.target = "quality"
    config.cat_features = ["type"]
    config.num_features = ["alcohol", "pH", "density"]
    config.test_size = 0.2
    config.random_state = 42
    config.experiment_name = "test_experiment"
    config.parameters = {"learning_rate": 0.01}

    return config


@pytest.fixture
def spark_session():
    """Faux SparkSession pour éviter l'installation de Java."""
    return MagicMock()


# --- TESTS ---


def test_init(wine_data, test_config, spark_session):
    processor = DataProcessor(wine_data, test_config, spark_session)
    assert processor.df.equals(wine_data)
    # On vérifie juste que le processeur a accepté nos faux objets
    assert processor.config == test_config
    assert processor.spark == spark_session


def test_preprocess(wine_data, test_config, spark_session):
    processor = DataProcessor(wine_data, test_config, spark_session)
    processor.preprocess()

    # Vérifie que les colonnes ont bien été supprimées
    for col in test_config.features_to_drop:
        assert col not in processor.df.columns


def test_split_data(wine_data, test_config, spark_session):
    processor = DataProcessor(wine_data, test_config, spark_session)
    processor.preprocess()

    train, test = processor.split_data()

    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(processor.df)
