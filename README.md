# <h1 align="center">Projet Template MLOps End-to-End - Prédiction de la qualité du vin avec Databricks</h1>

## Configuration de l'environnement

Ce projet utilise le runtime Databricks 15.4 LTS, qui s'appuie sur Python 3.11 pour assurer une compatibilité optimale et une stabilité à long terme.


Dans nos exemples, nous utilisons UV comme gestionnaire de paquets. Consultez la documentation pour l'installation : https://docs.astral.sh/uv/getting-started/installation/

Pour créer un nouvel environnement et générer le fichier de verrouillage des dépendances :

```bash
uv sync --extra dev
```


## Dataset

Ce projet utilise le **[Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)**, un jeu de données standard pour l'analyse de la qualité du vin.

Le dataset contient des informations détaillées sur les propriétés physico-chimiques des vins (acidité, sulfates, alcool, pH, etc.) ainsi que leur score de qualité évalué par des experts. Il est utilisé pour construire des modèles de régression et de classification dans le cadre de tâches MLOps, notamment pour prédire la qualité du vin à partir de ses caractéristiques chimiques.

## Scripts du Workflow

Le workflow MLOps s'articule autour de trois scripts principaux qui s'exécutent de manière séquentielle :

- **`1_process_data.py`** : Charge et prétraite le dataset, divise les données en ensembles d'entraînement/test, et sauvegarde dans Unity Catalog
- **`2_train_register_custom_model.py`** : Effectue l'ingénierie des features, entraîne le modèle de régression et l'enregistre dans MLflow Model Registry
- **`3_save_prediction.py`** : Génère les prédictions avec le modèle entraîné et sauvegarde les résultats dans Unity Catalog

## Architecture du Projet

```
project/
├── databricks.yml                           # Orchestrateur IaC
├── pyproject.toml                           # Gestion des dépendances
├── notebooks/                               # Notebooks pour tester en local
│   ├── 1_process_data_notebook.py
│   └── 2_train_register_custom_model_notebook.py
│   ├── 3_save_predictions_notebook.py
├── config/
│   └── project_config.yaml                  # Configuration centralisée des environnements
└── scripts/                                 # Scripts d'automatisation
│   ├── 1_process_data.py
│   ├── 2_train_register_custom_model.py
│   └── 3_save_prediction.py
└── src/                                     # Code métier modulaire (package)
│   ├── wine_quality/
│       └── data_processing.py
│       └── models/
│           └── custom_model.py

```

## Développement et Déploiement

Cette architecture permet un développement hybride où les notebooks facilitent le test local avec une connexion Databricks distante (extension Databricks), tandis que les scripts assurent l'exécution automatisée en production via des workflows Databricks orchestrés avec Databricks Assets Bundles (DABs).


## Guides Complémentaires

**📋 [Guide des Premiers Pas (cliquer ici)](https://cbcradiocanada.atlassian.net/wiki/spaces/MNBI/pages/5113807122/Premiers+Pas)**
Documentation complète couvrant l'installation de VS Code, la configuration Python avec UV, Git, les standards de qualité de code, et le développement local pour Databricks.

**🚀 [Guide MLOps et Databricks Asset Bundles (cliquer ici)](https://cbcradiocanada.atlassian.net/wiki/spaces/MNBI/pages/5113708738/Pratiques+MLOps)**
Guide spécialisé explorant en profondeur la plateforme Databricks pour MLOps, l'utilisation de MLflow et les bonnes pratiques de déploiement avec les Databricks Asset Bundles (DABs).
