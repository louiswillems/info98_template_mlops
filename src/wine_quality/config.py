from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    experiment_name: Optional[str]

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from a YAML file."""
        if env not in ["prd", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd' or 'dev'")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


#


class Tags(BaseModel):
    git_sha: str
    branch: str
