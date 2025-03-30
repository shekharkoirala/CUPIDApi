import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    _instance: Dict[str, Any] | None = None

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        if cls._instance is None:
            raise RuntimeError(
                "Config has not been initialized. Call load_config first."
            )
        return cls._instance

    @classmethod
    def load_config(cls, config_path: str | Path) -> Dict[str, Any]:
        if cls._instance is not None:
            return cls._instance

        with open(config_path, "r") as f:
            cls._instance = yaml.safe_load(f)
        return cls._instance
