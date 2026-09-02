from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    path = Path(config_path) if config_path else ROOT / "configs" / "default.yaml"
    if not path.is_absolute():
        candidate = ROOT / path
        path = candidate if candidate.exists() else path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, config_path=None):
    path = Path(config_path) if config_path else ROOT / "configs" / "default.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
