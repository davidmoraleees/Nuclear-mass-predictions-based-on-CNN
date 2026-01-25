from pathlib import Path
import yaml

_CONFIG_CACHE = None

def load_config(path=None, reload=False):
    """Load and cache the project configuration."""
    global _CONFIG_CACHE

    if _CONFIG_CACHE is None or reload:
        if path is None:
            # project_root/config.yaml
            cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
        else:
            cfg_path = Path(path)

        with open(cfg_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)

    return _CONFIG_CACHE
