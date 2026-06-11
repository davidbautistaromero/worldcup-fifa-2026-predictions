"""Carga y validación de configuración (config.yaml y tournament_2026.yaml).

Funcional: usa pydantic para validar tipos. Si añades campos al YAML,
extiende los modelos aquí.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Raíz del repo = dos niveles arriba de este archivo (src/worldcup/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Carga config/config.yaml (o una ruta explícita)."""
    cfg_path = Path(path) if path else CONFIG_DIR / "config.yaml"
    cfg = _load_yaml(cfg_path)
    # Resuelve rutas relativas respecto a la raíz del proyecto.
    for key, rel in cfg.get("paths", {}).items():
        cfg["paths"][key] = str((PROJECT_ROOT / rel).resolve())
    return cfg


@lru_cache(maxsize=1)
def load_tournament(path: str | Path | None = None) -> dict[str, Any]:
    """Carga config/tournament_2026.yaml (formato del torneo)."""
    t_path = Path(path) if path else CONFIG_DIR / "tournament_2026.yaml"
    return _load_yaml(t_path)
