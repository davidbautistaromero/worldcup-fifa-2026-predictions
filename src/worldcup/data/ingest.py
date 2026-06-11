"""Ingesta de datos crudos.

Dataset principal sugerido: "International football results from 1872 to 2024"
(Kaggle / RSSSF). Descárgalo manualmente o vía la API de Kaggle y colócalo en
data/raw/ con el nombre indicado en config.yaml (data.results_csv).

Esquema esperado de results.csv:
    date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup.config import load_config


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el CSV de resultados internacionales históricos."""
    cfg = load_config()
    csv = Path(path) if path else Path(cfg["paths"]["data_raw"]) / cfg["data"]["results_csv"]
    if not csv.exists():
        raise FileNotFoundError(
            f"No se encontró {csv}. Descarga el dataset de resultados y colócalo ahí. "
            "Ver docstring de este módulo."
        )
    df = pd.read_csv(csv, parse_dates=["date"])
    return df


def load_fifa_ranking(path: str | Path | None = None) -> pd.DataFrame:
    """Carga el ranking FIFA histórico (opcional, como feature alternativa)."""
    cfg = load_config()
    csv = Path(path) if path else Path(cfg["paths"]["data_raw"]) / cfg["data"]["fifa_ranking_csv"]
    if not csv.exists():
        raise FileNotFoundError(f"No se encontró {csv}.")
    return pd.read_csv(csv, parse_dates=["rank_date"])
