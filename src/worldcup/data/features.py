"""Feature engineering → data/processed/.

Genera variables para el modelo de partido (especialmente el benchmark ML) y
los pesos temporales que usa Dixon-Coles.

Features candidatas:
  - Localía / partido neutral.
  - Peso temporal exp(-xi * dias_desde_partido)  → partidos recientes pesan más.
  - Forma reciente (puntos/goles en últimos N partidos).
  - Diferencia de Elo y de ranking FIFA entre los equipos.
  - (Avanzado) xG, valor de plantel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def time_weights(dates: pd.Series, reference_date: pd.Timestamp, xi: float) -> np.ndarray:
    """Peso exponencial de Dixon-Coles: exp(-xi * dias_transcurridos).

    Funcional: lo usa el modelo Dixon-Coles para ponderar la verosimilitud.
    """
    days = (reference_date - dates).dt.days.clip(lower=0).to_numpy()
    return np.exp(-xi * days)


def add_recent_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Añade forma reciente por equipo (rolling). TODO: implementar."""
    raise NotImplementedError("TODO: forma reciente (rolling) por equipo.")


def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye la tabla de features para el modelo ML. TODO: implementar."""
    raise NotImplementedError("TODO: ensamblar features para XGBoost.")
