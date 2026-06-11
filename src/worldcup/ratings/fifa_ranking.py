"""Ranking FIFA como feature alternativa de fuerza.

Más débil que Elo por sí solo, pero útil como variable adicional para el
benchmark ML y como baseline de comparación en la evaluación.
"""
from __future__ import annotations

import pandas as pd


def latest_ranking(ranking_df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Devuelve el ranking FIFA vigente a una fecha dada (sin fuga temporal).

    TODO: filtrar por rank_date <= as_of y quedarse con el más reciente por equipo.
    """
    raise NotImplementedError("TODO: seleccionar ranking vigente a la fecha.")
