"""Backtesting honesto contra Mundiales pasados (sin fuga temporal).

Para cada torneo en config (2014/2018/2022): entrenar SOLO con partidos
anteriores a su fecha de inicio, predecir cada partido del torneo y calcular
las métricas. Comparar contra baselines (ranking FIFA puro, cuotas de apuestas).
"""
from __future__ import annotations

import pandas as pd

from worldcup.evaluation.metrics import brier_score, log_loss, ranked_probability_score
from worldcup.models.base import MatchModel


def outcome_to_class(home_score: int, away_score: int) -> int:
    """Marcador → clase {0=local, 1=empate, 2=visitante}."""
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def backtest_model(
    model_factory, matches: pd.DataFrame, tournaments: list[int]
) -> pd.DataFrame:
    """Evalúa un modelo en varios Mundiales.

    `model_factory` debe ser un callable que devuelve un MatchModel SIN ajustar
    (se reentrena por torneo con datos previos).

    TODO:
      1. Identificar los partidos de cada Mundial (por columna tournament/año).
      2. Entrenar el modelo con matches.date < inicio_del_torneo.
      3. predict_outcome de cada partido → acumular probs y resultados reales.
      4. Calcular RPS / log_loss / brier por torneo y global.
    """
    raise NotImplementedError("TODO: bucle de backtesting sin fuga temporal.")
