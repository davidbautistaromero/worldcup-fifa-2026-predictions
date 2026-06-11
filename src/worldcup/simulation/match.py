"""Resolución de un partido individual dentro de la simulación.

Usa cualquier `MatchModel` para muestrear un marcador. En eliminatorias
resuelve empates por prórroga/penales.
"""
from __future__ import annotations

import numpy as np

from worldcup.models.base import MatchModel


def simulate_group_match(
    model: MatchModel, home: str, away: str, neutral: bool, rng: np.random.Generator
) -> tuple[int, int]:
    """Marcador de un partido de fase de grupos (puede terminar en empate)."""
    return model.sample_score(home, away, neutral, rng)


def simulate_knockout_match(
    model: MatchModel, home: str, away: str, neutral: bool, rng: np.random.Generator
) -> str:
    """Devuelve el ganador de un partido eliminatorio (sin empates).

    Muestrea el marcador; si hay empate, resuelve por penales (moneda sesgada
    por la probabilidad relativa de victoria de cada equipo en tiempo regular).
    """
    h, a = model.sample_score(home, away, neutral, rng)
    if h > a:
        return home
    if a > h:
        return away
    # Empate → penales. Sesga la moneda con P(local) vs P(visitante).
    p_home, _, p_away = model.predict_outcome(home, away, neutral)
    total = p_home + p_away
    prob_home = 0.5 if total == 0 else p_home / total
    return home if rng.random() < prob_home else away
