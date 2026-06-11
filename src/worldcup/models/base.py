"""Interfaz común de modelos de partido.

⭐ Pieza central de la arquitectura: cualquier modelo (Dixon-Coles, Poisson,
XGBoost) implementa este contrato, de modo que el simulador de Monte Carlo
no necesita saber qué hay por dentro. Intercambiar de modelo = cambiar una
línea de config.

`predict_scoreline` devuelve una matriz P donde P[i, j] = probabilidad de que
el local marque i goles y el visitante j. De ahí se derivan trivialmente las
probabilidades de resultado (1-X-2).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MatchModel(ABC):
    """Clase base para todos los modelos de partido."""

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> "MatchModel":
        """Ajusta el modelo con el histórico de partidos.

        Espera columnas: home_team, away_team, home_score, away_score,
        date, neutral.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_scoreline(
        self, home: str, away: str, neutral: bool = False
    ) -> np.ndarray:
        """Matriz P[i, j] = P(local marca i, visitante marca j)."""
        raise NotImplementedError

    def predict_outcome(
        self, home: str, away: str, neutral: bool = False
    ) -> tuple[float, float, float]:
        """Deriva (P(victoria local), P(empate), P(victoria visitante)).

        Implementación por defecto a partir de la matriz de marcadores;
        sirve para cualquier subclase que defina `predict_scoreline`.
        """
        m = self.predict_scoreline(home, away, neutral)
        p_home = float(np.tril(m, -1).sum())   # i > j
        p_draw = float(np.trace(m))            # i == j
        p_away = float(np.triu(m, 1).sum())    # i < j
        return p_home, p_draw, p_away

    def sample_score(
        self, home: str, away: str, neutral: bool, rng: np.random.Generator
    ) -> tuple[int, int]:
        """Muestrea un marcador concreto desde la matriz de probabilidades.

        Usado por el simulador de Monte Carlo.
        """
        m = self.predict_scoreline(home, away, neutral)
        flat = m.ravel()
        idx = rng.choice(flat.size, p=flat / flat.sum())
        return int(idx // m.shape[1]), int(idx % m.shape[1])
