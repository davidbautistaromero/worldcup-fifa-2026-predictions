"""Baseline: Poisson independiente.

Goles de cada equipo ~ Poisson con tasas derivadas de parámetros de
ataque/defensa + localía. Simple y rápido; subestima los empates (de ahí
Dixon-Coles). Útil como punto de comparación.

Modelo (log-lineal):
    log(lambda_home) = mu + home_adv + atk[home] - def[away]
    log(lambda_away) = mu          + atk[away] - def[home]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from worldcup.models.base import MatchModel


class PoissonModel(MatchModel):
    def __init__(self, max_goals: int = 10) -> None:
        self.max_goals = max_goals
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.intercept: float = 0.0
        self._fitted = False

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        # TODO: ajustar por máxima verosimilitud (statsmodels GLM Poisson o
        # scipy.optimize). Estructura idéntica a dixon_coles sin la corrección rho.
        raise NotImplementedError("TODO: ajuste Poisson (GLM o MLE).")

    def _rates(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        ha = 0.0 if neutral else self.home_adv
        lam_home = np.exp(self.intercept + ha + self.attack[home] - self.defence[away])
        lam_away = np.exp(self.intercept + self.attack[away] - self.defence[home])
        return lam_home, lam_away

    def predict_scoreline(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Modelo sin ajustar: llama a fit() primero.")
        lam_home, lam_away = self._rates(home, away, neutral)
        g = np.arange(self.max_goals + 1)
        p_home = poisson.pmf(g, lam_home)
        p_away = poisson.pmf(g, lam_away)
        return np.outer(p_home, p_away)
