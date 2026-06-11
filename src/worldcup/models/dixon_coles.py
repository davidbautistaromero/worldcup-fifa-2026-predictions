"""⭐ Modelo núcleo: Dixon-Coles (1997).

Extiende el Poisson independiente con:
  - corrección `tau` (parámetro rho) que ajusta la dependencia en marcadores
    bajos (0-0, 1-0, 0-1, 1-1) → corrige la subestimación de empates;
  - ponderación temporal: partidos recientes pesan más en la verosimilitud
    (peso exp(-xi * dias), ver data/features.time_weights).

Referencia: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from worldcup.config import load_config
from worldcup.models.base import MatchModel


def dc_tau(i: np.ndarray, j: np.ndarray, lam: float, mu: float, rho: float) -> np.ndarray:
    """Factor de corrección de Dixon-Coles para marcadores bajos."""
    tau = np.ones_like(i, dtype=float)
    tau[(i == 0) & (j == 0)] = 1 - lam * mu * rho
    tau[(i == 0) & (j == 1)] = 1 + lam * rho
    tau[(i == 1) & (j == 0)] = 1 + mu * rho
    tau[(i == 1) & (j == 1)] = 1 - rho
    return tau


class DixonColesModel(MatchModel):
    def __init__(self, xi: float | None = None, max_goals: int | None = None) -> None:
        cfg = load_config()["models"]["dixon_coles"]
        self.xi = cfg["xi"] if xi is None else xi
        self.max_goals = cfg["max_goals"] if max_goals is None else max_goals
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.intercept: float = 0.0
        self.rho: float = 0.0
        self._fitted = False

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        """Ajuste por máxima verosimilitud ponderada en el tiempo.

        TODO: implementar la optimización (scipy.optimize.minimize sobre la
        log-verosimilitud negativa ponderada). Parámetros a estimar:
        attack[t], defence[t] (con restricción de identificabilidad, p. ej.
        media de attack = 0), home_adv, intercept y rho.
        Usa data.features.time_weights(date, reference_date, self.xi) como pesos.
        """
        raise NotImplementedError("TODO: MLE ponderada de Dixon-Coles.")

    def _rates(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        ha = 0.0 if neutral else self.home_adv
        lam = np.exp(self.intercept + ha + self.attack[home] - self.defence[away])
        mu = np.exp(self.intercept + self.attack[away] - self.defence[home])
        return lam, mu

    def predict_scoreline(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Modelo sin ajustar: llama a fit() primero.")
        lam, mu = self._rates(home, away, neutral)
        n = self.max_goals + 1
        g = np.arange(n)
        base = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        ii, jj = np.meshgrid(g, g, indexing="ij")
        correction = dc_tau(ii, jj, lam, mu, self.rho)
        m = base * correction
        return m / m.sum()
