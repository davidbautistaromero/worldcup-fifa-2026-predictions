"""Métricas para probabilidades de resultado (1-X-2).

Funcional. Convención de orden de clases: columna 0=local, 1=empate, 2=visitante.
  - RPS (Ranked Probability Score): métrica principal. Respeta el orden de las
    categorías → penaliza menos errar empate↔victoria que invertir local↔visitante.
    Menor es mejor.
  - log_loss y brier: complementarias.
"""
from __future__ import annotations

import numpy as np

N_CLASSES = 3  # home, draw, away


def _as_2d(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    return probs[None, :] if probs.ndim == 1 else probs


def _onehot(outcomes: np.ndarray) -> np.ndarray:
    """outcomes: enteros en {0,1,2} → matriz one-hot (n, 3)."""
    outcomes = np.asarray(outcomes, dtype=int)
    oh = np.zeros((outcomes.size, N_CLASSES))
    oh[np.arange(outcomes.size), outcomes] = 1.0
    return oh


def ranked_probability_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """RPS promedio. probs: (n, 3) o (3,); outcomes: enteros {0,1,2}."""
    p = _as_2d(probs)
    o = _onehot(outcomes)
    cum_p = np.cumsum(p, axis=1)
    cum_o = np.cumsum(o, axis=1)
    # Suma sobre las r-1 primeras categorías acumuladas.
    rps = np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1) / (N_CLASSES - 1)
    return float(rps.mean())


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(_as_2d(probs), eps, 1.0)
    o = _onehot(outcomes)
    return float(-np.sum(o * np.log(p), axis=1).mean())


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = _as_2d(probs)
    o = _onehot(outcomes)
    return float(np.sum((p - o) ** 2, axis=1).mean())
