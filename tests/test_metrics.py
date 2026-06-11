"""Tests de las métricas de evaluación (módulo funcional)."""
import numpy as np

from worldcup.evaluation.metrics import (
    brier_score,
    log_loss,
    ranked_probability_score,
)


def test_rps_perfect_prediction_is_zero():
    # Predicción perfecta de victoria local (clase 0).
    assert ranked_probability_score(np.array([1.0, 0.0, 0.0]), np.array([0])) == 0.0


def test_rps_orders_errors_correctly():
    # Errar local↔visitante (extremos) debe penalizar más que local↔empate.
    p = np.array([0.0, 0.0, 1.0])  # predijo visitante
    rps_far = ranked_probability_score(p, np.array([0]))   # real: local
    rps_near = ranked_probability_score(np.array([0.0, 1.0, 0.0]), np.array([0]))  # predijo empate
    assert rps_far > rps_near


def test_brier_and_logloss_perfect():
    p = np.array([1.0, 0.0, 0.0])
    assert brier_score(p, np.array([0])) == 0.0
    assert log_loss(p, np.array([0])) < 1e-9
