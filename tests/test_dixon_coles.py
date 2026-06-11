"""Tests de Dixon-Coles: corrección tau y matriz de marcadores.

El ajuste (fit) aún es un TODO; aquí probamos las partes ya funcionales fijando
parámetros a mano.
"""
import numpy as np

from worldcup.models.dixon_coles import DixonColesModel, dc_tau


def test_dc_tau_only_adjusts_low_scores():
    i = np.array([0, 0, 1, 1, 2])
    j = np.array([0, 1, 0, 1, 2])
    tau = dc_tau(i, j, lam=1.3, mu=1.1, rho=-0.1)
    # El marcador 2-2 no se corrige.
    assert tau[-1] == 1.0
    # Los marcadores bajos sí.
    assert not np.allclose(tau[:4], 1.0)


def test_predict_scoreline_is_normalized():
    m = DixonColesModel(xi=0.0, max_goals=8)
    # Fijamos parámetros a mano (sin entrenar) para probar predict_scoreline.
    m.attack = {"A": 0.2, "B": -0.1}
    m.defence = {"A": 0.1, "B": -0.2}
    m.home_adv = 0.25
    m.intercept = 0.0
    m.rho = -0.05
    m._fitted = True

    mat = m.predict_scoreline("A", "B", neutral=False)
    assert np.isclose(mat.sum(), 1.0)

    p_home, p_draw, p_away = m.predict_outcome("A", "B")
    assert np.isclose(p_home + p_draw + p_away, 1.0)
