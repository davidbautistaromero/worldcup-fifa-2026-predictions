"""Monte Carlo: corre N torneos y agrega las probabilidades.

Salida típica: por equipo, P(gana grupo), P(llega a octavos/cuartos/semis/
final) y P(campeón). Se guarda en outputs/reports/.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from worldcup.config import load_config
from worldcup.models.base import MatchModel
from worldcup.simulation.tournament import simulate_tournament

# Etapas en orden, para construir probabilidades acumuladas de "llegó al menos a".
STAGES = ["group", "r32", "r16", "qf", "sf", "final", "champion"]


def run_monte_carlo(
    model: MatchModel, n_simulations: int | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Ejecuta la simulación y devuelve un DataFrame de probabilidades por equipo.

    Reproducible vía la semilla de config.yaml (o el argumento `seed`).
    """
    cfg = load_config()
    n = n_simulations or cfg["simulation"]["n_simulations"]
    rng = np.random.default_rng(seed if seed is not None else cfg["seed"])

    champions: Counter[str] = Counter()
    reached_counts: dict[str, Counter[str]] = {}  # team -> Counter(stage)

    for _ in range(n):
        result = simulate_tournament(model, rng)
        if result.champion:
            champions[result.champion] += 1
        for team, stage in result.reached.items():
            reached_counts.setdefault(team, Counter())[stage] += 1

    # TODO: convertir conteos en probabilidades acumuladas por etapa y ensamblar
    # el DataFrame final (columnas: team, p_champion, p_final, p_sf, ...).
    raise NotImplementedError("TODO: agregar conteos → DataFrame de probabilidades.")
