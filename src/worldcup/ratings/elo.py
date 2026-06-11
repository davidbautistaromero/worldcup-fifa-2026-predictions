"""Rating Elo dinámico para selecciones (estilo World Football Elo).

Funcional como punto de partida. Procesa el histórico cronológicamente y
mantiene un rating por equipo, ajustado por:
  - resultado vs. resultado esperado,
  - ventaja de localía,
  - (opcional) margen de victoria (margin of victory).

La diferencia de Elo entre dos equipos es un buen predictor y alimenta al
modelo de partido (se traduce en goles esperados).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from worldcup.config import load_config


class EloRatings:
    def __init__(
        self,
        k_factor: float = 40,
        home_advantage: float = 65,
        initial_rating: float = 1500,
        mov_multiplier: bool = True,
    ) -> None:
        self.k = k_factor
        self.home_advantage = home_advantage
        self.initial = initial_rating
        self.mov_multiplier = mov_multiplier
        self.ratings: dict[str, float] = defaultdict(lambda: initial_rating)

    @classmethod
    def from_config(cls) -> "EloRatings":
        elo_cfg = load_config()["ratings"]["elo"]
        return cls(
            k_factor=elo_cfg["k_factor"],
            home_advantage=elo_cfg["home_advantage"],
            initial_rating=elo_cfg["initial_rating"],
            mov_multiplier=elo_cfg["mov_multiplier"],
        )

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Probabilidad esperada de que A puntúe frente a B (logística Elo)."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def _mov_factor(self, goal_diff: int, rating_diff: float) -> float:
        """Multiplicador por margen de victoria (atenúa rachas infladas)."""
        if not self.mov_multiplier or goal_diff == 0:
            return 1.0
        return float(np.log(abs(goal_diff) + 1) * (2.2 / (rating_diff * 0.001 + 2.2)))

    def update_match(
        self, home: str, away: str, home_score: int, away_score: int, neutral: bool = False
    ) -> None:
        ha = 0.0 if neutral else self.home_advantage
        r_home = self.ratings[home] + ha
        r_away = self.ratings[away]

        exp_home = self.expected_score(r_home, r_away)
        if home_score > away_score:
            s_home = 1.0
        elif home_score < away_score:
            s_home = 0.0
        else:
            s_home = 0.5

        mov = self._mov_factor(home_score - away_score, r_home - r_away)
        delta = self.k * mov * (s_home - exp_home)
        self.ratings[home] += delta
        self.ratings[away] -= delta

    def fit(self, matches: pd.DataFrame) -> "EloRatings":
        """Procesa todo el histórico en orden cronológico."""
        df = matches.sort_values("date")
        neutral = df["neutral"] if "neutral" in df.columns else pd.Series(False, index=df.index)
        for row, is_neutral in zip(df.itertuples(index=False), neutral):
            self.update_match(
                row.home_team, row.away_team, row.home_score, row.away_score, bool(is_neutral)
            )
        return self

    def get(self, team: str) -> float:
        return self.ratings[team]

    def as_dataframe(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"team": list(self.ratings), "elo": list(self.ratings.values())})
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
