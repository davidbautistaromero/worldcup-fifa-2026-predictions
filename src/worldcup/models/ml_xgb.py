"""Benchmark de machine learning: XGBoost que clasifica el resultado (1-X-2).

A diferencia de Poisson/Dixon-Coles (que modelan marcadores), este modelo
predice directamente la probabilidad de victoria local / empate / visitante a
partir de features (diff de Elo, ranking FIFA, localía, forma...).

Nota de arquitectura: como predice resultados y no marcadores, no produce una
matriz de scoreline natural. Se usa principalmente para EVALUACIÓN/comparación
(RPS, log-loss). Para la simulación del torneo, que necesita marcadores para
los desempates, se recomienda Dixon-Coles. `predict_scoreline` queda sin
implementar a propósito.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from worldcup.config import load_config
from worldcup.models.base import MatchModel

# Orden de clases fijo: 0=local, 1=empate, 2=visitante
CLASSES = ("home", "draw", "away")


class XGBoostModel(MatchModel):
    def __init__(self, **params) -> None:
        cfg = load_config()["models"]["xgboost"]
        self.params = {**cfg, **params}
        self.model = None
        self.feature_cols: list[str] = []

    def fit(self, matches: pd.DataFrame) -> "XGBoostModel":
        """Entrena el clasificador.

        TODO:
          1. Construir features con data.features.build_match_features.
          2. Etiqueta y = {home/draw/away} desde el marcador.
          3. Entrenar xgboost.XGBClassifier(objective="multi:softprob", ...).
        """
        raise NotImplementedError("TODO: entrenar XGBClassifier sobre features.")

    def predict_outcome(
        self, home: str, away: str, neutral: bool = False
    ) -> tuple[float, float, float]:
        """Probabilidades (local, empate, visitante) del clasificador."""
        raise NotImplementedError("TODO: construir features del partido y predict_proba.")

    def predict_scoreline(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        raise NotImplementedError(
            "XGBoostModel predice resultados (1-X-2), no marcadores. "
            "Usa predict_outcome, o Dixon-Coles para la simulación."
        )
