"""Limpieza y normalización de datos crudos → data/interim/.

Tareas clave:
  - Normalizar nombres de selecciones (p. ej. "West Germany" → "Germany",
    "USA"/"United States", "Korea Republic" → "South Korea"). Imprescindible
    para que los joins entre datasets no pierdan partidos.
  - Tipos de fecha y marcadores.
  - Marcar partidos neutrales.
"""
from __future__ import annotations

import pandas as pd

# TODO: ampliar este mapa según los nombres que aparezcan en tus datos.
COUNTRY_ALIASES: dict[str, str] = {
    "West Germany": "Germany",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "United States": "USA",
    "Czechoslovakia": "Czechia",  # decisión a revisar según el análisis
}


def normalize_team_names(df: pd.DataFrame, columns=("home_team", "away_team")) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = df[col].replace(COUNTRY_ALIASES)
    return df


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza del histórico de resultados.

    TODO: filtrar partidos sin marcador, deduplicar, validar tipos.
    """
    df = normalize_team_names(df)
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    if "neutral" in df.columns:
        df["neutral"] = df["neutral"].astype(bool)
    return df.sort_values("date").reset_index(drop=True)
