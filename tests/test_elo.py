"""Tests del rating Elo (módulo funcional)."""
import pandas as pd

from worldcup.ratings.elo import EloRatings


def test_expected_score_symmetric():
    # Igual rating → 0.5 cada uno.
    assert EloRatings.expected_score(1500, 1500) == 0.5


def test_winner_gains_loser_loses():
    elo = EloRatings(mov_multiplier=False)
    elo.update_match("A", "B", 2, 0, neutral=True)
    assert elo.get("A") > 1500
    assert elo.get("B") < 1500
    # Elo es de suma cero entre los dos equipos.
    assert round(elo.get("A") + elo.get("B"), 6) == 3000.0


def test_fit_runs_over_history():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "home_team": ["A", "B"],
            "away_team": ["B", "C"],
            "home_score": [1, 0],
            "away_score": [0, 0],
            "neutral": [True, True],
        }
    )
    elo = EloRatings().fit(df)
    assert elo.get("A") > elo.get("C") or elo.get("A") != 1500
    assert not elo.as_dataframe().empty
