"""Construcción y resolución del torneo desde config/tournament_2026.yaml.

Una sola corrida del torneo: fase de grupos → clasificados (2 por grupo + 8
mejores terceros) → cuadro eliminatorio → campeón. La aleatoriedad la inyecta
montecarlo.py vía el `rng`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldcup.config import load_tournament
from worldcup.models.base import MatchModel
from worldcup.simulation.match import simulate_group_match, simulate_knockout_match


@dataclass
class GroupStanding:
    team: str
    played: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


@dataclass
class TournamentResult:
    """Resultado de UNA simulación completa (qué etapa alcanzó cada equipo)."""
    champion: str | None = None
    finalists: list[str] = field(default_factory=list)
    semifinalists: list[str] = field(default_factory=list)
    reached: dict[str, str] = field(default_factory=dict)  # team -> última fase alcanzada


def simulate_group_stage(
    model: MatchModel, host_teams: set[str], rng: np.random.Generator
) -> dict[str, list[GroupStanding]]:
    """Juega todos los grupos (round-robin) y devuelve las tablas ordenadas.

    TODO:
      - Para cada grupo, generar los 6 enfrentamientos (round-robin de 4).
      - Marcar neutral=False si un anfitrión juega en casa.
      - Acumular puntos/goles, ordenar por los desempates del YAML.
    """
    raise NotImplementedError("TODO: round-robin por grupo + ordenación por desempates.")


def select_qualifiers(tables: dict[str, list[GroupStanding]], cfg: dict) -> list[str]:
    """2 primeros de cada grupo + 8 mejores terceros. TODO: implementar."""
    raise NotImplementedError("TODO: selección de clasificados y mejores terceros.")


def simulate_knockout(
    model: MatchModel, bracket_seeds: list[str], rng: np.random.Generator
) -> TournamentResult:
    """Resuelve R32 → R16 → cuartos → semis → final. TODO: implementar."""
    raise NotImplementedError("TODO: progresión del cuadro eliminatorio.")


def simulate_tournament(model: MatchModel, rng: np.random.Generator) -> TournamentResult:
    """Una corrida completa del Mundial 2026."""
    cfg = load_tournament()
    host_teams = set(cfg.get("host_teams", []))
    tables = simulate_group_stage(model, host_teams, rng)
    qualifiers = select_qualifiers(tables, cfg)
    return simulate_knockout(model, qualifiers, rng)
