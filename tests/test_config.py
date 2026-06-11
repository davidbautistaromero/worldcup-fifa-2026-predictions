"""Tests de carga de configuración y formato del torneo (funcional)."""
from worldcup.config import load_config, load_tournament


def test_config_loads_and_resolves_paths():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert "models" in cfg["paths"]  # ruta resuelta a absoluta


def test_tournament_has_12_groups():
    t = load_tournament()
    assert t["meta"]["n_groups"] == 12
    assert len(t["groups"]) == 12
    # Cada grupo tiene 4 plazas.
    assert all(len(teams) == 4 for teams in t["groups"].values())
