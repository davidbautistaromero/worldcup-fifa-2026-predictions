"""CLI del proyecto (typer). Punto de entrada `worldcup`.

Comandos:
  worldcup download-data   ingesta + limpieza + features → data/processed/
  worldcup train           ajusta ratings y modelos → outputs/models/
  worldcup evaluate        backtesting → métricas
  worldcup simulate        Monte Carlo del Mundial 2026 → outputs/reports/
"""
from __future__ import annotations

import typer

from worldcup.config import load_config

app = typer.Typer(help="Modelo de predicción del Mundial FIFA 2026.")


@app.command("download-data")
def download_data() -> None:
    """Descarga, limpia y genera features. TODO: cablear data.* ."""
    typer.echo("TODO: ingest → clean → features → data/processed/")


@app.command()
def train() -> None:
    """Ajusta Elo + Dixon-Coles (+ XGBoost) y los serializa. TODO: cablear."""
    typer.echo("TODO: ajustar modelos y guardar en outputs/models/")


@app.command()
def evaluate() -> None:
    """Backtesting contra Mundiales pasados. TODO: cablear evaluation.backtest."""
    cfg = load_config()
    typer.echo(f"TODO: backtesting en {cfg['evaluation']['backtest_tournaments']}")


@app.command()
def simulate(
    n: int = typer.Option(None, help="Nº de simulaciones (por defecto, config.yaml)."),
) -> None:
    """Monte Carlo del Mundial 2026. TODO: cablear simulation.montecarlo."""
    cfg = load_config()
    n = n or cfg["simulation"]["n_simulations"]
    typer.echo(f"TODO: correr {n} simulaciones → outputs/reports/")


if __name__ == "__main__":
    app()
