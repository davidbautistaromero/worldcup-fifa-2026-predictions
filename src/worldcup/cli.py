"""CLI del proyecto (typer). Punto de entrada `worldcup`.

Comandos:
  worldcup download-data   ingesta + limpieza + features → data/processed/
  worldcup train           ajusta ratings y modelos → outputs/models/
  worldcup evaluate        backtesting → métricas
  worldcup simulate        Monte Carlo del Mundial 2026 → outputs/reports/
"""
from __future__ import annotations

from pathlib import Path

import typer

from worldcup.config import load_config

app = typer.Typer(help="Modelo de predicción del Mundial FIFA 2026.")


@app.command("download-data")
def download_data() -> None:
    """Descarga (Kaggle) + limpia el histórico de resultados → data/interim/.

    El feature engineering (data/processed/) queda como paso posterior.
    """
    from worldcup.data.clean import clean_results
    from worldcup.data.download import KaggleCredentialsError, download_dataset
    from worldcup.data.ingest import load_results

    cfg = load_config()
    try:
        typer.echo(f"Descargando resultados '{cfg['data']['kaggle_dataset']}'...")
        download_dataset()
        typer.echo(f"Descargando ranking FIFA '{cfg['data']['kaggle_ranking_dataset']}'...")
        download_dataset(cfg["data"]["kaggle_ranking_dataset"])
    except KaggleCredentialsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo("Cargando y limpiando resultados...")
    df = clean_results(load_results())

    interim = Path(cfg["paths"]["data_interim"]) / "results_clean.parquet"
    interim.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim, index=False)
    typer.secho(
        f"OK: {len(df):,} partidos limpios -> {interim}", fg=typer.colors.GREEN
    )


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
