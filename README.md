# FIFA World Cup 2026 — Modelo de predicción

Modelo para predecir los resultados del Mundial 2026 (48 equipos, 12 grupos de 4).

Arquitectura en **3 capas** detrás de interfaces comunes, para poder intercambiar
modelos sin tocar el simulador:

```
datos crudos → INGESTA & FEATURES → RATINGS (Elo/FIFA) ─┐
                                                        ▼
                          HABILIDADES-POISSON ──► MODELO DE PARTIDO ──► SIMULACIÓN
                          (Dixon-Coles / bivariado)   (Random Forest    MONTE CARLO → probabilidades
                                                        híbrido)
                                                            ▼
                                  EVALUACIÓN / BACKTESTING (RPS vs cuotas, log-loss, Brier)
```

**Enfoque metodológico** (fundamentado en literatura — ver
[docs/metodologia.md](docs/metodologia.md)): el mejor predictor documentado es un
**Random Forest híbrido** que usa como *feature* central los **parámetros de
habilidad** estimados por un modelo de goles tipo Poisson/Dixon-Coles (ponderado
por decaimiento temporal e importancia del partido). Dixon-Coles no se descarta:
es el **motor de habilidades** y el generador de marcadores para la simulación.

El **formato del torneo es dato (YAML), no código** → ver `config/tournament_2026.yaml`.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
pip install -e ".[dev,dashboard]"
```

## Datos (Kaggle)

`worldcup download-data` descarga el dataset
[International football results from 1872 to 2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
vía la API de Kaggle. Necesitas un token **una sola vez**:

1. Crea una cuenta en [kaggle.com](https://www.kaggle.com) si no la tienes.
2. Ve a **Account → API → "Create New API Token"**. Descarga `kaggle.json`.
3. Colócalo en `C:\Users\<usuario>\.kaggle\kaggle.json`
   (o exporta las variables `KAGGLE_USERNAME` y `KAGGLE_KEY`).

> El dataset es de uso público (licencia CC0). No se versiona en git (`data/raw/`
> está en `.gitignore`).

## Uso

```bash
worldcup download-data    # descarga (Kaggle) + limpia  → data/interim/results_clean.parquet
worldcup train            # ajusta Elo + habilidades-Poisson + Random Forest → outputs/models/
worldcup evaluate         # backtesting 2014/2018/2022 (RPS vs cuotas) → métricas
worldcup simulate         # N simulaciones del Mundial 2026 → outputs/reports/
streamlit run dashboard/app.py    # dashboard local en http://localhost:8501
```

O con los scripts equivalentes en `scripts/`, o `make data|train|evaluate|simulate`.

## Estructura

| Carpeta | Qué hay |
|---|---|
| `docs/` | [Reporte técnico de metodología](docs/metodologia.md) (la matemática detrás del modelo) |
| `config/` | Configuración (`config.yaml`) y formato del torneo (`tournament_2026.yaml`) |
| `src/worldcup/data/` | Ingesta, limpieza y feature engineering |
| `src/worldcup/ratings/` | Elo dinámico y ranking FIFA |
| `src/worldcup/models/` | `base.py` (interfaz común), Poisson, Dixon-Coles, Random Forest híbrido |
| `src/worldcup/simulation/` | Partido, torneo (desde YAML) y Monte Carlo |
| `src/worldcup/evaluation/` | Métricas (RPS, log-loss, Brier) y backtesting |
| `dashboard/` | Dashboard Streamlit local (solo lee `outputs/reports/`) |
| `scripts/` | Ejecutables finos que llaman al paquete |
| `tests/` | Pruebas con pytest |

## Estado

Esqueleto inicial. Los módulos contienen stubs con `TODO` y `NotImplementedError`
donde falta la implementación. Elo, métricas (RPS) y carga de config/torneo están
funcionales como punto de partida.
