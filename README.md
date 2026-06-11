# FIFA World Cup 2026 — Modelo de predicción

Modelo para predecir los resultados del Mundial 2026 (48 equipos, 12 grupos de 4).

Arquitectura en **3 capas** detrás de interfaces comunes, para poder intercambiar
modelos sin tocar el simulador:

```
datos crudos → INGESTA & FEATURES → RATINGS (Elo/FIFA) → MODELO DE PARTIDO → SIMULACIÓN
                                                          (Dixon-Coles/XGBoost)   MONTE CARLO → probabilidades
                                                                  ↓
                                                          EVALUACIÓN / BACKTESTING (RPS, log-loss, Brier)
```

El **formato del torneo es dato (YAML), no código** → ver `config/tournament_2026.yaml`.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
pip install -e ".[dev,dashboard]"
```

## Uso

```bash
worldcup download-data    # descarga + limpia + features  → data/processed/
worldcup train            # ajusta Elo + Dixon-Coles + XGBoost → outputs/models/
worldcup evaluate         # backtesting 2014/2018/2022 → métricas
worldcup simulate         # N simulaciones del Mundial 2026 → outputs/reports/
streamlit run dashboard/app.py    # dashboard local en http://localhost:8501
```

O con los scripts equivalentes en `scripts/`, o `make data|train|evaluate|simulate`.

## Estructura

| Carpeta | Qué hay |
|---|---|
| `config/` | Configuración (`config.yaml`) y formato del torneo (`tournament_2026.yaml`) |
| `src/worldcup/data/` | Ingesta, limpieza y feature engineering |
| `src/worldcup/ratings/` | Elo dinámico y ranking FIFA |
| `src/worldcup/models/` | `base.py` (interfaz común), Poisson, Dixon-Coles, XGBoost |
| `src/worldcup/simulation/` | Partido, torneo (desde YAML) y Monte Carlo |
| `src/worldcup/evaluation/` | Métricas (RPS, log-loss, Brier) y backtesting |
| `dashboard/` | Dashboard Streamlit local (solo lee `outputs/reports/`) |
| `scripts/` | Ejecutables finos que llaman al paquete |
| `tests/` | Pruebas con pytest |

## Estado

Esqueleto inicial. Los módulos contienen stubs con `TODO` y `NotImplementedError`
donde falta la implementación. Elo, métricas (RPS) y carga de config/torneo están
funcionales como punto de partida.
