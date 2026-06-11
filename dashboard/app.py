"""Dashboard local (Streamlit). Corre en tu máquina, sin nube:

    streamlit run dashboard/app.py     # → http://localhost:8501

Solo LEE de outputs/reports/ (las probabilidades ya calculadas por
`worldcup simulate`). No contiene lógica de modelado: mantiene la separación
limpia entre cálculo y visualización.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from worldcup.config import load_config

st.set_page_config(page_title="Mundial 2026 — Predicciones", page_icon="⚽", layout="wide")


@st.cache_data
def load_report() -> pd.DataFrame | None:
    cfg = load_config()
    report = Path(cfg["paths"]["reports"]) / "probabilities.csv"
    if not report.exists():
        return None
    return pd.read_csv(report)


def main() -> None:
    st.title("⚽ Mundial FIFA 2026 — Probabilidades")
    st.caption("Dashboard local. Datos generados por `worldcup simulate`.")

    df = load_report()
    if df is None:
        st.info(
            "Aún no hay resultados. Corre la simulación primero:\n\n"
            "```bash\nworldcup simulate\n```"
        )
        # TODO: cuando exista el reporte, renderizar:
        #   - tabla ordenada por P(campeón)
        #   - gráfico de barras de favoritos
        #   - bracket interactivo (dashboard/components/bracket.py)
        return

    st.subheader("Favoritos al título")
    st.dataframe(df.sort_values("p_champion", ascending=False), use_container_width=True)


if __name__ == "__main__":
    main()
