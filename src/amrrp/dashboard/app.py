"""Streamlit dashboard entry point."""

from __future__ import annotations

from pathlib import Path


def main(artifact_root: str = "artifacts") -> None:
    """Render the artifact-driven Streamlit research dashboard."""
    import pandas as pd
    import streamlit as st

    root = Path(artifact_root)
    st.set_page_config(page_title="Adaptive Market Regime Research", layout="wide")
    st.title("Adaptive Market Regime Research Platform")
    runs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    page = st.sidebar.radio(
        "Page",
        [
            "Home",
            "Market Overview",
            "Current Regime",
            "Strategy Comparison",
            "Performance Metrics",
            "Regime History",
            "Interactive Charts",
            "Export Results",
        ],
    )
    if not runs:
        st.info("Run an experiment to populate dashboard artifacts.")
        return
    run = st.sidebar.selectbox("Experiment", runs, format_func=lambda path: path.name)
    metrics = pd.read_csv(run / "metrics.csv", index_col=0)
    if page in {"Home", "Strategy Comparison", "Performance Metrics", "Export Results"}:
        st.dataframe(metrics, use_container_width=True)
        if page == "Export Results":
            st.download_button("Download metrics", metrics.to_csv(), file_name="metrics.csv")
    elif page in {"Current Regime", "Regime History"}:
        st.line_chart(pd.read_parquet(run / "regimes.parquet"))
    else:
        st.line_chart(pd.read_parquet(run / "strategy_returns.parquet").cumsum())


if __name__ == "__main__":
    main()
