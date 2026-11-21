"""Streamlit dashboard for browsing experiment artifacts.

Launch with ``amrrp dashboard`` or ``streamlit run src/amrrp/dashboard/app.py``.
The artifact directory comes from the first script argument, then the
``AMRRP_ARTIFACTS`` environment variable, then ``artifacts``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PAGES = (
    "Overview",
    "Equity Curves",
    "Significance",
    "Regime History",
    "Strategies by Regime",
    "Regime Drivers",
    "Export",
)


def artifact_root() -> Path:
    """Resolve the artifact directory from argv, environment or default."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return Path(sys.argv[1])
    return Path(os.environ.get("AMRRP_ARTIFACTS", "artifacts"))


def main(root: Path | None = None) -> None:
    """Render the artifact-driven research dashboard."""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    root = root or artifact_root()
    st.set_page_config(page_title="Adaptive Market Regime Research", layout="wide")
    st.title("Adaptive Market Regime Research")
    runs = (
        sorted((path for path in root.iterdir() if (path / "manifest.json").exists()), reverse=True)
        if root.exists()
        else []
    )
    if not runs:
        st.info(f"No experiment runs found in `{root}`. Run `amrrp run` first.")
        return
    run = st.sidebar.selectbox("Experiment run", runs, format_func=lambda path: path.name)
    page = st.sidebar.radio("Page", PAGES)
    manifest = json.loads((run / "manifest.json").read_text())
    spec = manifest["specification"]
    train_end = pd.Timestamp(spec["train_end"])
    st.sidebar.markdown(
        f"**{spec['symbol']}** · `{spec['regime_model']}`  \n"
        f"Train: {spec['start']} → {spec['train_end']}  \n"
        f"Test: {spec['train_end']} → {spec['end']}"
    )

    def split_line(figure: go.Figure) -> go.Figure:
        figure.add_vline(x=train_end, line_dash="dash", line_color="gray")
        figure.add_annotation(
            x=train_end, y=1, yref="paper", text="out-of-sample →", showarrow=False, xanchor="left"
        )
        return figure

    if page == "Overview":
        st.subheader(f"Out-of-sample metrics ({spec['train_end']} → {spec['end']})")
        st.caption("Regime models never saw this period. Ratios assume a zero risk-free rate.")
        st.dataframe(pd.read_csv(run / "metrics_test.csv", index_col=0).round(3))
        regime_map = manifest.get("regime_strategy_map") or {}
        if regime_map:
            st.subheader("Regime-switching rule (chosen on training data only)")
            st.table(
                pd.DataFrame(
                    {"strategy": list(regime_map.values())},
                    index=pd.Index([f"regime {k}" for k in regime_map], name="regime"),
                )
            )
        st.subheader(f"In-sample metrics ({spec['start']} → {spec['train_end']})")
        st.caption("regime_switching is optimistic here: its rule was chosen on this period.")
        st.dataframe(pd.read_csv(run / "metrics_train.csv", index_col=0).round(3))

    elif page == "Significance":
        table = pd.read_csv(run / "sharpe_inference_test.csv", index_col=0)
        figure = go.Figure(
            go.Scatter(
                x=table["sharpe"],
                y=table.index,
                mode="markers",
                marker={"size": 10},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": table["ci_high"] - table["sharpe"],
                    "arrayminus": table["sharpe"] - table["ci_low"],
                },
            )
        )
        figure.add_vline(x=0, line_color="gray")
        figure.update_layout(
            title="Out-of-sample Sharpe ratio with 95% bootstrap interval",
            xaxis_title="Sharpe ratio",
            height=120 + 60 * len(table),
        )
        st.plotly_chart(figure, use_container_width=True)
        st.caption(
            "Stationary bootstrap, 2,000 resamples, 20-day mean blocks. p_value tests whether "
            "each strategy's Sharpe differs from buy-and-hold (paired on the same days)."
        )
        st.dataframe(table.round(3))

    elif page == "Equity Curves":
        returns = pd.read_parquet(run / "strategy_returns.parquet")
        scope = st.radio("Period", ["Full history", "Out-of-sample only"], horizontal=True)
        if scope == "Out-of-sample only":
            returns = returns[returns.index >= train_end]
        equity = (1 + returns.fillna(0)).cumprod()
        figure = px.line(equity, labels={"value": "Growth of $1", "index": "Date"})
        if scope == "Full history":
            split_line(figure)
        st.plotly_chart(figure, use_container_width=True)

    elif page == "Regime History":
        prices = pd.read_parquet(run / "prices.parquet")["Close"]
        regimes = pd.read_parquet(run / "regimes.parquet")["regime"]
        frame = pd.concat([prices, regimes], axis=1, join="inner").reset_index()
        frame.columns = ["Date", "Close", "regime"]
        order = [str(regime) for regime in sorted(frame["regime"].unique())]
        frame["regime"] = frame["regime"].astype(str)
        # Calm -> turbulent reads green -> amber -> red when there are three states.
        palette = (
            ["#2a9d8f", "#e9a23b", "#d62828"] if len(order) == 3 else px.colors.qualitative.Plotly
        )
        figure = px.scatter(
            frame,
            x="Date",
            y="Close",
            color="regime",
            log_y=True,
            category_orders={"regime": order},
            color_discrete_sequence=palette,
            labels={"Close": f"{spec['symbol']} close (log scale)"},
        )
        figure.update_traces(marker={"size": 3})
        st.plotly_chart(split_line(figure), use_container_width=True)
        if spec["regime_model"] != "bayesian_changepoint":
            st.caption("Regimes are numbered by training-period volatility: 0 is the calmest.")
        else:
            st.caption("Change-point labels are consecutive segments, not recurring states.")
        left, right = st.columns(2)
        left.subheader("Out-of-sample regime summary")
        left.dataframe(pd.read_csv(run / "regime_summary_test.csv", index_col=0).round(3))
        right.subheader("Transition probabilities (row → column)")
        right.dataframe(pd.read_csv(run / "transition_matrix_test.csv", index_col=0).round(3))

    elif page == "Strategies by Regime":
        table = pd.read_csv(run / "regime_metrics_test.csv", index_col=[0, 1])
        metric = st.selectbox("Metric", ["sharpe", "annual_return", "volatility", "max_drawdown"])
        pivot = table[metric].unstack("strategy")
        pivot.index = [f"regime {r}" for r in pivot.index]
        figure = px.imshow(
            pivot,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"y": "regime", "x": "strategy"},
            title=f"Out-of-sample {metric} by regime",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(table.round(3))

    elif page == "Regime Drivers":
        importance = pd.read_csv(run / "feature_importance.csv", index_col=0)["importance"]
        figure = px.bar(
            importance.sort_values(),
            orientation="h",
            labels={"value": "importance", "index": "feature"},
            title="Features that best separate regimes (surrogate random forest)",
        )
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, use_container_width=True)

    else:
        for path in sorted(run.glob("*.csv")) + [run / "report.md"]:
            st.download_button(f"Download {path.name}", path.read_bytes(), file_name=path.name)


if __name__ == "__main__":
    main()
