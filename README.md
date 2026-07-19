# Adaptive Market Regime Research Platform

Production-oriented research platform for market-regime detection and strategy evaluation across regimes.

## Status

All planned layers are implemented. Install dependencies before downloading data or running experiments.

## Objectives

- Detect market regimes with multiple machine-learning approaches.
- Evaluate daily trading strategies under realistic execution assumptions.
- Compare strategy performance by regime.
- Produce reproducible research artifacts and an interactive Streamlit dashboard.

## Repository layout

```text
src/amrrp/       Production package
tests/           Unit and integration tests
configs/         Version-controlled experiment configurations
data/            Local cached and derived data (ignored by Git)
artifacts/       Generated models, metrics, figures, tables, and reports
notebooks/       Exploratory analysis
scripts/         Thin command-line entry points
docs/            Architecture and methodology documentation
```

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Planned phases

1. Architecture — approved
2. Repository scaffold — complete
3. Market-data ingestion — complete
4. Feature engineering — complete
5. Regime detection — complete
6. Event-driven backtesting — complete
7. Baseline strategies — complete
8. Research experiment runner — complete
9. Explainability — complete
10. Streamlit dashboard — complete

## License

MIT. See [LICENSE](LICENSE).
