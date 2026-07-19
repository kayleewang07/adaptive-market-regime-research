from pathlib import Path

from amrrp.data.providers import YFinanceProvider
from amrrp.experiments.runner import ExperimentRunner
from amrrp.experiments.specification import ExperimentSpec


def main():

    spec = ExperimentSpec()

    provider = YFinanceProvider()

    frames = []

    for symbol in spec.symbols:
        frame = provider.download(
            symbol=symbol,
            start=spec.start,
            end=spec.end,
        )

        frame["symbol"] = symbol
        frames.append(frame)

    # Use SPY as the primary dataset for the experiment
    market_data = frames[0]

    runner = ExperimentRunner(
        artifact_root=Path("artifacts")
    )

    result = runner.run(
        market_data,
        spec
    )

    print("Experiment complete!")
    print(result)


if __name__ == "__main__":
    main()
