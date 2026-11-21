"""Convenience wrapper: ``python run_research.py [options]`` is the same as ``amrrp run``."""

import sys

from amrrp.cli import app

if __name__ == "__main__":
    app(["run", *sys.argv[1:]])
