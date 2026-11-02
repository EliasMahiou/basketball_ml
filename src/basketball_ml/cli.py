"""Command line entry point: ``basketball-ml <command> ...``."""

from __future__ import annotations

import argparse

from .projections import cli as projections_cli
from .twins import cli as twins_cli


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="basketball-ml", description="Machine learning on NBA data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    twins_cli.add_parser(subparsers)
    projections_cli.add_parser(subparsers)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
