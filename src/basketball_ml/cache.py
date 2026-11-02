"""Cache derived tables as CSV next to the raw data."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd


def cached_csv(path: str | Path, build: Callable[[], pd.DataFrame], parse_dates: Sequence[str] = ()) -> pd.DataFrame:
    """Read a derived table from ``path``, building and saving it on first use."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, parse_dates=list(parse_dates))
    table = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table
