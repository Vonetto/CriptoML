"""Date helpers for scheduling rebalances and universes."""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def normalize_calendar(dates: Iterable[pd.Timestamp | str]) -> pd.DatetimeIndex:
    idx = pd.to_datetime(pd.Index(list(dates)))
    idx = idx.sort_values().unique()
    return pd.DatetimeIndex(idx.normalize())


def align_to_calendar(
    target_dates: Iterable[pd.Timestamp],
    calendar: pd.DatetimeIndex,
    method: str = "backward",
) -> List[pd.Timestamp]:
    aligned: List[pd.Timestamp] = []
    values = calendar.values
    for target in target_dates:
        ts = pd.Timestamp(target).normalize()
        if method == "backward":
            pos = values.searchsorted(ts.to_datetime64(), side="right") - 1
        else:
            pos = values.searchsorted(ts.to_datetime64(), side="left")
        if pos < 0 or pos >= len(values):
            continue
        aligned.append(pd.Timestamp(values[pos]))
    if not aligned:
        return []
    idx = pd.DatetimeIndex(aligned).unique().sort_values()
    return list(idx)


def generate_schedule(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
    calendar: pd.DatetimeIndex,
    method: str = "backward",
) -> List[pd.Timestamp]:
    if start >= end:
        return []
    raw = pd.date_range(start=start, end=end, freq=freq)
    return align_to_calendar(raw, calendar, method=method)
