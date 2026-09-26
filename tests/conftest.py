from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantrace.models import MarketData, Timeframe


@pytest.fixture
def synthetic_md() -> MarketData:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2019-01-01", "2023-12-29")
    symbols = ["SPY", "QQQ"]
    frames = {}
    for s in symbols:
        ret = rng.normal(0.0004, 0.012, len(idx))
        close = 100 * np.exp(np.cumsum(ret))
        frames[s] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, len(idx))),
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, len(idx)).astype(float),
            },
            index=idx,
        )
    combined = pd.concat(frames, axis=1)
    combined.columns.names = ["symbol", "field"]
    return MarketData(
        universe="synthetic",
        symbols=symbols,
        timeframe=Timeframe.DAILY,
        start=idx[0].date(),
        end=idx[-1].date(),
        provider="synthetic",
        frame=combined,
    )


@pytest.fixture(autouse=True)
def _api_prozess_darf_pumpen(monkeypatch):
    """Tests laufen als API-Prozess, auch wenn ein Test ``worker.main`` importiert.

    ``worker/main.py`` setzt beim Import ``QUANTRACE_QUEUE_PUMPE=0`` — der Worker
    ist die Spur, nicht ihr Dispatcher (#335). In einer Testsitzung ist das ein
    Prozess für alles; ohne diese Zeile hinge, ob die Schlange pumpt, davon ab,
    welcher Test vorher den Worker importiert hat.
    """
    monkeypatch.setenv("QUANTRACE_QUEUE_PUMPE", "1")
