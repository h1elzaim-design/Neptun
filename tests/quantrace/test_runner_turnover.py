"""Turnover-Extraktion aus den Order-Records — ohne vectorbt.

Der Runner reicht nur durch; die Rechnung selbst lebt in
`quantrace.stats.capacity`. Getestet wird hier die Brücke: Spaltenauflösung,
Robustheit gegen fehlende/kaputte Records und dass ein fehlender Turnover
nie den Backtest umbringt.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantrace.backtest_runner import _order_notionals, _turnover_annual


class _FakeOrders:
    def __init__(self, frame: pd.DataFrame | None, raises: bool = False):
        self._frame = frame
        self._raises = raises

    @property
    def records_readable(self) -> pd.DataFrame:
        if self._raises:
            raise RuntimeError("records_readable existiert in dieser Version nicht")
        return self._frame


class _FakePortfolio:
    def __init__(self, orders: _FakeOrders):
        self.orders = orders


def _pf(frame: pd.DataFrame | None, raises: bool = False) -> _FakePortfolio:
    return _FakePortfolio(_FakeOrders(frame, raises))


def _equity(n: int = 252, value: float = 100_000.0) -> pd.Series:
    return pd.Series([value] * n, index=pd.bdate_range("2020-01-01", periods=n))


# --- Notional-Extraktion ------------------------------------------------------


def test_notionals_multiply_size_by_price():
    frame = pd.DataFrame({"Size": [10.0, 5.0], "Price": [100.0, 200.0]})
    assert _order_notionals(frame) == [1000.0, 1000.0]


def test_notionals_are_absolute():
    """Verkäufe kommen mit negativer Size — Turnover ist ein Betrag."""
    frame = pd.DataFrame({"Size": [10.0, -10.0], "Price": [100.0, 100.0]})
    assert _order_notionals(frame) == [1000.0, 1000.0]


def test_notionals_resolve_column_case_insensitively():
    frame = pd.DataFrame({"size": [2.0], "price": [50.0]})
    assert _order_notionals(frame) == [100.0]


def test_notionals_drop_non_finite_rows():
    frame = pd.DataFrame({"Size": [1.0, float("nan")], "Price": [100.0, 100.0]})
    assert _order_notionals(frame) == [100.0]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"Size": [1.0]}),  # Price fehlt
        pd.DataFrame({"Price": [1.0]}),  # Size fehlt
        None,
    ],
)
def test_notionals_empty_when_records_unusable(frame):
    assert _order_notionals(frame) == []


# --- Brücke in den Runner -----------------------------------------------------


def test_turnover_annualises_correctly():
    # 4 × 25k gehandelt auf 100k NAV über 126 Bars (halbes Jahr) → 2.0 p.a.
    frame = pd.DataFrame({"Size": [250.0] * 4, "Price": [100.0] * 4})
    assert _turnover_annual(_pf(frame), _equity(126), 252) == pytest.approx(2.0)


def test_turnover_is_none_when_records_raise():
    """Alte/andere vectorbt-Version darf keinen Backtest umbringen."""
    assert _turnover_annual(_pf(None, raises=True), _equity(), 252) is None


def test_turnover_is_none_without_orders():
    assert _turnover_annual(_pf(pd.DataFrame()), _equity(), 252) is None


def test_turnover_is_none_on_degenerate_account():
    frame = pd.DataFrame({"Size": [1.0], "Price": [100.0]})
    assert _turnover_annual(_pf(frame), _equity(value=0.0), 252) is None


def test_turnover_is_none_without_equity():
    frame = pd.DataFrame({"Size": [1.0], "Price": [100.0]})
    assert _turnover_annual(_pf(frame), pd.Series(dtype=float), 252) is None


def test_backtest_result_defaults_turnover_to_none():
    """Alt-JSONs ohne das Feld müssen weiter parsen."""
    from quantrace.models import BacktestResult

    payload = {
        "strategy_id": "x",
        "data_hash": "h",
        "config": {},
        "start": "2020-01-01",
        "end": "2020-12-31",
        "total_return": 0.1,
        "cagr": 0.1,
        "sharpe": 1.0,
        "sortino": 1.2,
        "calmar": 0.5,
        "max_drawdown": -0.2,
        "avg_drawdown": -0.05,
        "ulcer_index": 0.03,
        "trades": {
            "n_trades": 10,
            "win_rate": 0.6,
            "avg_trade_return": 0.01,
            "avg_winner": 0.02,
            "avg_loser": -0.01,
            "profit_factor": 1.5,
            "expectancy": 0.01,
        },
    }
    assert BacktestResult.model_validate(payload).turnover_annual is None
