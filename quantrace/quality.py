"""Data-Quality-Gate — fängt stille Datenfehler, bevor sie Backtests vergiften.

Ein Backtest auf kaputten Daten ist schlimmer als kein Backtest: er produziert
plausibel aussehende, aber falsche Kennzahlen. Dieses Modul prüft eine OHLCV-
Serie auf die üblichen Verdächtigen und gibt strukturierte Issues zurück.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    symbol: str
    kind: str
    detail: str
    severity: str = "warning"  # "warning" | "error"


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add(self, symbol: str, kind: str, detail: str, severity: str = "warning") -> None:
        self.issues.append(QualityIssue(symbol, kind, detail, severity))

    def log(self) -> None:
        for i in self.issues:
            fn = log.error if i.severity == "error" else log.warning
            fn("Data-Quality [%s] %s: %s", i.symbol, i.kind, i.detail)


def check_symbol(
    symbol: str,
    frame: pd.DataFrame,
    *,
    max_gap_days: int = 5,
    report: QualityReport | None = None,
) -> QualityReport:
    """Prüft eine Einzel-Symbol-OHLCV-Serie (DatetimeIndex, OHLCV-Spalten)."""
    report = report or QualityReport()

    if frame.empty:
        report.add(symbol, "empty", "keine Datenpunkte", "error")
        return report

    idx = frame.index
    if not idx.is_monotonic_increasing:
        report.add(symbol, "unsorted", "Index nicht monoton steigend", "error")
    if idx.has_duplicates:
        n = int(idx.duplicated().sum())
        report.add(symbol, "duplicate_dates", f"{n} doppelte Zeitstempel", "error")

    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            report.add(symbol, "missing_column", f"Spalte {col} fehlt", "error")
            continue
        s = frame[col].astype(float)
        n_nan = int(s.isna().sum())
        if n_nan:
            report.add(symbol, "nan", f"{col}: {n_nan} NaN-Werte")
        n_nonpos = int((s <= 0).sum())
        if n_nonpos:
            report.add(symbol, "nonpositive_price", f"{col}: {n_nonpos} Werte <= 0")

    if {"high", "low"} <= set(frame.columns):
        bad = int((frame["high"] < frame["low"]).sum())
        if bad:
            report.add(symbol, "high_lt_low", f"{bad} Bars mit high < low", "error")

    if "volume" in frame.columns:
        n_zero_vol = int((frame["volume"].fillna(0) <= 0).sum())
        if n_zero_vol:
            report.add(symbol, "zero_volume", f"{n_zero_vol} Bars mit Volumen <= 0")

    # Kalender-Lücken (grob: aufeinanderfolgende Datenpunkte > max_gap_days auseinander,
    # ohne Wochenenden penibel zu zählen — ein Loch von >1 Woche ist verdächtig).
    if len(idx) > 1:
        gaps = idx.to_series().diff().dt.days.dropna()
        big = gaps[gaps > max_gap_days]
        if len(big):
            worst = int(big.max())
            report.add(symbol, "calendar_gap", f"{len(big)} Lücken > {max_gap_days}d (max {worst}d)")

    return report


def check_universe(frames: dict[str, pd.DataFrame], **kwargs) -> QualityReport:
    """Prüft mehrere Symbole und sammelt alle Issues in einem Report."""
    report = QualityReport()
    for sym, frame in frames.items():
        check_symbol(sym, frame, report=report, **kwargs)
    return report
