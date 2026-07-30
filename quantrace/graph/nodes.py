"""Node-Katalog der Graph-IR — das geschlossene Vokabular (#179).

Look-Ahead ist hier STRUKTURELL ausgeschlossen: jede Operation im Katalog ist
kausal (rolling/ewm/diff sehen nur Gegenwart + Vergangenheit), und der einzige
zeitverschiebende Knoten (`transform.shift`) erlaubt nur `periods >= 0` —
in die Vergangenheit schieben ja, in die Zukunft nie. Es gibt schlicht keinen
Knoten, mit dem ein Graph zukünftige Werte referenzieren könnte.

Der Execution-Lag (Signal bei t → Position bei t+1) bleibt bewusst Sache des
Runners (`backtest_runner._lag_signals` / `config.execution_lag`) — exakt wie
bei den handgeschriebenen Templates. Die Paritäts-Tests in `tests/test_graph.py`
erzwingen Bit-Identität gegen `strategies/templates/`.

Alle Rechenfunktionen arbeiten auf breiten DataFrames (Index=Zeit,
Spalten=Symbol) — dieselbe Form, die `Strategy.generate_signals` liefert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantrace.graph.schema import BOOL, SERIES, SIGNAL
from quantrace.models import MarketData

# ---------------------------------------------------------------------------
# Definitions-Objekte


@dataclass(frozen=True)
class ParamDef:
    name: str
    kind: str  # "int" | "float"
    default: Any = None  # None = Pflichtparameter
    min: float | None = None
    doc: str = ""

    @property
    def required(self) -> bool:
        return self.default is None

    def coerce(self, value: Any) -> Any:
        """Wert typisieren + Grenzen prüfen. ValueError mit Klartext bei Verstoß."""
        try:
            out = int(value) if self.kind == "int" else float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Parameter {self.name!r}: {value!r} ist kein {self.kind}") from e
        if self.kind == "int" and float(value) != float(out):
            raise ValueError(f"Parameter {self.name!r}: {value!r} ist kein ganzzahliger Wert")
        if self.min is not None and out < self.min:
            raise ValueError(f"Parameter {self.name!r}: {out} < Minimum {self.min}")
        return out


@dataclass(frozen=True)
class NodeDef:
    type: str
    label: str
    doc: str
    inputs: tuple[tuple[str, str], ...]  # (port_name, port_type), Reihenfolge = UI-Reihenfolge
    output: str  # SERIES | BOOL | SIGNAL
    params: tuple[ParamDef, ...] = field(default_factory=tuple)
    #: fn(inputs: {port: DataFrame}, params, data) → DataFrame | (entries, exits)
    fn: Callable[..., Any] = None  # type: ignore[assignment]

    def param(self, name: str) -> ParamDef:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Rechen-Helpers


def _ohlcv_field(data: MarketData, name: str) -> pd.DataFrame:
    """OHLCV-Feld als breite Matrix. high/low fallen auf close zurück, wenn der
    Datensatz sie nicht trägt — dieselbe Semantik wie donchian_breakout._field,
    damit Graph und Template auf identischen Daten identisch rechnen."""
    try:
        return data.frame.xs(name, level="field", axis=1)
    except KeyError:
        if name in ("high", "low"):
            return data.frame.xs("close", level="field", axis=1)
        raise


def _as_bool(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(False).astype(bool)


def _rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    # MUSS bit-identisch zu strategies/templates/rsi_2._rsi bleiben (Wilder-
    # Glättung via ewm alpha=1/period). Der Paritäts-Test erzwingt das.
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    return 100.0 - (100.0 / (1.0 + rs))


def _edge_signals(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bool-Zustand → (Entry bei steigender, Exit bei fallender Flanke).
    Exakt das sma_/ema_crossover-Muster."""
    now = _as_bool(state)
    prev = now.shift(1, fill_value=False).astype(bool)
    return now & ~prev, ~now & prev


# ---------------------------------------------------------------------------
# Katalog

_DEFS: list[NodeDef] = []


def _register(node: NodeDef) -> None:
    _DEFS.append(node)


def _source(name: str, doc: str) -> None:
    _register(
        NodeDef(
            type=f"source.{name}",
            label=name.capitalize(),
            doc=doc,
            inputs=(),
            output=SERIES,
            fn=lambda inputs, params, data, _n=name: _ohlcv_field(data, _n),
        )
    )


_source("close", "Schlusskurse (adjustiert, wenn die Daten adjustiert geladen sind).")
_source("open", "Eröffnungskurse.")
_source("high", "Tageshochs (Fallback: close, wenn nicht vorhanden).")
_source("low", "Tagestiefs (Fallback: close, wenn nicht vorhanden).")
_source("volume", "Handelsvolumen.")

_register(
    NodeDef(
        type="indicator.sma",
        label="SMA",
        doc="Einfacher gleitender Durchschnitt über `window` Perioden.",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("window", "int", min=1, doc="Fensterlänge in Perioden"),),
        fn=lambda inputs, params, data: inputs["series"].rolling(params["window"]).mean(),
    )
)
_register(
    NodeDef(
        type="indicator.ema",
        label="EMA",
        doc="Exponentieller gleitender Durchschnitt (span, adjust=False).",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("span", "int", min=1),),
        fn=lambda inputs, params, data: inputs["series"]
        .ewm(span=params["span"], adjust=False)
        .mean(),
    )
)
_register(
    NodeDef(
        type="indicator.rolling_std",
        label="Rolling StdDev",
        doc="Rollierende Standardabweichung (ddof=0, wie bollinger_bands).",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("window", "int", min=1),),
        fn=lambda inputs, params, data: inputs["series"].rolling(params["window"]).std(ddof=0),
    )
)
_register(
    NodeDef(
        type="indicator.rolling_max",
        label="Rolling Max",
        doc="Rollierendes Maximum — z.B. das N-Tage-Hoch eines Donchian-Kanals.",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("window", "int", min=1),),
        fn=lambda inputs, params, data: inputs["series"].rolling(params["window"]).max(),
    )
)
_register(
    NodeDef(
        type="indicator.rolling_min",
        label="Rolling Min",
        doc="Rollierendes Minimum — z.B. das M-Tage-Tief eines Donchian-Kanals.",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("window", "int", min=1),),
        fn=lambda inputs, params, data: inputs["series"].rolling(params["window"]).min(),
    )
)
_register(
    NodeDef(
        type="indicator.rsi",
        label="RSI",
        doc="Relative Strength Index (Wilder-Glättung) — bit-identisch zu rsi_2.",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("period", "int", min=1),),
        fn=lambda inputs, params, data: _rsi(inputs["series"], params["period"]),
    )
)
_register(
    NodeDef(
        type="indicator.zscore",
        label="Z-Score",
        doc="(x − rolling mean) / rolling std (ddof=0) über `window` Perioden.",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("window", "int", min=2),),
        fn=lambda inputs, params, data: (
            inputs["series"] - inputs["series"].rolling(params["window"]).mean()
        )
        / inputs["series"].rolling(params["window"]).std(ddof=0),
    )
)

_register(
    NodeDef(
        type="transform.shift",
        label="Shift",
        doc="Serie um `periods` in die Vergangenheit schieben (Wert von t−periods "
        "bei t verfügbar). periods >= 0 — in die Zukunft schieben gibt es nicht.",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("periods", "int", default=1, min=0),),
        fn=lambda inputs, params, data: inputs["series"].shift(params["periods"]),
    )
)

for _op, _sym, _f in (
    ("add", "a + b", lambda a, b: a + b),
    ("sub", "a − b", lambda a, b: a - b),
    ("mul", "a × b", lambda a, b: a * b),
    ("div", "a ÷ b", lambda a, b: a / b),
):
    _register(
        NodeDef(
            type=f"math.{_op}",
            label=_sym,
            doc=f"Elementweise Rechnung: {_sym}.",
            inputs=(("a", SERIES), ("b", SERIES)),
            output=SERIES,
            fn=lambda inputs, params, data, _f=_f: _f(inputs["a"], inputs["b"]),
        )
    )

_register(
    NodeDef(
        type="math.scale",
        label="× Faktor",
        doc="Serie mit konstantem Faktor multiplizieren (z.B. k·StdDev).",
        inputs=(("series", SERIES),),
        output=SERIES,
        params=(ParamDef("factor", "float"),),
        fn=lambda inputs, params, data: inputs["series"] * params["factor"],
    )
)

for _op, _sym, _f in (
    ("gt", "a > b", lambda a, b: a > b),
    ("ge", "a ≥ b", lambda a, b: a >= b),
    ("lt", "a < b", lambda a, b: a < b),
    ("le", "a ≤ b", lambda a, b: a <= b),
):
    _register(
        NodeDef(
            type=f"logic.{_op}",
            label=_sym,
            doc=f"Vergleich {_sym}; NaN → False.",
            inputs=(("a", SERIES), ("b", SERIES)),
            output=BOOL,
            fn=lambda inputs, params, data, _f=_f: _as_bool(_f(inputs["a"], inputs["b"])),
        )
    )

for _op, _sym, _f in (
    ("gt_value", "x > Wert", lambda a, v: a > v),
    ("lt_value", "x < Wert", lambda a, v: a < v),
):
    _register(
        NodeDef(
            type=f"logic.{_op}",
            label=_sym,
            doc=f"Vergleich gegen Konstante: {_sym}; NaN → False.",
            inputs=(("series", SERIES),),
            output=BOOL,
            params=(ParamDef("value", "float"),),
            fn=lambda inputs, params, data, _f=_f: _as_bool(
                _f(inputs["series"], params["value"])
            ),
        )
    )

_register(
    NodeDef(
        type="logic.and",
        label="AND",
        doc="Beide Bedingungen wahr.",
        inputs=(("a", BOOL), ("b", BOOL)),
        output=BOOL,
        fn=lambda inputs, params, data: inputs["a"] & inputs["b"],
    )
)
_register(
    NodeDef(
        type="logic.or",
        label="OR",
        doc="Mindestens eine Bedingung wahr.",
        inputs=(("a", BOOL), ("b", BOOL)),
        output=BOOL,
        fn=lambda inputs, params, data: inputs["a"] | inputs["b"],
    )
)
_register(
    NodeDef(
        type="logic.not",
        label="NOT",
        doc="Bedingung negieren.",
        inputs=(("a", BOOL),),
        output=BOOL,
        fn=lambda inputs, params, data: ~inputs["a"],
    )
)
_register(
    NodeDef(
        type="logic.cross_above",
        label="Cross Above",
        doc="True genau in der Periode, in der a von ≤b auf >b wechselt.",
        inputs=(("a", SERIES), ("b", SERIES)),
        output=BOOL,
        fn=lambda inputs, params, data: _edge_signals(inputs["a"] > inputs["b"])[0],
    )
)
_register(
    NodeDef(
        type="logic.cross_below",
        label="Cross Below",
        doc="True genau in der Periode, in der a von >b auf ≤b wechselt.",
        inputs=(("a", SERIES), ("b", SERIES)),
        output=BOOL,
        fn=lambda inputs, params, data: _edge_signals(inputs["a"] > inputs["b"])[1],
    )
)

_register(
    NodeDef(
        type="signal.enter_exit_on_state",
        label="Signal aus Zustand",
        doc="Bool-Zustand → Entry bei steigender, Exit bei fallender Flanke "
        "(das sma_/ema_crossover-Muster).",
        inputs=(("state", BOOL),),
        output=SIGNAL,
        fn=lambda inputs, params, data: _edge_signals(inputs["state"]),
    )
)
_register(
    NodeDef(
        type="signal.enter_exit",
        label="Signal aus Entry/Exit",
        doc="Getrennte Entry-/Exit-Bedingungen direkt als Signale übernehmen "
        "(das bollinger_/rsi_2-/donchian-Muster).",
        inputs=(("enter", BOOL), ("exit", BOOL)),
        output=SIGNAL,
        fn=lambda inputs, params, data: (_as_bool(inputs["enter"]), _as_bool(inputs["exit"])),
    )
)


CATALOG: dict[str, NodeDef] = {d.type: d for d in _DEFS}


def catalog_payload() -> list[dict[str, Any]]:
    """Maschinenlesbare Katalog-Form für API/Editor — eine Quelle für beide."""
    return [
        {
            "type": d.type,
            "label": d.label,
            "doc": d.doc,
            "inputs": [{"name": n, "port_type": t} for n, t in d.inputs],
            "output": d.output,
            "params": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "default": p.default,
                    "min": p.min,
                    "required": p.required,
                    "doc": p.doc,
                }
                for p in d.params
            ],
        }
        for d in _DEFS
    ]
