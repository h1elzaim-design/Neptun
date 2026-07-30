"""Capacity, Turnover und Break-even-Kosten — trägt der Edge echtes Geld?

Motivation
----------
Nach „ist der Edge statistisch echt?" (DSR, PBO, Bootstrap) kommt auf jedem
Desk sofort die zweite Frage: **„Wie viel Kapital trägt das, und wie schnell
frisst die Realität den Edge auf?"** Ein Mean-Reverter mit 800 % Turnover und
ein Trendfolger mit 40 % sehen im Sharpe identisch aus und sind live völlig
verschiedene Dinge.

Dieses Modul liefert drei Sichten, alle pure (numpy, kein I/O):

1. **Turnover** — Σ|gehandeltes Notional| / NAV, annualisiert. Aus den
   Order-Records, wenn sie vorliegen; sonst als klar markierte Schätzung aus
   der Trade-Zahl.
2. **Kosten-Sensitivität + Break-even-bps** — Sharpe/CAGR als Funktion eines
   Kostenmultiplikators, plus der geschlossene Break-even: ab welchen Kosten
   pro Order-Seite ist der Edge weg? Das ist die *ökonomische* Ergänzung zur
   statistischen Disziplin: eine Strategie mit DSR 0.9, die bei 1.3× Kosten
   kippt, ist kein Edge, sondern eine Kostenwette.
3. **Capacity** — bis zu welchem AUM bleibt der Market-Impact unter einer
   Schranke? Über das **Square-root-Law** (Almgren et al. 2005;
   Tóth et al. 2011):

       impact_bps = 10⁴ · k · σ_daily · √(participation)

   mit ``participation`` = tagesgehandeltes Notional / ADV. Nach AUM
   aufgelöst ergibt das eine Kapazitätsgrenze **pro Symbol**; die bindende
   ist die des dünnsten Namens.

Konventionen
------------
- **Turnover ist one-way**: Σ|Δw| (gekauftes + verkauftes Notional / NAV),
  dieselbe Konvention wie :mod:`quantrace.paper.rebalance`. Kosten werden pro
  Order-*Seite* in bps angegeben (wie ``config/costs.yaml``), passen also
  ohne Faktor 2 dazu.
- Renditen sind **periodisch und bereits netto** der Baseline-Kosten (so
  kommen sie aus dem Runner). Die Brutto-Rendite wird für den Break-even
  zurückgerechnet.

Grenzen — bewusst und laut
--------------------------
Das Impact-Modell ist eine **Schätzung, keine Messung**. Wir haben keine
Live-Fills, keine Orderbuch-Tiefe, kein Intraday-Volumen. ``k`` ist eine
Literatur-Kalibrierung (typisch 0.5…1.0), kein für dieses Konto gemessener
Wert, und ADV aus Tagesdaten ignoriert, dass Volumen im Stress verschwindet —
genau dann, wenn eine Trendstrategie handelt. Die Zahlen taugen für
Größenordnungen („~2 Mio. oder ~200 Mio.?"), nicht für Zusagen.

Quellen
-------
- Almgren, Thum, Hauptmann & Li (2005), "Direct Estimation of Equity Market
  Impact", *Risk*.
- Tóth et al. (2011), "Anomalous Price Impact and the Critical Nature of
  Liquidity in Financial Markets", *Physical Review X*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

#: Literatur-Kalibrierung des Square-root-Law. Kein gemessener Wert.
DEFAULT_IMPACT_COEFFICIENT = 1.0
#: Impact-Schranke, bis zu der eine Strategie als „trägt das Kapital" gilt.
DEFAULT_MAX_IMPACT_BPS = 10.0
#: Kostenmultiplikatoren der Sensitivitätskurve.
DEFAULT_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)

TURNOVER_FROM_ORDERS = "orders"
TURNOVER_ESTIMATED = "trade_count_estimate"


# ---------------------------------------------------------------------------
# 1) Turnover
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnoverProfile:
    """Wie viel Notional bewegt die Strategie pro Jahr, relativ zum Konto.

    Attributes
    ----------
    annual_turnover:
        Σ|gehandeltes Notional| / NAV pro Jahr. 1.0 = das Konto wird einmal
        jährlich einseitig umgeschlagen.
    basis:
        ``orders`` (aus echten Order-Records) oder ``trade_count_estimate``
        (aus der Trade-Zahl hochgerechnet — Größenordnung, kein Messwert).
    """

    annual_turnover: float
    basis: str
    n_orders: int
    traded_notional: float
    average_account_value: float
    years: float

    @property
    def is_estimate(self) -> bool:
        return self.basis != TURNOVER_FROM_ORDERS

    def to_dict(self) -> dict[str, object]:
        return {
            "annual_turnover": self.annual_turnover,
            "basis": self.basis,
            "is_estimate": self.is_estimate,
            "n_orders": self.n_orders,
            "traded_notional": self.traded_notional,
            "average_account_value": self.average_account_value,
            "years": self.years,
        }


def turnover_from_orders(
    order_notionals: Sequence[float],
    account_values: Sequence[float],
    *,
    periods_per_year: float = 252.0,
) -> TurnoverProfile:
    """Annualisierter Turnover aus den tatsächlich platzierten Orders.

    Parameters
    ----------
    order_notionals:
        |Größe × Preis| je Order (Vorzeichen egal — es wird der Betrag
        genommen; Kauf und Verkauf zählen beide).
    account_values:
        Kontowert je Periode. Der Mittelwert ist der Nenner; das ist
        robuster als der Startwert, wenn das Konto über den Zeitraum wächst.

    Raises
    ------
    ValueError
        Leere oder non-finite Kontowerte, nicht-positiver Mittelwert.
    """
    notionals = np.abs(np.asarray(list(order_notionals), dtype=float))
    values = np.asarray(list(account_values), dtype=float)

    if values.size == 0:
        raise ValueError("account_values ist leer — Turnover braucht einen Nenner")
    if not np.isfinite(values).all() or not np.isfinite(notionals).all():
        raise ValueError("non-finite Werte in Orders/Kontowerten")

    avg_value = float(values.mean())
    if avg_value <= 0:
        raise ValueError(f"durchschnittlicher Kontowert {avg_value} ≤ 0")

    traded = float(notionals.sum())
    years = max(values.size / periods_per_year, 1e-9)
    return TurnoverProfile(
        annual_turnover=traded / avg_value / years,
        basis=TURNOVER_FROM_ORDERS,
        n_orders=int(notionals.size),
        traded_notional=traded,
        average_account_value=avg_value,
        years=years,
    )


def estimate_turnover_from_trades(
    n_trades: int,
    n_periods: int,
    *,
    average_position_weight: float,
    periods_per_year: float = 252.0,
) -> TurnoverProfile:
    """Turnover-Schätzung, wenn keine Order-Records persistiert sind.

    Ein „Trade" im Sinne der Trade-Metriken ist ein **Round-Trip** (rein und
    raus), bewegt also zweimal das Positions-Notional. Mit dem mittleren
    Positionsgewicht *w* folgt

        Turnover_p.a. ≈ 2 · n_trades · w / Jahre

    Das ist bewusst grob — die Positionsgröße schwankt mit der Zahl der
    gleichzeitig aktiven Signale. Das Ergebnis trägt deshalb
    ``basis='trade_count_estimate'`` und muss überall als Schätzung
    ausgewiesen werden.
    """
    if n_trades < 0:
        raise ValueError("n_trades muss ≥ 0 sein")
    if n_periods <= 0:
        raise ValueError("n_periods muss > 0 sein")
    if not 0.0 < average_position_weight <= 1.0:
        raise ValueError("average_position_weight muss in (0, 1] liegen")

    years = max(n_periods / periods_per_year, 1e-9)
    return TurnoverProfile(
        annual_turnover=2.0 * n_trades * average_position_weight / years,
        basis=TURNOVER_ESTIMATED,
        n_orders=2 * int(n_trades),
        traded_notional=float("nan"),
        average_account_value=float("nan"),
        years=years,
    )


# ---------------------------------------------------------------------------
# 2) Kosten-Sensitivität + Break-even
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostSensitivityPoint:
    """Ein Punkt der Sensitivitätskurve."""

    multiplier: float
    cost_bps_per_side: float
    annual_drag: float
    sharpe: float
    cagr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "multiplier": self.multiplier,
            "cost_bps_per_side": self.cost_bps_per_side,
            "annual_drag": self.annual_drag,
            "sharpe": self.sharpe,
            "cagr": self.cagr,
        }


@dataclass(frozen=True, slots=True)
class CostSensitivityResult:
    """Wie robust ist der Edge gegen die Kostenannahme?

    Attributes
    ----------
    break_even_bps:
        Kosten **pro Order-Seite**, bei denen die annualisierte Rendite auf 0
        fällt. ``None``, wenn die Strategie schon brutto nicht verdient.
    cost_buffer:
        ``break_even_bps / baseline_cost_bps`` — „wie viel Faktor Luft habe
        ich?". < 2 heißt: der Edge ist eine Kostenwette.
    """

    turnover_annual: float
    baseline_cost_bps: float
    baseline_sharpe: float
    baseline_cagr: float
    gross_annual_return: float
    break_even_bps: float | None
    cost_buffer: float | None
    points: list[CostSensitivityPoint]
    turnover_is_estimate: bool

    @property
    def survives_double_costs(self) -> bool:
        """Hält der Sharpe bei 2× Kosten noch über 0?"""
        for p in self.points:
            if abs(p.multiplier - 2.0) < 1e-9:
                return p.sharpe > 0
        return bool(self.break_even_bps and self.break_even_bps >= 2 * self.baseline_cost_bps)

    def to_dict(self) -> dict[str, object]:
        return {
            "turnover_annual": self.turnover_annual,
            "turnover_is_estimate": self.turnover_is_estimate,
            "baseline_cost_bps": self.baseline_cost_bps,
            "baseline_sharpe": self.baseline_sharpe,
            "baseline_cagr": self.baseline_cagr,
            "gross_annual_return": self.gross_annual_return,
            "break_even_bps": self.break_even_bps,
            "cost_buffer": self.cost_buffer,
            "survives_double_costs": self.survives_double_costs,
            "points": [p.to_dict() for p in self.points],
        }


def _annualised(returns: np.ndarray, periods_per_year: float) -> tuple[float, float]:
    """(Sharpe, CAGR) einer periodischen Return-Reihe."""
    if returns.size < 2:
        return 0.0, 0.0
    sigma = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / sigma * math.sqrt(periods_per_year)) if sigma > 0 else 0.0
    growth = float(np.prod(1.0 + returns))
    years = max(returns.size / periods_per_year, 1e-9)
    cagr = growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0
    return sharpe, cagr


def cost_sensitivity(
    returns: Sequence[float],
    *,
    turnover_annual: float,
    baseline_cost_bps: float,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    periods_per_year: float = 252.0,
    turnover_is_estimate: bool = False,
) -> CostSensitivityResult:
    """Sharpe/CAGR als Funktion der Kostenannahme, plus Break-even-bps.

    Die Renditen sind **netto** der Baseline-Kosten. Ein Multiplikator *m*
    legt den zusätzlichen Drag

        Δ_periode = (m − 1) · Turnover_p.a. · Kosten_bps / 10⁴ / Perioden

    gleichmäßig auf jede Periode. Das ist die ehrliche Näherung ohne
    Trade-Level-Resimulation: Kosten fallen in Wahrheit gebündelt an
    Rebalance-Tagen an, was Sharpe minimal anders trifft, aber dieselbe
    Jahres-Drag-Summe hat.

    Break-even (geschlossen, nicht gesucht): mit der annualisierten
    Brutto-Rendite ``r_gross = r_net + Turnover · c₀/10⁴`` ist

        c* = r_gross · 10⁴ / Turnover_p.a.   [bps pro Order-Seite]

    Raises
    ------
    ValueError
        Negativer Turnover/Kosten, zu kurze Reihe, non-finite Werte.
    """
    r = np.asarray(list(returns), dtype=float)
    if r.size < 2:
        raise ValueError(f"Return-Reihe zu kurz (T={r.size})")
    if not np.isfinite(r).all():
        raise ValueError("Return-Reihe enthält non-finite Werte")
    if turnover_annual < 0:
        raise ValueError("turnover_annual muss ≥ 0 sein")
    if baseline_cost_bps < 0:
        raise ValueError("baseline_cost_bps muss ≥ 0 sein")

    base_sharpe, base_cagr = _annualised(r, periods_per_year)
    net_annual = float(r.mean()) * periods_per_year
    baseline_drag = turnover_annual * baseline_cost_bps / 1e4
    gross_annual = net_annual + baseline_drag

    points: list[CostSensitivityPoint] = []
    for m in multipliers:
        extra_annual = (float(m) - 1.0) * baseline_drag
        stressed = r - extra_annual / periods_per_year
        sharpe, cagr = _annualised(stressed, periods_per_year)
        points.append(
            CostSensitivityPoint(
                multiplier=float(m),
                cost_bps_per_side=baseline_cost_bps * float(m),
                annual_drag=baseline_drag * float(m),
                sharpe=sharpe,
                cagr=cagr,
            )
        )

    break_even: float | None = None
    buffer: float | None = None
    if turnover_annual > 1e-12 and gross_annual > 0:
        break_even = gross_annual * 1e4 / turnover_annual
        if baseline_cost_bps > 1e-12:
            buffer = break_even / baseline_cost_bps

    return CostSensitivityResult(
        turnover_annual=float(turnover_annual),
        baseline_cost_bps=float(baseline_cost_bps),
        baseline_sharpe=base_sharpe,
        baseline_cagr=base_cagr,
        gross_annual_return=gross_annual,
        break_even_bps=break_even,
        cost_buffer=buffer,
        points=points,
        turnover_is_estimate=turnover_is_estimate,
    )


# ---------------------------------------------------------------------------
# 3) Capacity + Liquiditätsprofil
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SymbolCapacity:
    """Kapazitätsgrenze eines einzelnen Namens."""

    symbol: str
    weight: float
    adv_notional: float
    daily_volatility: float
    capacity_usd: float | None
    pct_adv_at_reference: float
    impact_bps_at_reference: float

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "weight": self.weight,
            "adv_notional": self.adv_notional,
            "daily_volatility": self.daily_volatility,
            "capacity_usd": self.capacity_usd,
            "pct_adv_at_reference": self.pct_adv_at_reference,
            "impact_bps_at_reference": self.impact_bps_at_reference,
        }


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    """Bis zu welchem AUM bleibt der Impact unter der Schranke?

    ``capacity_usd`` ist das Minimum über die Symbole — die Kapazität eines
    Portfolios ist die seines dünnsten gehandelten Namens, nicht sein
    Durchschnitt.
    """

    capacity_usd: float | None
    binding_symbol: str | None
    max_impact_bps: float
    impact_coefficient: float
    #: Repräsentative Tages-Vol (Median über die gehandelten Namen), nur fürs
    #: Reporting — gerechnet wird pro Symbol.
    daily_volatility: float
    turnover_annual: float
    reference_aum: float
    per_symbol: list[SymbolCapacity]
    warnings: list[str] = field(default_factory=list)

    @property
    def worst_pct_adv(self) -> float:
        return max((s.pct_adv_at_reference for s in self.per_symbol), default=0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity_usd": self.capacity_usd,
            "binding_symbol": self.binding_symbol,
            "max_impact_bps": self.max_impact_bps,
            "impact_coefficient": self.impact_coefficient,
            "daily_volatility": self.daily_volatility,
            "turnover_annual": self.turnover_annual,
            "reference_aum": self.reference_aum,
            "worst_pct_adv": self.worst_pct_adv,
            "per_symbol": [s.to_dict() for s in self.per_symbol],
            "warnings": list(self.warnings),
            "method": "square_root_impact_law",
        }


def capacity_estimate(
    *,
    turnover_annual: float,
    weights: Mapping[str, float],
    adv_notional: Mapping[str, float],
    daily_volatility: float | Mapping[str, float],
    max_impact_bps: float = DEFAULT_MAX_IMPACT_BPS,
    impact_coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
    reference_aum: float = 1_000_000.0,
    periods_per_year: float = 252.0,
) -> CapacityEstimate:
    """Kapazität nach dem Square-root-Impact-Law, pro Symbol aufgelöst.

    Das tägliche Handelsvolumen eines Symbols wird proportional zu seinem
    Portfoliogewicht angesetzt::

        notional_i(AUM) = AUM · Turnover_p.a. · wᵢ / Perioden

    Daraus die Partizipationsrate gegen ADV und der Impact
    ``10⁴ · k · σ · √participation``. Nach AUM aufgelöst::

        AUMᵢ* = participation* · ADVᵢ · Perioden / (Turnover · wᵢ)
        participation* = (max_impact_bps / (10⁴ · k · σ))²

    Parameters
    ----------
    weights:
        Symbol → Portfoliogewicht. Wird auf Summe 1 normiert.
    adv_notional:
        Symbol → durchschnittliches **Dollar**-Tagesvolumen.
    daily_volatility:
        Tages-Vol — entweder ein Wert für alle Namen oder ``Symbol → σ``.
        Pro Symbol ist ehrlicher: dünne Namen sind meist auch die volatilen,
        eine gemeinsame Vol würde ihren Impact systematisch unterschätzen.
        Fehlt ein Symbol im Mapping, gilt der Median der übrigen.

    Raises
    ------
    ValueError
        Leere Gewichte, negativer Turnover, nicht-positive Parameter.
    """
    if turnover_annual < 0:
        raise ValueError("turnover_annual muss ≥ 0 sein")
    if max_impact_bps <= 0 or impact_coefficient <= 0:
        raise ValueError("max_impact_bps und impact_coefficient müssen > 0 sein")
    if reference_aum <= 0:
        raise ValueError("reference_aum muss > 0 sein")
    if not weights:
        raise ValueError("weights ist leer — ohne Portfolio keine Kapazität")

    warnings: list[str] = []
    total = sum(max(float(w), 0.0) for w in weights.values())
    if total <= 0:
        raise ValueError("Summe der Gewichte ist 0")

    vol_by_symbol, representative_vol = _resolve_volatilities(daily_volatility, weights)
    if representative_vol <= 0:
        warnings.append("Tages-Vol ≤ 0 — Impact nicht schätzbar, Kapazität unbestimmt.")

    capped_once = False
    per_symbol: list[SymbolCapacity] = []
    for symbol, raw_weight in sorted(weights.items()):
        weight = max(float(raw_weight), 0.0) / total
        adv = float(adv_notional.get(symbol, 0.0))

        if weight <= 0:
            continue
        if adv <= 0:
            warnings.append(f"{symbol}: kein ADV im Lake — bleibt außen vor.")
            continue

        sigma = vol_by_symbol.get(symbol, representative_vol)
        daily_notional_ref = reference_aum * turnover_annual * weight / periods_per_year
        pct_adv_ref = daily_notional_ref / adv
        impact_ref = 1e4 * impact_coefficient * sigma * math.sqrt(max(pct_adv_ref, 0.0))

        if sigma <= 0:
            participation_limit = float("inf")
        else:
            participation_limit = (max_impact_bps / (1e4 * impact_coefficient * sigma)) ** 2
            if participation_limit > 1.0:
                participation_limit = 1.0
                capped_once = True

        if turnover_annual <= 1e-12 or not math.isfinite(participation_limit):
            capacity = None
        else:
            capacity = participation_limit * adv * periods_per_year / (turnover_annual * weight)

        per_symbol.append(
            SymbolCapacity(
                symbol=symbol,
                weight=weight,
                adv_notional=adv,
                daily_volatility=sigma,
                capacity_usd=capacity,
                pct_adv_at_reference=pct_adv_ref,
                impact_bps_at_reference=impact_ref,
            )
        )

    if capped_once:
        warnings.append(
            f"Die Impact-Schranke ({max_impact_bps:.0f} bps) erlaubt für mindestens einen "
            "Namen rechnerisch mehr als 100 % des ADV — jenseits des Gültigkeitsbereichs "
            "des Square-root-Law. Dort auf 100 % des ADV gedeckelt."
        )

    if turnover_annual <= 1e-12:
        warnings.append("Turnover ≈ 0 — kein Handel, keine Kapazitätsgrenze.")

    binding = min(
        (s for s in per_symbol if s.capacity_usd is not None),
        key=lambda s: s.capacity_usd,  # type: ignore[arg-type,return-value]
        default=None,
    )

    return CapacityEstimate(
        capacity_usd=binding.capacity_usd if binding else None,
        binding_symbol=binding.symbol if binding else None,
        max_impact_bps=float(max_impact_bps),
        impact_coefficient=float(impact_coefficient),
        daily_volatility=float(representative_vol),
        turnover_annual=float(turnover_annual),
        reference_aum=float(reference_aum),
        per_symbol=per_symbol,
        warnings=warnings,
    )


def _resolve_volatilities(
    daily_volatility: float | Mapping[str, float], weights: Mapping[str, float]
) -> tuple[dict[str, float], float]:
    """(Symbol → σ, repräsentatives σ) aus Skalar oder Mapping.

    Das repräsentative σ ist der **Median** der gehandelten Namen — robuster
    gegen einen einzelnen Ausreißer als der Mittelwert, und der Fallback für
    Symbole, die im Mapping fehlen.
    """
    if not isinstance(daily_volatility, Mapping):
        sigma = float(daily_volatility)
        return {}, sigma

    per_symbol = {
        str(s): float(v)
        for s, v in daily_volatility.items()
        if s in weights and np.isfinite(float(v)) and float(v) > 0
    }
    if not per_symbol:
        return {}, 0.0
    return per_symbol, float(np.median(list(per_symbol.values())))


__all__ = [
    "DEFAULT_IMPACT_COEFFICIENT",
    "DEFAULT_MAX_IMPACT_BPS",
    "DEFAULT_MULTIPLIERS",
    "TURNOVER_ESTIMATED",
    "TURNOVER_FROM_ORDERS",
    "CapacityEstimate",
    "CostSensitivityPoint",
    "CostSensitivityResult",
    "SymbolCapacity",
    "TurnoverProfile",
    "capacity_estimate",
    "cost_sensitivity",
    "estimate_turnover_from_trades",
    "turnover_from_orders",
]
