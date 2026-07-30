"""Direkte Marktdaten-Provider-Clients (httpx-basiert, ohne openbb).

Liefern OHLCV im selben MultiIndex-Format wie der openbb-Pfad in data_agent:
    columns = MultiIndex[(symbol, field)], field ∈ {open, high, low, close, volume}
    index   = DatetimeIndex (tz-naiv, sortiert)

Damit bleibt der gesamte Backtest-/Sweep-Code provider-agnostisch.
"""
