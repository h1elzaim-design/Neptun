from quantrace.brokers.base import Broker, Order, OrderSide, OrderType, Position

__all__ = ["Broker", "Order", "OrderSide", "OrderType", "Position"]


def get_broker(name: str, **kwargs: object) -> Broker:
    """Factory — wählt den Broker-Adapter ohne Top-Level-Import (alpaca/ib_insync lazy).

    Args:
        name: "alpaca" oder "ibkr".
        **kwargs: An den Broker-Konstruktor weitergereicht.
    """
    n = name.strip().lower()
    if n == "alpaca":
        from quantrace.brokers.alpaca import AlpacaBroker

        return AlpacaBroker(**kwargs)  # type: ignore[arg-type]
    if n == "ibkr":
        from quantrace.brokers.ibkr import IBKRBroker

        return IBKRBroker(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unbekannter Broker: {name!r}. Erwartet: 'alpaca' oder 'ibkr'.")
