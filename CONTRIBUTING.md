# Contributing to Neptun

Thanks for your interest in the framework. Neptun is the open-source engine for
building a quant research lab. Contributions that make that engine more useful,
more correct, or easier to learn are very welcome.

## Before you start: the public/private boundary

Neptun is deliberately **framework-only**. There is a hard line between the
open-source tooling and the proprietary research that runs *on top of* it:

**In scope for this repo (please contribute!):**
- Backtest/evaluation engine, data contracts, walk-forward and sweep logic
- Statistics utilities (Sharpe variants, validation, cost stress)
- Textbook strategy templates (from published literature, with a citation)
- Examples, docs, tests, tooling, performance and correctness fixes

**Out of scope (do not submit, and please do not ask others to):**
- Specific, non-public trading signals or "alpha"-bearing strategy logic
- Proprietary scoring/ranking thresholds or factor-engineering recipes
- Real datasets, derived features, or research notes
- Anything that only makes sense as someone's private edge

If a strategy is in a textbook or a published paper, it's a fine template — add
the reference. If it's a private edge, keep it private.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # should pass offline
```

## Workflow

1. Fork the repo and create a feature branch: `your-handle/short-description`.
2. Make your change. Keep it focused — one idea per pull request.
3. Run the checks locally:
   ```bash
   ruff check .
   ruff format .
   pytest -q
   ```
4. Add or update tests for any behavior you change. Tests must run **offline**
   (use the synthetic-data fixtures; never require network or API keys in CI).
5. Open a pull request describing **what** changed and **why**.

## Style

- Python ≥ 3.10, type hints on public functions.
- `ruff` is the linter and formatter (config in `pyproject.toml`).
- Match the surrounding code: small, stateless, well-named functions.
- A strategy is a class implementing `generate_signals(data) -> (entries, exits)`.
  Keep template strategies short and readable — they are teaching material.

## Reporting bugs

Open an issue with a minimal reproduction (the synthetic-data example is a good
starting point) and the output you expected vs. what you got.

## Security

Never commit secrets. If you find a vulnerability, see
[SECURITY.md](SECURITY.md) for how to report it responsibly.
