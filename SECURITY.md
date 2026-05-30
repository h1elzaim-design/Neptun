# Security & Privacy

## Reporting a vulnerability

If you discover a security issue, please report it **privately** — do not open a
public issue. Use GitHub's "Report a vulnerability" (Security Advisories) on
this repository, or email the maintainer listed on the GitHub profile. We aim
to acknowledge reports within a few days.

## Secret handling — rules for contributors

This framework is built so that **no secret is ever required in the repository**:

- All credentials are read from **environment variables** at runtime
  (see `.env.example` for the full list of names — all values are blank).
- Copy `.env.example` to `.env` and fill it in **locally**. `.env` is in
  `.gitignore` and must never be committed.
- Never hardcode API keys, tokens, account numbers, or endpoints in code,
  tests, notebooks, or docs.
- Live-trading paths are gated off by default in the broker adapter and require
  an explicit, human opt-in. Do not change that default in a pull request.

If you accidentally commit a secret: rotate it immediately (assume it is
compromised), then remove it from history. A pushed secret should be treated as
already leaked even after deletion.

## The public/private boundary

Neptun is intentionally the **public framework only**. It does not contain, and
must not receive, proprietary research:

| Public (belongs here) | Private (never commit here) |
|-----------------------|-----------------------------|
| Backtest/evaluation engine, data contracts | Real strategies & signal definitions |
| Statistics & validation utilities | Factor engineering, proprietary scoring/ranking |
| Textbook strategy templates (with citations) | Production scoring thresholds & weights |
| Synthetic-data examples, public ticker lists | Real datasets & derived features |
| Docs, tests, tooling | Research notes, approval history, decision memos |

**What counts as "alpha-bearing logic"** (and therefore must stay private): any
code, parameter set, or note whose *value comes from being non-public* — a
signal that stops working once others know it, a threshold tuned on private
research, a feature engineered from a proprietary dataset. When unsure, keep it
private.

Maintainers will close, without merging, any contribution that appears to leak
private research, regardless of intent.
