# AI Trader — Interactive Brokers

An always-on, AI-driven intraday trading system for US stocks and ETFs on Interactive
Brokers. It plans before the open, re-evaluates every few minutes through the session,
manages its own stops, enforces its own risk limits, reconciles against the broker
constantly, and writes down why it did everything.

**Guiding principle: the LLM proposes; deterministic code disposes.** The model never
computes a price, sizes a position, places an order, or holds authoritative state. It reads
pre-computed indicators and emits a structured proposal (an action, a conviction, a risk
preference expressed in ATRs). Plain Python turns that into prices and share counts, runs it
through a 24-check deterministic risk gate, and only then reaches the broker. See
[`src/aitrader/domain/proposals.py`](src/aitrader/domain/proposals.py) for exactly what the
model is — and is not — allowed to say, and
[`src/aitrader/risk/engine.py`](src/aitrader/risk/engine.py) for the gate.

Every entry is placed as a **native IBKR bracket order** (parent + stop + target), so the
position stays protected even if this process crashes, the machine reboots, or the daily
Gateway restart happens mid-session.

## Before you start

You need, in this order:

1. **An IBKR account with API access** (IBKR Pro; Lite does not support Israeli residency).
   Confirm which entity carries your account — Interactive Brokers LLC directly, or an
   Israeli introducing broker such as MEXEM — from your statement header. This determines
   market-data pricing and whether the "no delayed data" restriction below applies to you.
2. **A live US market-data subscription** (the US Securities Snapshot and Futures Value
   Bundle, NonProfessional pricing if you qualify). IBKR no longer offers delayed US equity
   quotes to Interactive Brokers LLC clients, so there is no fallback — the system fails
   loudly rather than trading on stale data if this isn't set up.
3. **That subscription explicitly shared to your paper account** in Client Portal
   (Settings → Account Settings → Paper Trading Account). Skip this and the paper account
   gets nothing at all.
4. **An Ollama Cloud API key** from <https://ollama.com/settings/keys> (or a local Ollama
   instance — see `llm.provider` in `config/config.yaml`).
5. **Docker and Docker Compose.**

## Setup

```bash
cp .env.example .env            # fill in TWS_USERID, TWS_PASSWORD, OLLAMA_API_KEY
mkdir -p secrets
echo -n 'your-ibkr-password' > secrets/tws_password.txt

docker compose up ib-gateway    # bring the broker connection up first
```

Open `127.0.0.1:5900` with a VNC client and approve the initial IB Key 2FA push on your
phone. Once IB Gateway reports connected:

```bash
docker compose up -d            # trader + dashboard
```

The dashboard is at `http://127.0.0.1:8080`. Everything defaults to **paper trading** —
see [Going live](#going-live) before that changes.

### Running a connectivity check first

```bash
docker compose run --rm trader python -m aitrader doctor
```

This proves the things that must be true before the system is trusted with a session:
connected, account ID matches the configured mode, real-time market data actually flows
(`marketDataType == 1`, a live bid/ask), and the scanner returns rows. If market data
doesn't come through, stop here and fix the subscription — nothing downstream can compensate
for it.

## How it fits together

```
IB Gateway (Docker, IBC auto-login)
        │  ib_async
   broker layer  ── connection watchdog · scanner · historical · orders · account
        │
   data layer    ── DuckDB bar store, token-bucket rate limiters
        │
   analytics     ── indicators → FeaturePack (the model's entire numeric view)
        │
   LLM layer     ── Ollama Cloud/local · structured-output ladder · session narrative
        │            proposals only — no prices, no sizes, no order IDs
   sizing → risk gate (24 deterministic checks, non-bypassable)
        │
   execution     ── native IBKR bracket orders
        │
   dashboard     ── FastAPI + WebSocket, kill switch
```

### The 100+ symbol problem

IBKR allows roughly 60 historical-data requests per 10 minutes and ~100 concurrent
market-data lines — nowhere near enough to poll a large universe directly. The system
instead uses three tiers:

- **Broad** (`config/universe.csv`, 100+ symbols): daily bars backfilled once pre-market,
  refreshed intraday only via IBKR's server-side scanner, which costs zero market-data
  lines because scanner results carry no quote fields.
- **Focus** (~20 symbols, `universe.focus_list_size`): the deterministically-ranked
  shortlist that actually gets streaming subscriptions and reaches the model. Membership
  has hysteresis (`churn_margin`, `churn_persistence_cycles`) so it doesn't thrash against
  IBKR's subscription-rate limits.
- **Positions**: always subscribed regardless of rank.

See [`src/aitrader/analytics/ranking.py`](src/aitrader/analytics/ranking.py) and
[`src/aitrader/data/ratelimit.py`](src/aitrader/data/ratelimit.py).

### Continuity — the trader's notebook

A rolling session narrative (`src/aitrader/llm/narrative.py`) is threaded through every
cycle's prompt: the pre-market thesis, what's been done and why, what's being watched,
remaining risk budget. This is what makes the system reason like one continuous trader
through the day rather than independent, amnesiac calls every few minutes. Read it live at
`/narrative` on the dashboard.

### The weekly 2FA outage — read this

IBKR invalidates the session token roughly every Sunday at 01:00 ET, and the next login
needs an interactive IB Key push on your phone. **This cannot be automated, by design** —
the system detects the connected-but-unauthenticated state, refuses to trade, raises a loud
dashboard banner, and keeps retrying. Expect to approve one push a week. The daily Gateway
restart (`AUTO_RESTART_TIME`) is different and does not need this.

## The dashboard

| Page | What it shows |
|---|---|
| `/` | Equity, day P&L, positions, connection health, recent cycles, the kill button |
| `/narrative` | The live session log — what the AI is thinking and why |
| `/trades` | Order and fill history |
| `/rejections` | Every risk-gate veto with its reason — if the system isn't trading, this tells you why |
| `/universe` | Focus list, scanner output, market-data line budget, pacing utilization |

**Kill switch**: the dashboard button (and `python -m aitrader kill --mode halt|flatten`)
writes a sentinel file at `data/KILL`. `touch data/KILL` from any shell works too, even if
the web process is unresponsive — that's deliberate.

## Configuration

- `config/config.yaml` — risk limits, cadence, LLM models, broker/data settings. Committed,
  no secrets. Every risk number is clamped to an absolute ceiling in
  [`src/aitrader/hard_limits.py`](src/aitrader/hard_limits.py) at load time — this file can
  only make a limit *tighter*, never looser. A clamp is logged loudly at startup.
- `config/universe.csv` — the broad symbol list.
- `.env` — secrets and the trading-mode switch. Never committed.

## Going live

Real money requires **three independent things to agree** (checked at startup and refused
otherwise — see `Settings.verify_live_interlock` in `src/aitrader/config.py`):

1. `TRADING_MODE=live` in `.env`
2. `ALLOW_LIVE_TRADING=true` in `.env`
3. `IB_ACCOUNT_ID` set to the exact account IBKR reports (must not start with `DU`/`DF`)

Do not do this until you have run a long, unattended paper session and reviewed the
`/rejections` and `/narrative` pages. Read the caveats below first.

## Testing

```bash
uv venv && uv pip install -e ".[dev]"
pytest                    # 800+ tests, no Gateway or network required
ruff check src tests
mypy src/aitrader
```

The whole suite runs against `tests/fakes/fake_broker.py`, an in-process double, and a
scripted LLM provider — nothing here needs IBKR or Ollama reachable.
[`tests/test_architecture.py`](tests/test_architecture.py) is the one that actually keeps
the safety boundaries intact over time: it fails CI if `ib_async` or a raw `placeOrder` call
appears anywhere outside the broker adapter, or if anything outside the risk gate and the
engine's flatten paths can originate an order.

## Honest caveats

- **Fundamentals are gone from the TWS API** (`reqFundamentalData` and tick type 47 were
  removed 29 May 2026). The design is technical-only partly by necessity.
- **This will not be profitable by default.** The architecture is sound; the *strategy* is
  the hard part, and adding an LLM doesn't solve it. Expect a long paper-trading period.
- **Backtests of LLM strategies are structurally compromised** — the model's training data
  postdates any window you'd test against, so it has effectively memorized outcomes.
  Forward paper trading is the only honest evaluation.
- **Paper fills overstate quality.** IBKR paper fills simulate from top-of-book only and
  stops are always simulated. Don't calibrate slippage assumptions from paper.
- **Ollama Cloud means your positions and P&L leave your machine.** The local-Ollama adapter
  exists (`llm.provider: ollama_local`) if that matters to you.

## Project layout

```
src/aitrader/
  domain/       value objects and the LLM-boundary schemas
  broker/       the only code that talks to ib_async (port.py + ib_adapter.py)
  data/         rate limiters, SQLite + DuckDB persistence
  analytics/    indicators, feature packs, deterministic ranking
  llm/          providers, structured-output ladder, gateway, session narrative
  agents/       the four roles (strategist, trader, risk officer, reviewer)
  risk/         sizing, kill switch, the risk gate
  engine/       session calendar, decision cycle, the scheduler
  web/          FastAPI dashboard
tests/          800+ tests, all offline
config/         config.yaml, universe.csv
docker/         Dockerfile
```
