"""Pre-trade risk gate for live (real-money) trading.

Default-OFF safety design: this module is consulted before EVERY order
placement. If LIVE_TRADING env var is not exactly the string "true",
every check returns rejection — the live trading path is dead unless
explicitly armed.

Three hard limits enforced:
    1. Per-trade size cap     (default $2/leg)
    2. Daily realized-loss kill switch (default -$10 since UTC midnight)
    3. Per-exchange exposure cap (default $50 = full sub-bankroll)

Risk decisions are PURE and STATELESS w.r.t. orders — input is the
proposed trade + the realized P&L history; output is permit/reject
with a reason string. Persistence (today's realized losses) is read
from data/live/position_history.jsonl, so the gate works correctly
across process restarts.

Kill-switch behavior: when triggered, NEW ENTRIES are blocked, but
EXISTING POSITIONS are allowed to exit per their normal rules. Force-
flattening real money positions in a widening-spread regime can double
the realized loss vs. holding to a more favorable exit.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# --- Canary config (2026-05-12) --------------------------------------------
DEFAULT_PER_TRADE_USD = 2.0          # 4% of $50 sub-bankroll per leg
DEFAULT_DAILY_LOSS_LIMIT = 10.0      # halt new entries when realized losses ≥ this
DEFAULT_EXPOSURE_CAP_PER_EXCHANGE = 50.0   # full bankroll per exchange


@dataclass(frozen=True)
class RiskDecision:
    permit: bool
    reason: str


def is_live_trading_armed() -> bool:
    """Single source of truth for whether real orders may be placed.

    Defaults to False. Only True when:
        - LIVE_TRADING env var is the exact string "true"
        - AND EMERGENCY_HALT env var is NOT set to "true"
    """
    if os.environ.get("EMERGENCY_HALT", "").lower() == "true":
        return False
    return os.environ.get("LIVE_TRADING", "").lower() == "true"


def _utc_day_start_iso() -> str:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def realized_loss_today(history_path: Path | str = Path("data/live/position_history.jsonl")) -> float:
    """Sum of realized losses (negative P&L) since UTC midnight.

    Returns a POSITIVE number representing how much we're down today.
    Returns 0.0 if file missing or no records.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        return 0.0

    cutoff = _utc_day_start_iso()
    total_loss = 0.0
    try:
        with open(history_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "realized_pnl" not in r or "exit_time" not in r:
                    continue
                if r["exit_time"] < cutoff:
                    continue
                pnl = float(r["realized_pnl"])
                if pnl < 0:
                    total_loss += -pnl
    except OSError as e:
        logger.warning("Failed to read %s: %s", history_path, e)
        return 0.0

    return total_loss


def current_exposure(history_path: Path | str = Path("data/live/position_history.jsonl"),
                     positions_db: Path | str = Path("data/live/positions.db")) -> dict[str, float]:
    """Open dollar exposure broken down by exchange.

    Reads the SQLite positions table for currently-open positions and
    sums the cost basis on each exchange. Currently approximates cost
    basis as entry_kalshi_price + entry_poly_price per position,
    multiplied by the size committed to that leg (assumed equal to
    DEFAULT_PER_TRADE_USD when live trading is armed).
    """
    import sqlite3

    positions_db = Path(positions_db)
    if not positions_db.exists():
        return {"kalshi": 0.0, "polymarket": 0.0}

    try:
        with sqlite3.connect(positions_db) as conn:
            cur = conn.execute(
                "SELECT entry_kalshi_price, entry_poly_price FROM positions"
            )
            rows = cur.fetchall()
    except sqlite3.Error as e:
        logger.warning("Failed to read positions.db: %s", e)
        return {"kalshi": 0.0, "polymarket": 0.0}

    # We don't store actual filled USD per leg yet (paper era), so we
    # approximate. When live orders land, the order-confirmation path
    # should update this.
    kalshi_exposure = sum(float(k) * DEFAULT_PER_TRADE_USD for k, _ in rows if k is not None)
    poly_exposure = sum(float(p) * DEFAULT_PER_TRADE_USD for _, p in rows if p is not None)
    return {"kalshi": kalshi_exposure, "polymarket": poly_exposure}


def check_pretrade(
    trade_size_usd: float,
    exchange: str,
    *,
    per_trade_cap: float = DEFAULT_PER_TRADE_USD,
    daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT,
    exposure_cap: float = DEFAULT_EXPOSURE_CAP_PER_EXCHANGE,
    history_path: Path | str = Path("data/live/position_history.jsonl"),
    positions_db: Path | str = Path("data/live/positions.db"),
) -> RiskDecision:
    """Run all pre-trade risk gates. Returns RiskDecision(permit, reason).

    Order of checks matters: most-fundamental first so error messages
    are informative even if many limits would have been breached.
    """
    # Gate 0: live trading must be armed
    if not is_live_trading_armed():
        return RiskDecision(False, "LIVE_TRADING env var not set to 'true' (default-OFF)")

    # Gate 1: per-trade size
    if trade_size_usd > per_trade_cap:
        return RiskDecision(
            False,
            f"trade_size_usd={trade_size_usd:.2f} > per_trade_cap={per_trade_cap:.2f}",
        )

    # Gate 2: daily loss kill switch
    loss_today = realized_loss_today(history_path)
    if loss_today >= daily_loss_limit:
        return RiskDecision(
            False,
            f"daily loss kill-switch triggered: realized loss today "
            f"${loss_today:.2f} >= limit ${daily_loss_limit:.2f}",
        )

    # Gate 3: per-exchange exposure cap
    exposure = current_exposure(history_path, positions_db)
    current = exposure.get(exchange.lower(), 0.0)
    if current + trade_size_usd > exposure_cap:
        return RiskDecision(
            False,
            f"exposure cap on {exchange}: current ${current:.2f} + ${trade_size_usd:.2f} "
            f"> ${exposure_cap:.2f}",
        )

    return RiskDecision(True, "ok")
