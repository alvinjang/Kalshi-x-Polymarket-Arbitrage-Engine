"""SQLite-backed position manager for multi-bar prediction market trading.

Tracks open positions across cron/process restarts, updates position
metrics each cycle, and checks 5 independent exit rules with concrete
thresholds.  Closed positions are recorded in both a SQLite table and
an append-only JSONL backup file.

Exit rules (priority order):
    1. RESOLUTION_EXIT  -- < 24 hours to resolution
    2. STOP_LOSS        -- spread widened > 30% beyond entry
    3. TAKE_PROFIT      -- spread narrowed > 50% from entry
    4. MOMENTUM         -- 3 consecutive bars moving against position
    5. TIME_STOP        -- bars_held >= tier-specific max hold
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import enum
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ExitReason(enum.Enum):
    """Reason a position was closed."""

    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LOSS = "STOP_LOSS"
    MOMENTUM = "MOMENTUM"
    TIME_STOP = "TIME_STOP"
    RESOLUTION_EXIT = "RESOLUTION_EXIT"
    MANUAL = "MANUAL"


# Maximum bars before time-stop fires, by contract tier.
MAX_BARS: dict[str, int] = {
    "DAILY": 4,
    "WEEKLY": 24,
    "MONTHLY": 168,
    "QUARTERLY": 500,
    "UNKNOWN": 168,
}


@dataclass
class Position:
    """In-memory representation of an open position row."""

    pair_id: str
    kalshi_ticker: str
    direction: str  # "short_spread" or "long_spread"
    entry_spread: float
    entry_time: str  # ISO 8601 UTC
    entry_kalshi_price: float
    entry_poly_price: float
    current_spread: float
    bars_held: int
    max_spread_since_entry: float
    min_spread_since_entry: float
    unrealized_pnl: float
    tier: str  # DAILY / WEEKLY / MONTHLY / QUARTERLY / UNKNOWN
    bar_interval_seconds: int
    resolution_date: Optional[str]  # ISO 8601 or None
    spread_history: str  # JSON array of last N spreads


class PositionManager:
    """SQLite-backed manager for open and closed positions.

    Args:
        db_path: Path to the SQLite database file.
        history_jsonl_path: Path to the append-only JSONL backup of
            closed positions.
    """

    def __init__(
        self,
        db_path: str = "data/live/positions.db",
        history_jsonl_path: str = "data/live/position_history.jsonl",
    ) -> None:
        self.db_path = db_path
        self.history_jsonl_path = history_jsonl_path

        # Ensure parent directories exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.history_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        # Use TRUNCATE journaling instead of WAL: WAL requires shared memory
        # primitives that aren't available on networked filesystems (NFS),
        # and BU SCC's /usr4 is NFS-mounted. TRUNCATE is the fastest
        # NFS-compatible journal mode and is equivalent for our
        # single-writer cron workload.
        self._conn.execute("PRAGMA journal_mode=TRUNCATE")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create positions and closed_positions tables if not present."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS positions (
                pair_id TEXT PRIMARY KEY,
                kalshi_ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_spread REAL NOT NULL,
                entry_time TEXT NOT NULL,
                entry_kalshi_price REAL NOT NULL,
                entry_poly_price REAL NOT NULL,
                current_spread REAL NOT NULL,
                bars_held INTEGER NOT NULL DEFAULT 0,
                max_spread_since_entry REAL NOT NULL,
                min_spread_since_entry REAL NOT NULL,
                unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                tier TEXT NOT NULL,
                bar_interval_seconds INTEGER NOT NULL,
                resolution_date TEXT,
                spread_history TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS closed_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                kalshi_ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_spread REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_kalshi_price REAL NOT NULL,
                entry_poly_price REAL NOT NULL,
                exit_spread REAL NOT NULL,
                bars_held INTEGER NOT NULL,
                realized_pnl REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                tier TEXT NOT NULL,
                max_spread REAL NOT NULL,
                min_spread REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Open / update / close
    # ------------------------------------------------------------------

    def open_position(
        self,
        pair_id: str,
        kalshi_ticker: str,
        direction: str,
        entry_spread: float,
        kalshi_price: float,
        poly_price: float,
        tier: str,
        bar_interval_seconds: int,
        resolution_date: Optional[str] = None,
    ) -> None:
        """Insert a new open position into the database."""
        entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._conn.execute(
            """
            INSERT INTO positions (
                pair_id, kalshi_ticker, direction, entry_spread, entry_time,
                entry_kalshi_price, entry_poly_price, current_spread,
                bars_held, max_spread_since_entry, min_spread_since_entry,
                unrealized_pnl, tier, bar_interval_seconds, resolution_date,
                spread_history
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 0.0, ?, ?, ?, '[]')
            """,
            (
                pair_id,
                kalshi_ticker,
                direction,
                entry_spread,
                entry_time,
                kalshi_price,
                poly_price,
                entry_spread,  # current_spread starts at entry
                entry_spread,  # max starts at entry
                entry_spread,  # min starts at entry
                tier,
                bar_interval_seconds,
                resolution_date,
            ),
        )
        self._conn.commit()

    def update_position(self, pair_id: str, current_spread: float) -> None:
        """Update an open position with the latest spread observation.

        Increments bars_held, updates min/max spread, computes unrealized
        P&L, and appends current_spread to the spread_history JSON array
        (keeping the last 10 entries).
        """
        row = self._conn.execute(
            "SELECT * FROM positions WHERE pair_id = ?", (pair_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No open position for pair_id={pair_id!r}")

        direction = row["direction"]
        entry_spread = row["entry_spread"]
        bars_held = row["bars_held"] + 1
        max_spread = max(row["max_spread_since_entry"], current_spread)
        min_spread = min(row["min_spread_since_entry"], current_spread)

        # Direction-aware unrealized P&L
        if direction == "short_spread":
            unrealized_pnl = entry_spread - current_spread
        else:  # long_spread
            unrealized_pnl = current_spread - entry_spread

        # Spread history: append and keep last 10
        history: list[float] = json.loads(row["spread_history"])
        history.append(current_spread)
        history = history[-10:]
        history_json = json.dumps(history)

        self._conn.execute(
            """
            UPDATE positions SET
                current_spread = ?,
                bars_held = ?,
                max_spread_since_entry = ?,
                min_spread_since_entry = ?,
                unrealized_pnl = ?,
                spread_history = ?
            WHERE pair_id = ?
            """,
            (
                current_spread,
                bars_held,
                max_spread,
                min_spread,
                unrealized_pnl,
                history_json,
                pair_id,
            ),
        )
        self._conn.commit()

    def close_position(
        self,
        pair_id: str,
        reason: ExitReason,
        exit_spread: float,
        exit_time: str,
    ) -> dict:
        """Close an open position: move to closed_positions + JSONL backup.

        Args:
            pair_id: The position to close.
            reason: Why the position was closed.
            exit_spread: Spread at the time of exit.
            exit_time: ISO 8601 UTC timestamp of exit.

        Returns:
            Dict with the closed position record.
        """
        row = self._conn.execute(
            "SELECT * FROM positions WHERE pair_id = ?", (pair_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No open position for pair_id={pair_id!r}")

        direction = row["direction"]
        entry_spread = row["entry_spread"]

        # Direction-aware realized P&L
        if direction == "short_spread":
            realized_pnl = entry_spread - exit_spread
        else:  # long_spread
            realized_pnl = exit_spread - entry_spread

        record = {
            "pair_id": pair_id,
            "kalshi_ticker": row["kalshi_ticker"],
            "direction": direction,
            "entry_spread": entry_spread,
            "entry_time": row["entry_time"],
            "exit_time": exit_time,
            "entry_kalshi_price": row["entry_kalshi_price"],
            "entry_poly_price": row["entry_poly_price"],
            "exit_spread": exit_spread,
            "bars_held": row["bars_held"],
            "realized_pnl": realized_pnl,
            "exit_reason": reason.value,
            "tier": row["tier"],
            "max_spread": row["max_spread_since_entry"],
            "min_spread": row["min_spread_since_entry"],
        }

        # Insert into closed_positions
        self._conn.execute(
            """
            INSERT INTO closed_positions (
                pair_id, kalshi_ticker, direction, entry_spread, entry_time,
                exit_time, entry_kalshi_price, entry_poly_price, exit_spread,
                bars_held, realized_pnl, exit_reason, tier, max_spread,
                min_spread
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["pair_id"],
                record["kalshi_ticker"],
                record["direction"],
                record["entry_spread"],
                record["entry_time"],
                record["exit_time"],
                record["entry_kalshi_price"],
                record["entry_poly_price"],
                record["exit_spread"],
                record["bars_held"],
                record["realized_pnl"],
                record["exit_reason"],
                record["tier"],
                record["max_spread"],
                record["min_spread"],
            ),
        )

        # Delete from open positions
        self._conn.execute(
            "DELETE FROM positions WHERE pair_id = ?", (pair_id,)
        )
        self._conn.commit()

        # Append to JSONL backup
        with open(self.history_jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> dict[str, Position]:
        """Return all open positions as {pair_id: Position}."""
        rows = self._conn.execute("SELECT * FROM positions").fetchall()
        result: dict[str, Position] = {}
        for row in rows:
            result[row["pair_id"]] = Position(
                pair_id=row["pair_id"],
                kalshi_ticker=row["kalshi_ticker"],
                direction=row["direction"],
                entry_spread=row["entry_spread"],
                entry_time=row["entry_time"],
                entry_kalshi_price=row["entry_kalshi_price"],
                entry_poly_price=row["entry_poly_price"],
                current_spread=row["current_spread"],
                bars_held=row["bars_held"],
                max_spread_since_entry=row["max_spread_since_entry"],
                min_spread_since_entry=row["min_spread_since_entry"],
                unrealized_pnl=row["unrealized_pnl"],
                tier=row["tier"],
                bar_interval_seconds=row["bar_interval_seconds"],
                resolution_date=row["resolution_date"],
                spread_history=row["spread_history"],
            )
        return result

    def has_position(self, pair_id: str) -> bool:
        """Check whether a position is currently open for pair_id."""
        row = self._conn.execute(
            "SELECT 1 FROM positions WHERE pair_id = ?", (pair_id,)
        ).fetchone()
        return row is not None

    def get_closed_positions(self) -> list[dict]:
        """Return all closed positions as a list of dicts, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM closed_positions ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Exit rule checks (implemented in Task 2)
    # ------------------------------------------------------------------

    def _check_take_profit(self, pos: Position) -> bool:
        """Take profit: spread magnitude narrowed >= 50% from entry.

        Compares MAGNITUDES so the check is correct regardless of
        whether entry_spread is stored signed or (historically) as abs.
        Protects against the Task #28 live_0237 bug class.

        2026-05-12 retest: a sweep over 49,727 historical positions
        (scripts/backtest_strategy_changes.py) shows estimated P&L is
        monotonically increasing as TP_RATIO → 0:
            0.20 = +$288   0.40 = +$233   0.50 = +$194
            0.60 = +$156   0.70 = +$105   0.90 = -$43
        Looser TP fires more often but exits trades that were on track
        for larger profit. Tighter TP captures only strong moves. The
        original 0.5 is the conservative middle; tightening further is
        a data-supported follow-up but requires live validation.
        """
        return abs(pos.current_spread) <= abs(pos.entry_spread) * 0.5

    def _check_stop_loss(self, pos: Position) -> bool:
        """Stop loss: spread magnitude widened > 30% beyond entry."""
        return abs(pos.current_spread) > abs(pos.entry_spread) * 1.3

    def _check_time_stop(self, pos: Position) -> bool:
        """Time stop: bars_held >= max_bars for tier."""
        max_bars = MAX_BARS.get(pos.tier, 168)
        return pos.bars_held >= max_bars

    def _check_momentum(self, pos: Position) -> bool:
        """Momentum exit: 3 consecutive bars moving against position.

        Requires at least 4 entries in spread_history (to see 3 diffs).
        For short_spread: 3 consecutive increases (spread going up = against us).
        For long_spread: 3 consecutive decreases (spread going down = against us).
        """
        history: list[float] = json.loads(pos.spread_history)
        if len(history) < 4:
            return False

        # Check last 3 diffs
        last4 = history[-4:]
        diffs = [last4[i + 1] - last4[i] for i in range(3)]

        if pos.direction == "short_spread":
            # Against us = spread increasing
            return all(d > 0 for d in diffs)
        else:  # long_spread
            # Against us = spread decreasing
            return all(d < 0 for d in diffs)

    def _check_resolution(self, pos: Position, now: datetime) -> bool:
        """Resolution proximity: < 24 hours to resolution.

        ``pos.resolution_date`` is an ISO 8601 string that MAY or MAY NOT
        carry a timezone. Contract classifier's ``parse_resolution_date``
        returns naive datetimes (year/month/day/23/59 with no tzinfo),
        whose ``.isoformat()`` has no ``Z`` or ``+00:00`` suffix. When
        parsed back here we need to force UTC so we can subtract it from
        the aware ``now`` without a TypeError.
        """
        if pos.resolution_date is None:
            return False
        try:
            res_dt = datetime.fromisoformat(
                pos.resolution_date.replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            return False
        # Defensive: if the parsed date is naive, assume UTC.
        if res_dt.tzinfo is None:
            res_dt = res_dt.replace(tzinfo=timezone.utc)
        # Same defensive treatment for 'now' if a caller ever passes naive.
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        hours_remaining = (res_dt - now).total_seconds() / 3600.0
        return hours_remaining < 24.0

    def check_exits(
        self, pair_id: str, now: datetime
    ) -> Optional[ExitReason]:
        """Check all exit rules for a position in priority order.

        Priority: RESOLUTION > STOP_LOSS > TAKE_PROFIT > MOMENTUM > TIME_STOP

        Returns:
            The first triggered ExitReason, or None if no exit triggered.
        """
        positions = self.get_open_positions()
        if pair_id not in positions:
            return None
        pos = positions[pair_id]

        if self._check_resolution(pos, now):
            return ExitReason.RESOLUTION_EXIT
        if self._check_stop_loss(pos):
            return ExitReason.STOP_LOSS
        if self._check_take_profit(pos):
            return ExitReason.TAKE_PROFIT
        if self._check_momentum(pos):
            return ExitReason.MOMENTUM
        if self._check_time_stop(pos):
            return ExitReason.TIME_STOP
        return None

    def check_all_exits(
        self, now: datetime
    ) -> list[tuple[str, ExitReason]]:
        """Check exit rules for all open positions.

        Returns:
            List of (pair_id, ExitReason) for positions that triggered an exit.
        """
        triggered: list[tuple[str, ExitReason]] = []
        positions = self.get_open_positions()
        for pair_id in positions:
            reason = self.check_exits(pair_id, now)
            if reason is not None:
                triggered.append((pair_id, reason))
        return triggered
