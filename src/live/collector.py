"""Live data collector for prediction market prices.

Polls Kalshi and Polymarket for current prices on matched pairs,
constructs 4-hour bars matching the train.parquet 39-column schema,
and appends them to a growing parquet dataset.

Usage:
    python -m src.live.collector              # one collection cycle
    python -m src.live.collector --loop       # continuous every 4 hours
    python -m src.live.collector --demo       # synthetic bars for testing
    python -m src.live.collector --build-mapping  # create pair_mapping.json
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features.engineering import compute_derived_features
from src.features.schemas import ALIGNED_COLUMNS, OUTPUT_COLUMNS

logger = logging.getLogger(__name__)

# Train data paths for schema reference
TRAIN_PARQUET = Path("data/processed/train.parquet")
ALL_PAIRS_JSON = Path("data/processed/all_pairs_v2.json")

# Expected 39 columns in final output
FINAL_COLUMNS = OUTPUT_COLUMNS + ["time_idx", "group_id"]


class LiveCollector:
    """Collects live prediction market prices and constructs bars.

    Fetches current prices from Kalshi (orderbook API) and Polymarket
    (Gamma API or pmxt SDK), builds snapshot bars, computes derived
    features, and appends to a growing parquet file.

    Supports two pair sources:
    - Historical pairs (default): 144 pairs from train.parquet
    - Live pairs (--live-pairs): 615+ actively-trading matched pairs
      from data/live/active_matches.json
    """

    KALSHI_ORDERBOOK_URL = (
        "https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook"
    )
    POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
    ACTIVE_MATCHES_PATH = Path("data/live/active_matches.json")

    # Maximum pairs to price per cycle. Kalshi's orderbook API rate
    # limits at ~20 req/s; at 60ms spacing, 2000 pairs takes ~2 min
    # which fits comfortably in a 15-min cycle. The pair universe can
    # be 10k+ but we only need the highest-similarity ones for trading.
    MAX_LIVE_PAIRS = 2000

    def __init__(
        self,
        live_dir: Path = Path("data/live"),
        use_live_pairs: bool = False,
        max_live_pairs: int | None = None,
    ):
        self.live_dir = Path(live_dir)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.bars_path = self.live_dir / "bars.parquet"
        self.mapping_path = self.live_dir / "pair_mapping.json"
        self._use_live_pairs = use_live_pairs
        self._max_live_pairs = max_live_pairs or self.MAX_LIVE_PAIRS

        if use_live_pairs:
            self._all_pairs = []
            self._active_pairs = self._load_live_pairs()
        else:
            self._all_pairs = self._load_all_pairs()
            self._active_pairs = self._filter_active_pairs()
        self._group_id_map = self._load_group_id_map()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_all_pairs(self) -> list[dict]:
        """Load all matched pairs from all_pairs_v2.json."""
        if not ALL_PAIRS_JSON.exists():
            logger.warning(f"Pairs file not found: {ALL_PAIRS_JSON}")
            return []
        with open(ALL_PAIRS_JSON) as f:
            return json.load(f)

    def _filter_active_pairs(self) -> dict[str, dict]:
        """Filter to the 144 pair_ids present in train.parquet.

        Returns:
            Dict mapping pair_id -> {kalshi_market_id, polymarket_market_id}
        """
        if not TRAIN_PARQUET.exists():
            logger.warning(f"Train parquet not found: {TRAIN_PARQUET}")
            return {}

        train_df = pd.read_parquet(TRAIN_PARQUET, columns=["pair_id"])
        active_pids = set(train_df["pair_id"].unique())

        active = {}
        for pair in self._all_pairs:
            pid = pair.get("pair_id", "")
            if pid in active_pids:
                active[pid] = {
                    "kalshi_market_id": pair["kalshi_market_id"],
                    "polymarket_market_id": pair["polymarket_market_id"],
                    # Convert hex token ID to decimal for Gamma API lookups
                    "polymarket_token_decimal": str(
                        int(pair["polymarket_market_id"], 16)
                    ),
                }
        logger.info(f"Active pairs loaded: {len(active)}")
        return active

    def _load_live_pairs(self) -> dict[str, dict]:
        """Load actively-trading matched pairs from active_matches.json.

        These are pairs found by semantic matching of currently-active
        Kalshi and Polymarket markets. Uses kalshi_ticker for orderbook
        API and poly_id for pmxt/Gamma price fetching.

        Pair_ids are **content-addressed** via
        :func:`src.live.pair_ids.make_pair_id`, producing strings like
        ``kxwti26apr08t10799-0x43d5953d``. This matches the format
        ``train.parquet`` already uses and ensures a given
        (kalshi_ticker, poly_id) always resolves to the same pair_id
        regardless of discovery, filter, or eviction state — unlike the
        previous ``live_{i:04d}`` scheme which silently reassigned
        positions every time the filtered-list indices shifted.

        Returns:
            Dict mapping pair_id -> {kalshi_market_id, polymarket_market_id, ...}
        """
        from src.live.pair_ids import make_pair_id
        from src.matching.quality_filter import filter_active_match

        matches_path = self.live_dir / "active_matches.json"
        if not matches_path.exists():
            matches_path = self.ACTIVE_MATCHES_PATH
        if not matches_path.exists():
            logger.warning(f"No active matches found at {matches_path}")
            return {}

        with open(matches_path) as f:
            matches = json.load(f)

        active: dict[str, dict] = {}
        rejected = 0
        skipped_empty = 0
        for m in matches:
            ok, reason = filter_active_match(m)
            if not ok:
                rejected += 1
                continue
            pair_id = make_pair_id(
                m.get("kalshi_ticker", ""),
                m.get("poly_id", ""),
            )
            if not pair_id:
                # Either kalshi_ticker or poly_id was missing/blank —
                # can't address this pair stably, skip.
                skipped_empty += 1
                continue
            active[pair_id] = {
                "kalshi_market_id": m["kalshi_ticker"],
                "polymarket_market_id": m["poly_id"],
                # For live pairs, poly_id is a condition_id (not hex token)
                # We'll use pmxt SDK or direct slug lookup for prices
                "polymarket_token_decimal": m["poly_id"],
                "is_live_pair": True,
                "kalshi_title": m.get("kalshi_title", ""),
                "poly_title": m.get("poly_title", ""),
                "similarity": m.get("similarity", 0),
                # YES/NO outcome token IDs are required for placing real
                # orders on Polymarket CLOB. discover_pairs() captures
                # them from the Gamma API; pre-2026-05-12 entries don't
                # have them and need a one-shot enrichment via
                # scripts/enrich_pair_mapping_with_tokens.py.
                "polymarket_yes_token_id": m.get("polymarket_yes_token_id", ""),
                "polymarket_no_token_id": m.get("polymarket_no_token_id", ""),
            }
        # Cap to top N by similarity to stay within Kalshi rate limits.
        # With 13k+ pairs, pricing all of them takes ~13 min and triggers
        # 429s. Top 2000 by similarity covers the best trading candidates
        # and fits in ~2 min of API calls.
        #
        # Phase 15 (2026-04-23): Reserve a commodity quota in the cap.
        # Post-Phase-15 discovery, commodity pairs (KXWTI, KXBRENT*,
        # KXAAAGAS*) have maximum similarity ~0.79 while the 2000th-best
        # pair has similarity ~0.83 — so without reservation, ALL
        # commodity pairs would be evicted by the similarity sort, and
        # COM-04 (≥ 1 non-KXWTIMAX commodity pair in pair_mapping.json)
        # would fail. Reserve up to COMMODITY_RESERVATION slots for the
        # top-similarity non-KXWTIMAX commodity pairs so Phase 15's
        # discovery fix actually reaches the live trading pipeline.
        COMMODITY_RESERVATION = 200  # ~ 6% of total slots
        COMMODITY_PREFIXES = (
            "KXWTI", "KXBRENT", "KXAAAGAS",
            "KXDIESEL", "KXHEATINGOIL", "KXCRUDE", "KXGASOLINE",
        )

        def _is_reserved_commodity(pair_info: dict) -> bool:
            """Non-KXWTIMAX commodity tickers earn a reserved slot."""
            t = pair_info.get("kalshi_market_id", "")
            return t.startswith(COMMODITY_PREFIXES) and not t.startswith("KXWTIMAX")

        if len(active) > self._max_live_pairs:
            sorted_pairs = sorted(
                active.items(),
                key=lambda kv: kv[1].get("similarity", 0),
                reverse=True,
            )
            # Partition: top-similarity commodity pairs get reserved;
            # everything else fills the remaining slots by similarity.
            commodity_sorted = [kv for kv in sorted_pairs if _is_reserved_commodity(kv[1])]
            non_commodity_sorted = [kv for kv in sorted_pairs if not _is_reserved_commodity(kv[1])]
            reserved = commodity_sorted[:COMMODITY_RESERVATION]
            remaining_slots = self._max_live_pairs - len(reserved)
            non_reserved = non_commodity_sorted[:remaining_slots]
            capped = dict(reserved + non_reserved)
            logger.info(
                "Live pairs loaded: %d (quality-filter rejected %d, "
                "skipped %d empty, capped from %d to %d by similarity; "
                "commodity-reserved=%d of %d commodity candidates)",
                len(capped), rejected, skipped_empty,
                len(active), self._max_live_pairs,
                len(reserved), len(commodity_sorted),
            )
            return capped

        logger.info(
            "Live pairs loaded: %d (quality-filter rejected %d, "
            "skipped %d with missing id)",
            len(active), rejected, skipped_empty,
        )
        return active

    def _load_group_id_map(self) -> dict[str, int]:
        """Load pair_id -> group_id mapping from train.parquet."""
        if not TRAIN_PARQUET.exists():
            return {}
        train_df = pd.read_parquet(TRAIN_PARQUET, columns=["pair_id", "group_id"])
        mapping = (
            train_df[["pair_id", "group_id"]]
            .drop_duplicates()
            .set_index("pair_id")["group_id"]
            .to_dict()
        )
        return mapping

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def fetch_kalshi_prices(self) -> dict[str, float | None]:
        """Fetch current prices from Kalshi orderbook API.

        For each active pair's Kalshi ticker, queries the orderbook endpoint.
        Extracts best yes price from the order book.

        Returns:
            Dict mapping kalshi_market_id -> price (or None if unavailable).
        """
        prices: dict[str, float | None] = {}

        for pair_id, pair_info in self._active_pairs.items():
            ticker = pair_info["kalshi_market_id"]
            try:
                url = self.KALSHI_ORDERBOOK_URL.format(ticker=ticker)
                r = requests.get(url, params={"depth": 1}, timeout=10)

                if r.status_code == 404:
                    logger.debug(f"Kalshi market not found: {ticker}")
                    prices[ticker] = None
                    time.sleep(0.06)
                    continue

                r.raise_for_status()
                data = r.json()
                ob = data.get("orderbook_fp", data.get("orderbook", {}))

                yes_dollars = ob.get("yes_dollars", []) or []
                no_dollars = ob.get("no_dollars", []) or []

                price = self._extract_kalshi_price(yes_dollars, no_dollars)
                prices[ticker] = price

            except requests.exceptions.RequestException as e:
                logger.warning(f"Kalshi API error for {ticker}: {e}")
                prices[ticker] = None
            except (KeyError, ValueError, IndexError) as e:
                logger.warning(f"Kalshi parse error for {ticker}: {e}")
                prices[ticker] = None

            # Rate limit: ~16.7 req/s (below Kalshi's 20 req/s)
            time.sleep(0.06)

        return prices

    @staticmethod
    def _extract_kalshi_price(
        yes_dollars: list, no_dollars: list
    ) -> float | None:
        """Extract best yes price from Kalshi orderbook sides.

        Logic:
        - yes_dollars entries: price someone is willing to pay for Yes
        - no_dollars entries: price someone is willing to pay for No
        - Best yes from yes_dollars: float(yes_dollars[0][0])
        - Best yes from no_dollars: 1.0 - float(no_dollars[0][0])
        - If both sides present: midpoint
        """
        best_from_yes = None
        best_from_no = None

        if yes_dollars:
            try:
                best_from_yes = float(yes_dollars[0][0])
            except (ValueError, IndexError):
                pass

        if no_dollars:
            try:
                best_from_no = 1.0 - float(no_dollars[0][0])
            except (ValueError, IndexError):
                pass

        if best_from_yes is not None and best_from_no is not None:
            return (best_from_yes + best_from_no) / 2
        elif best_from_yes is not None:
            return best_from_yes
        elif best_from_no is not None:
            return best_from_no
        return None

    def fetch_polymarket_prices(self) -> dict[str, float | None]:
        """Fetch current prices from Polymarket.

        For historical pairs: uses Gamma API with decimal clobTokenId.
        For live pairs: uses pmxt SDK which handles the lookup natively.

        Returns:
            Dict mapping polymarket_market_id -> price (or None).
        """
        if self._use_live_pairs:
            return self._fetch_polymarket_prices_pmxt()
        return self._fetch_polymarket_prices_gamma()

    def _fetch_polymarket_prices_pmxt(self) -> dict[str, float | None]:
        """Fetch Polymarket prices via Gamma API slug lookup (for live pairs).

        Uses the Gamma API with slug-based lookup for speed instead of
        individual pmxt fetch_market calls.
        """
        prices: dict[str, float | None] = {}

        # Build unique poly IDs to fetch
        unique_poly_ids = list(
            set(p["polymarket_market_id"] for p in self._active_pairs.values())
        )

        # Use Gamma API directly — fetch by condition_id in batches
        GAMMA_URL = "https://gamma-api.polymarket.com/markets"
        fetched = 0
        batch_size = 20  # Gamma supports batch queries

        for i in range(0, len(unique_poly_ids), batch_size):
            batch = unique_poly_ids[i : i + batch_size]

            for poly_id in batch:
                try:
                    # Gamma API: ``condition_ids`` (PLURAL) is the only
                    # param that does exact conditionId lookup. Confusingly:
                    #   - ``id=0x...``          → returns [] (expects numeric id)
                    #   - ``condition_id=0x...`` → returns RANDOM unrelated markets
                    #   - ``condition_ids=0x...``→ returns the exact match (1 result)
                    # Discovered empirically 2026-04-12 after 0/444 Polymarket
                    # price failures in live trading.
                    r = requests.get(
                        GAMMA_URL,
                        params={"condition_ids": poly_id},
                        timeout=10,
                    )
                    if r.status_code != 200 or not r.json():
                        # Fallback: try slug lookup for edge cases
                        r = requests.get(
                            GAMMA_URL,
                            params={"slug": poly_id},
                            timeout=10,
                        )

                    r.raise_for_status()
                    data = r.json()

                    if data and isinstance(data, list) and len(data) > 0:
                        market = data[0]
                        outcome_prices = market.get("outcomePrices", "")
                        if outcome_prices:
                            if isinstance(outcome_prices, str):
                                parsed = json.loads(outcome_prices)
                            else:
                                parsed = outcome_prices
                            yes_price = float(parsed[0])
                            if 0 < yes_price < 1:
                                prices[poly_id] = yes_price
                                fetched += 1
                            else:
                                prices[poly_id] = None
                        else:
                            prices[poly_id] = None
                    else:
                        prices[poly_id] = None
                except Exception as e:
                    logger.debug(f"Polymarket error for {poly_id[:30]}: {e}")
                    prices[poly_id] = None

                time.sleep(0.05)

            if (i + batch_size) % 100 == 0 and i > 0:
                logger.info(
                    f"  Polymarket: {fetched} prices from {i + batch_size}/{len(unique_poly_ids)}"
                )

        logger.info(f"Polymarket prices: {fetched}/{len(unique_poly_ids)} with data")
        return prices

    def _fetch_polymarket_prices_gamma(self) -> dict[str, float | None]:
        """Fetch Polymarket prices via Gamma API (for historical pairs)."""
        prices: dict[str, float | None] = {}

        for pair_id, pair_info in self._active_pairs.items():
            poly_hex = pair_info["polymarket_market_id"]
            poly_decimal = pair_info["polymarket_token_decimal"]

            try:
                r = requests.get(
                    self.POLYMARKET_GAMMA_URL,
                    params={"clob_token_ids": poly_decimal},
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()

                if data and isinstance(data, list) and len(data) > 0:
                    market = data[0]
                    outcome_prices = market.get("outcomePrices", "")
                    if outcome_prices:
                        if isinstance(outcome_prices, str):
                            parsed = json.loads(outcome_prices)
                        else:
                            parsed = outcome_prices
                        yes_price = float(parsed[0])
                        if 0 < yes_price < 1:
                            prices[poly_hex] = yes_price
                        else:
                            prices[poly_hex] = None
                    else:
                        prices[poly_hex] = None
                else:
                    prices[poly_hex] = None

            except requests.exceptions.RequestException as e:
                logger.debug(f"Polymarket API error for {poly_hex[:16]}...: {e}")
                prices[poly_hex] = None
            except (KeyError, ValueError, IndexError, json.JSONDecodeError) as e:
                logger.debug(f"Polymarket parse error for {poly_hex[:16]}...: {e}")
                prices[poly_hex] = None

            time.sleep(0.1)

        return prices

    # ------------------------------------------------------------------
    # Bar construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_snapshot_bar(
        kalshi_price: float,
        polymarket_price: float,
        pair_id: str,
        timestamp: int,
    ) -> dict:
        """Build a single snapshot bar from current prices.

        In snapshot mode, all OHLCV fields are set to the single price
        value, with volume/trade fields set to zero.

        Args:
            kalshi_price: Current Kalshi yes price (0-1 range).
            polymarket_price: Current Polymarket yes price (0-1 range).
            pair_id: Unique pair identifier.
            timestamp: Unix epoch seconds.

        Returns:
            Dict with all 31 ALIGNED_COLUMNS keys.
        """
        return {
            "timestamp": timestamp,
            # Kalshi side
            "kalshi_vwap": kalshi_price,
            "kalshi_open": kalshi_price,
            "kalshi_high": kalshi_price,
            "kalshi_low": kalshi_price,
            "kalshi_close": kalshi_price,
            "kalshi_volume": 0.0,
            "kalshi_trade_count": 0.0,
            "kalshi_dollar_volume": 0.0,
            "kalshi_buy_volume": 0.0,
            "kalshi_sell_volume": 0.0,
            "kalshi_realized_spread": 0.0,
            "kalshi_max_trade_size": 0.0,
            "kalshi_has_trade": False,
            "kalshi_hours_since_last_trade": 0.0,
            # Polymarket side
            "polymarket_vwap": polymarket_price,
            "polymarket_open": polymarket_price,
            "polymarket_high": polymarket_price,
            "polymarket_low": polymarket_price,
            "polymarket_close": polymarket_price,
            "polymarket_volume": 0.0,
            "polymarket_trade_count": 0.0,
            "polymarket_dollar_volume": 0.0,
            "polymarket_buy_volume": 0.0,
            "polymarket_sell_volume": 0.0,
            "polymarket_realized_spread": 0.0,
            "polymarket_max_trade_size": 0.0,
            "polymarket_has_trade": False,
            "polymarket_hours_since_last_trade": 0.0,
            # Spread and identity
            "spread": kalshi_price - polymarket_price,
            "pair_id": pair_id,
        }

    def assemble_bar_dataframe(self, bars: list[dict]) -> pd.DataFrame:
        """Assemble snapshot bars into a train.parquet-compatible DataFrame.

        Computes derived features via compute_derived_features(), adds
        time_idx and group_id columns. Returns a 39-column DataFrame.

        Args:
            bars: List of dicts from build_snapshot_bar().

        Returns:
            DataFrame with 39 columns matching train.parquet schema.
        """
        if not bars:
            return pd.DataFrame(columns=FINAL_COLUMNS)

        df = pd.DataFrame(bars)

        # Ensure column order matches ALIGNED_COLUMNS
        df = df[ALIGNED_COLUMNS]

        # Compute derived features (adds 6 columns -> 37 total)
        df = compute_derived_features(df)

        # Add time_idx: increment from existing bars.parquet max
        max_time_idx = self._get_max_time_idx()
        df["time_idx"] = max_time_idx + 1

        # Add group_id from train.parquet mapping
        df["group_id"] = df["pair_id"].map(self._group_id_map)
        # For unknown pair_ids, assign a new group_id
        unknown_mask = df["group_id"].isna()
        if unknown_mask.any():
            next_gid = max(self._group_id_map.values(), default=-1) + 1
            for idx in df.index[unknown_mask]:
                pid = df.at[idx, "pair_id"]
                if pid not in self._group_id_map:
                    self._group_id_map[pid] = next_gid
                    next_gid += 1
                df.at[idx, "group_id"] = self._group_id_map[pid]

        # Enforce dtypes to match train.parquet
        df = self._enforce_dtypes(df)

        # Ensure final column order
        df = df[FINAL_COLUMNS]

        return df

    def _get_max_time_idx(self) -> int:
        """Get the max time_idx from existing bars.parquet, or from train."""
        if self.bars_path.exists():
            try:
                existing = pd.read_parquet(self.bars_path, columns=["time_idx"])
                return int(existing["time_idx"].max())
            except Exception:
                pass
        # Fall back to train.parquet max
        if TRAIN_PARQUET.exists():
            try:
                train = pd.read_parquet(TRAIN_PARQUET, columns=["time_idx"])
                return int(train["time_idx"].max())
            except Exception:
                pass
        return -1

    @staticmethod
    def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """Enforce dtypes to match train.parquet schema."""
        float_cols = [
            c for c in df.columns
            if c not in ("timestamp", "pair_id", "kalshi_has_trade",
                         "polymarket_has_trade", "time_idx", "group_id")
        ]
        for col in float_cols:
            df[col] = df[col].astype("float64")

        df["timestamp"] = df["timestamp"].astype("int64")
        df["time_idx"] = df["time_idx"].astype("int64")
        df["group_id"] = df["group_id"].astype("int64")
        df["kalshi_has_trade"] = df["kalshi_has_trade"].astype("bool")
        df["polymarket_has_trade"] = df["polymarket_has_trade"].astype("bool")
        df["pair_id"] = df["pair_id"].astype("object")

        return df

    def append_to_parquet(self, df: pd.DataFrame, path: Path | None = None) -> None:
        """Append bars DataFrame to a parquet file.

        Creates a new file if none exists. Appends and deduplicates
        if the file already exists. Always sorts by (pair_id, time_idx).

        Args:
            df: DataFrame with 39 columns.
            path: Output path. Defaults to self.bars_path.
        """
        if path is None:
            path = self.bars_path
        path = Path(path)

        if df.empty:
            logger.info("No bars to append.")
            return

        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df.copy()

        combined = combined.sort_values(["pair_id", "time_idx"]).reset_index(drop=True)
        combined.to_parquet(path, index=False)
        logger.info(f"Wrote {len(combined)} bars to {path}")

    # ------------------------------------------------------------------
    # Pair mapping
    # ------------------------------------------------------------------

    def build_pair_mapping(self) -> dict:
        """Build and save pair mapping to JSON.

        Maps pair_id -> {kalshi_market_id, polymarket_market_id,
        polymarket_token_decimal, group_id}.

        Returns:
            The mapping dict.
        """
        mapping = {}
        for pair_id, info in self._active_pairs.items():
            mapping[pair_id] = {
                "kalshi_market_id": info["kalshi_market_id"],
                "polymarket_market_id": info["polymarket_market_id"],
                "polymarket_token_decimal": info["polymarket_token_decimal"],
                "group_id": self._group_id_map.get(pair_id),
            }

        with open(self.mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"Pair mapping saved: {len(mapping)} pairs -> {self.mapping_path}")
        return mapping

    # ------------------------------------------------------------------
    # Collection cycle
    # ------------------------------------------------------------------

    def collect_once(self) -> int:
        """Run one collection cycle.

        Fetches prices from both platforms, builds snapshot bars for
        pairs with both prices available, assembles DataFrame, and
        appends to bars.parquet.

        Returns:
            Number of bars collected.
        """
        ts = int(time.time())
        logger.info(f"Starting collection cycle at {ts}")

        # Fetch prices from both platforms
        kalshi_prices = self.fetch_kalshi_prices()
        poly_prices = self.fetch_polymarket_prices()

        # Build bars for pairs with both prices
        bars = []
        skipped = 0
        for pair_id, pair_info in self._active_pairs.items():
            k_ticker = pair_info["kalshi_market_id"]
            p_hex = pair_info["polymarket_market_id"]

            k_price = kalshi_prices.get(k_ticker)
            p_price = poly_prices.get(p_hex)

            if k_price is None or p_price is None:
                skipped += 1
                continue

            bar = self.build_snapshot_bar(k_price, p_price, pair_id, ts)
            bars.append(bar)

        if not bars:
            logger.info(
                f"No active pairs found with current prices on both platforms. "
                f"Skipped {skipped} pairs."
            )
            return 0

        df = self.assemble_bar_dataframe(bars)
        self.append_to_parquet(df)
        logger.info(f"Collected {len(bars)} bars for {len(bars)} pairs at {ts}")
        return len(bars)

    def collect_demo(self, n_pairs: int = 5) -> int:
        """Create synthetic bars for pipeline demonstration.

        Generates fake prices for n_pairs active pairs to test the
        full pipeline end-to-end without needing live API data.

        Args:
            n_pairs: Number of pairs to generate bars for.

        Returns:
            Number of bars created.
        """
        ts = int(time.time())
        logger.info(f"Demo mode: generating {n_pairs} synthetic bars")

        pair_ids = list(self._active_pairs.keys())[:n_pairs]
        if not pair_ids:
            logger.error("No active pairs available for demo mode.")
            return 0

        random.seed(42)
        bars = []
        for pid in pair_ids:
            k_price = 0.5 + random.uniform(-0.15, 0.15)
            p_price = 0.5 + random.uniform(-0.15, 0.15)
            k_price = max(0.01, min(0.99, k_price))
            p_price = max(0.01, min(0.99, p_price))
            bar = self.build_snapshot_bar(k_price, p_price, pid, ts)
            bars.append(bar)

        df = self.assemble_bar_dataframe(bars)
        self.append_to_parquet(df)
        logger.info(f"Demo: created {len(bars)} synthetic bars")
        return len(bars)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def run_loop(self, interval: int = 14400) -> None:
        """Run collection continuously at the specified interval.

        Args:
            interval: Seconds between collection cycles (default 4h).
        """
        logger.info(f"Starting continuous collection (interval={interval}s)")
        while True:
            try:
                n = self.collect_once()
                logger.info(f"Cycle complete: {n} bars. Next in {interval}s.")
            except Exception as e:
                logger.error(f"Collection cycle error: {e}", exc_info=True)
            time.sleep(interval)


def main():
    """CLI entry point for the live collector."""
    parser = argparse.ArgumentParser(
        description="Live prediction market data collector",
        prog="python -m src.live.collector",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously (default: one cycle)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=14400,
        help="Seconds between cycles in loop mode (default: 14400 = 4h)",
    )
    parser.add_argument(
        "--build-mapping",
        action="store_true",
        help="Build pair_mapping.json and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic bars for pipeline testing",
    )
    parser.add_argument(
        "--demo-pairs",
        type=int,
        default=5,
        help="Number of pairs for demo mode (default: 5)",
    )
    parser.add_argument(
        "--live-pairs",
        action="store_true",
        help="Use 615+ actively-trading matched pairs (from active_matches.json) "
        "instead of the historical 144 pairs",
    )
    parser.add_argument(
        "--live-dir",
        type=str,
        default="data/live",
        help="Output directory for live data (default: data/live)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    collector = LiveCollector(
        live_dir=Path(args.live_dir),
        use_live_pairs=args.live_pairs,
    )

    if args.build_mapping:
        mapping = collector.build_pair_mapping()
        print(f"Pair mapping created: {len(mapping)} pairs")
        return

    if args.demo:
        n = collector.collect_demo(n_pairs=args.demo_pairs)
        print(f"Demo: created {n} synthetic bars")
        return

    if args.loop:
        collector.run_loop(interval=args.interval)
    else:
        n = collector.collect_once()
        print(f"Collected {n} bars")


if __name__ == "__main__":
    main()
