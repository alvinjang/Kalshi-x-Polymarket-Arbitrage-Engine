"""Continuous market discovery for cross-platform arbitrage pairs.

The live trading edge comes from short-dated contracts (daily/weekly oil,
Fed decisions, expiring sports). Those contracts resolve and disappear,
so the pair universe in ``data/live/active_matches.json`` decays unless
we keep searching for new matches.

This module runs periodically (via GitHub Actions or cron) and:

  1. Pulls currently-active Kalshi markets (``/markets?status=open``).
  2. Pulls currently-active Polymarket markets (Gamma API,
     ``active=true&closed=false&archived=false``).
  3. Generates candidate pairs via keyword overlap (cheap pre-filter).
  4. Scores candidates with sentence-transformers cosine similarity.
  5. Filters via ``src.matching.quality_filter.filter_active_match`` to
     drop structural mismatches (NBA season-wins vs champion, Fed year
     mismatches, cabinet vs nomination).
  6. Upserts results into ``active_matches.json`` — preserving existing
     pair_ids (array index) so the collector and paper trader keep
     working without reindexing.

Upsert invariants (see tests/live/test_market_discovery.py):

  - Existing pair_ids NEVER change across runs.
  - New pairs are appended at the end with a fresh ``discovered_at``.
  - Each pair tracks a ``last_seen`` timestamp that refreshes only when
    the pair is present in the current discovery run. Pairs that haven't
    been seen in N runs can later be garbage-collected by a downstream
    process — this module just marks, it doesn't delete.

Heavy dependencies (sentence_transformers, requests) are imported lazily
inside the functions that use them so the pure upsert logic can be
unit-tested without the full CI stack.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Numeric fields that should be refreshed when we rediscover an existing
# pair. String fields (kalshi_ticker, poly_id, titles) are the stable
# identity of the match and should not change.
_REFRESHABLE_FIELDS = (
    "similarity",
    "spread",
    "kalshi_mid",
    "kalshi_vol",
    "poly_price",
    "poly_vol",
    "kalshi_title",
    "poly_title",
)


def make_match_key(match: dict) -> tuple[str, str]:
    """Stable identity key for a match.

    Two entries with the same (kalshi_ticker, poly_id) are the same
    underlying pair — even if their prices/similarities differ.
    """
    return (
        (match.get("kalshi_ticker") or "").strip(),
        str(match.get("poly_id") or "").strip(),
    )


# Default pairs-not-seen-recently TTL. Discovery runs every 3 hours so
# 7 days = 56 runs; a pair needs to miss 56 consecutive runs to get
# tombstoned. Empirically this drops the 615 pre-upsert-era zombie
# pairs (no last_seen at all) on the first post-fix discovery run
# without touching any currently-active pair.
DEFAULT_EVICTION_TTL_SECONDS = 7 * 24 * 3600


def _is_tombstone(m: dict) -> bool:
    """Return True if a merged entry is an evicted tombstone."""
    return bool(m.get("evicted"))


def upsert_active_matches(
    existing: list[dict],
    new_matches: list[dict],
    now_ts: int | None = None,
    eviction_ttl_seconds: int = DEFAULT_EVICTION_TTL_SECONDS,
    protected_indices: set[int] | None = None,
) -> tuple[list[dict], dict]:
    """Merge fresh discovery results into the existing match list.

    Args:
        existing: Current contents of active_matches.json (list order =
            pair_id ordering; must not be reshuffled).
        new_matches: Matches from the most recent discovery run.
        now_ts: Unix timestamp for discovered_at/last_seen. Defaults to
            current time when omitted.
        eviction_ttl_seconds: Pairs not seen in this many seconds get
            tombstoned in place. Tombstones preserve the underlying
            (kalshi_ticker, poly_id) identity via prev_kalshi_ticker /
            prev_poly_id so a later rediscovery revives the SAME
            content-addressed pair_id (see src/live/pair_ids.py).
            bars.parquet and positions.db references survive eviction
            and revival unchanged. The tombstone itself is rejected by
            the collector's quality filter because kalshi_ticker is
            cleared.
        protected_indices: Set of array indices that must never be
            tombstoned, even if stale. Used by ``run_discovery`` to
            pass in indices referenced by open positions so the
            collector can keep pricing until the position closes.

    Returns:
        (merged, stats) where:
          - merged: updated list with existing indices preserved and
            new pairs appended.
          - stats: {added, updated, stale, evicted, total} counts.
    """
    if now_ts is None:
        now_ts = int(time.time())

    # Start with a deep-ish copy of existing so we can update in place.
    merged: list[dict] = [dict(m) for m in existing]

    # Build two indices:
    #   active_by_key:     live (non-tombstone) pair key -> index
    #   evicted_by_prev:   tombstone prev_key -> index (for revival)
    active_by_key: dict[tuple[str, str], int] = {}
    evicted_by_prev: dict[tuple[str, str], int] = {}
    for i, m in enumerate(merged):
        if _is_tombstone(m):
            prev_key = (
                (m.get("prev_kalshi_ticker") or "").strip(),
                str(m.get("prev_poly_id") or "").strip(),
            )
            if prev_key != ("", ""):
                evicted_by_prev[prev_key] = i
        else:
            active_by_key[make_match_key(m)] = i

    # Dedupe new_matches within this batch (last-write-wins on same key).
    new_by_key: dict[tuple[str, str], dict] = {}
    for m in new_matches:
        new_by_key[make_match_key(m)] = m

    added = 0
    updated = 0
    revived = 0

    for key, nm in new_by_key.items():
        if key in active_by_key:
            # Existing live pair: refresh numeric + title fields, bump last_seen.
            idx = active_by_key[key]
            target = merged[idx]
            for field in _REFRESHABLE_FIELDS:
                if field in nm:
                    target[field] = nm[field]
            target["last_seen"] = now_ts
            target.setdefault("discovered_at", target.get("discovered_at", now_ts))
            updated += 1
        elif key in evicted_by_prev:
            # Tombstone for this exact pair is being rediscovered.
            # Replace the tombstone with a fresh live entry at the
            # same index so pair_id stays stable.
            idx = evicted_by_prev[key]
            revived_entry = dict(nm)
            revived_entry["discovered_at"] = now_ts
            revived_entry["last_seen"] = now_ts
            merged[idx] = revived_entry
            revived += 1
        else:
            # Brand-new pair: append with fresh timestamps.
            appended = dict(nm)
            appended["discovered_at"] = now_ts
            appended["last_seen"] = now_ts
            merged.append(appended)
            added += 1

    # --- Tombstone pairs that haven't been seen for > TTL ---
    # Runs AFTER the update/add phase so pairs rediscovered this run
    # can't be evicted (their last_seen was just refreshed).
    protected = protected_indices or set()
    newly_evicted = 0
    protected_skipped = 0
    for i, pair in enumerate(merged):
        if _is_tombstone(pair):
            continue
        last_seen = pair.get("last_seen") or 0
        # Zero or missing last_seen means the pair predates the upsert
        # timestamping era — these are all zombies and should be evicted.
        stale = last_seen == 0 or (now_ts - last_seen) > eviction_ttl_seconds
        if not stale:
            continue
        if i in protected:
            # Pair has an open position — keep it alive so the collector
            # can price it until the position closes. Log so we can audit
            # why a stale pair is still in the active set.
            protected_skipped += 1
            continue
        merged[i] = {
            "evicted": True,
            "evicted_at": now_ts,
            "kalshi_ticker": "",   # triggers missing_kalshi_ticker filter
            "poly_id": "",
            "prev_kalshi_ticker": pair.get("kalshi_ticker", ""),
            "prev_poly_id": pair.get("poly_id", ""),
        }
        newly_evicted += 1

    # Legacy: count stale pairs (existing but not seen in this batch)
    # for backward-compatible stats consumers. "stale" predates
    # eviction and just means "wasn't in the current new_matches batch";
    # it's always >= newly_evicted.
    new_key_set = set(new_by_key.keys())
    stale = sum(1 for k in active_by_key if k not in new_key_set)

    active_count = sum(1 for m in merged if not _is_tombstone(m))
    tombstone_count = len(merged) - active_count

    stats = {
        "total": len(merged),
        "active": active_count,
        "tombstones": tombstone_count,
        "added": added,
        "updated": updated,
        "revived": revived,
        "stale": stale,
        "evicted": newly_evicted,
        "protected_skipped": protected_skipped,
        "run_ts": now_ts,
    }
    logger.info(
        "upsert_active_matches: added=%d updated=%d revived=%d evicted=%d "
        "protected=%d stale=%d active=%d tombstones=%d total=%d",
        added, updated, revived, newly_evicted, protected_skipped,
        stale, active_count, tombstone_count, len(merged),
    )
    return merged, stats


# ----------------------------------------------------------------------
# Live market fetching (lazy-imports requests/Gamma)
# ----------------------------------------------------------------------


KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
POLY_GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

# Kalshi categories to scan. Matches src/data/kalshi.py defaults plus
# Politics (for nomination / election markets) and Climate.
#
# Phase 15 (2026-04-23): Added "Commodities" per 15-01-DIAGNOSTIC.md H1
# CONFIRMED. Kalshi migrated daily WTI / Brent / grain / metal series
# into a dedicated Commodities category (47 series live; 125+ open oil
# markets). Prior to this fix, /series?category=Financials returned 0
# commodity-prefixed series, so KXWTI / KXWTIW / KXBRENTMON / KXBRENTW
# / KXWTIMINM / KXWTIMIN never entered the discovery candidate pool —
# leaving KXWTIMAX-26DEC31-T130 to incorrectly match against Bitcoin-$130K
# (poly_id=0x885a6abe...a859, similarity=0.707). See Plan 15-03.
KALSHI_DISCOVERY_CATEGORIES = ("Economics", "Crypto", "Financials", "Politics", "Climate", "Commodities")

# Rate-limit handling for Kalshi /events.
# Kalshi throttles aggressively when we hit /events ~200 times in a row
# during discovery (Financials has 265 series, Politics 1800+). A blanket
# 100ms delay is not enough — observed ~40% of /events calls returning
# HTTP 429 in the middle of a discovery run. We retry each throttled
# request up to KALSHI_MAX_RETRIES times with exponential backoff.
KALSHI_MAX_RETRIES = 4
KALSHI_RETRY_BASE_DELAY = 1.0  # seconds; doubles on each retry
KALSHI_EVENTS_BASE_DELAY = 0.25  # per-series pacing between /events calls


def _kalshi_events_get_with_retry(
    session,
    url: str,
    params: dict,
    series_ticker: str,
) -> dict | None:
    """GET Kalshi /events with 429-aware exponential backoff.

    Returns the parsed JSON dict on success, or ``None`` if all retries
    are exhausted. A None return should be treated as "skip this series"
    by the caller — it is ALWAYS logged at WARNING so gaps are visible.
    """
    for attempt in range(KALSHI_MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)
        except Exception as e:
            logger.warning(
                "Kalshi /events network error for %s (attempt %d/%d): %s",
                series_ticker, attempt + 1, KALSHI_MAX_RETRIES + 1, e,
            )
            if attempt >= KALSHI_MAX_RETRIES:
                return None
            time.sleep(KALSHI_RETRY_BASE_DELAY * (2 ** attempt))
            continue

        status = getattr(r, "status_code", 200)
        if status == 429:
            if attempt >= KALSHI_MAX_RETRIES:
                logger.warning(
                    "Kalshi /events 429 for %s after %d retries — skipping",
                    series_ticker, KALSHI_MAX_RETRIES,
                )
                return None
            backoff = KALSHI_RETRY_BASE_DELAY * (2 ** attempt)
            logger.info(
                "Kalshi /events 429 for %s (attempt %d) — sleeping %.1fs",
                series_ticker, attempt + 1, backoff,
            )
            time.sleep(backoff)
            continue

        if status >= 400:
            logger.warning(
                "Kalshi /events HTTP %d for %s — skipping",
                status, series_ticker,
            )
            return None

        try:
            return r.json() or {}
        except Exception as e:
            logger.warning(
                "Kalshi /events JSON decode failed for %s: %s",
                series_ticker, e,
            )
            return None

    return None


def fetch_active_kalshi_markets(
    categories: tuple[str, ...] | list[str] = KALSHI_DISCOVERY_CATEGORIES,
    max_series_per_category: int = 500,
) -> list[dict]:
    """Fetch currently-tradeable Kalshi markets via /series -> /events.

    The flat ``/markets?status=open`` endpoint is swamped with multi-leg
    KXMVE* parlay markets (20k+ rows of noise). The /series -> /events
    chain is more targeted: it gives us actual single-question markets
    like KXWTI-*, KXFEDDECISION-*, KXPRESNOMD-* that our matcher can
    reason about.

    Returns a flat list of dicts in the same schema as /markets so the
    rest of the pipeline doesn't have to know which endpoint was used.
    """
    import requests  # lazy

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    markets: list[dict] = []
    seen_tickers: set[str] = set()

    for category in categories:
        # 1. Pull the list of series for this category.
        # Note: ``.get("series", [])`` isn't sufficient because Kalshi
        # sometimes returns ``{"series": null}`` for empty categories
        # (observed live on Climate). ``None`` then breaks slicing
        # downstream, so we explicitly coalesce to an empty list.
        try:
            r = session.get(
                f"{KALSHI_BASE_URL}/series",
                params={"category": category},
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json() or {}
            series_list = payload.get("series") or []
        except Exception as e:
            logger.warning("Kalshi /series failed for %s: %s", category, e)
            continue

        if not series_list:
            logger.info("Kalshi discovery category=%s: no series returned", category)
            continue

        series_ct = 0
        for series in series_list[:max_series_per_category]:
            series_ticker = series.get("ticker")
            if not series_ticker:
                continue

            # Per-series pacing: keeps us under Kalshi's /events rate
            # limit when iterating 200+ series in a row. The retry
            # helper handles spikes that still get throttled.
            time.sleep(KALSHI_EVENTS_BASE_DELAY)

            cursor: str | None = None
            while True:
                params: dict[str, Any] = {
                    "series_ticker": series_ticker,
                    "status": "open",
                    "with_nested_markets": True,
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor
                data = _kalshi_events_get_with_retry(
                    session,
                    f"{KALSHI_BASE_URL}/events",
                    params,
                    series_ticker,
                )
                if data is None:
                    # Retry helper already logged; skip this series.
                    break

                events = data.get("events") or []
                for event in events:
                    for m in (event.get("markets") or []):
                        ticker = m.get("ticker")
                        if not ticker or ticker in seen_tickers:
                            continue
                        seen_tickers.add(ticker)
                        # Normalize: store event_ticker for display
                        m.setdefault("event_ticker", event.get("event_ticker", ""))
                        markets.append(m)
                        series_ct += 1

                cursor = data.get("cursor") or None
                if not cursor:
                    break
                time.sleep(0.1)

        logger.info(
            "Kalshi discovery category=%s: %d markets from %d series",
            category, series_ct, min(len(series_list), max_series_per_category),
        )
        time.sleep(0.2)

    logger.info("fetch_active_kalshi_markets: %d markets total", len(markets))
    return markets


def fetch_active_poly_markets(
    max_pages: int = 300,
    page_size: int = 100,
) -> list[dict]:
    """Fetch currently-tradeable Polymarket markets via Gamma API.

    Returns a flat list of market dicts with fields:
      - conditionId (poly_id)
      - question, description
      - outcomes, outcomePrices (string-encoded JSON arrays)
      - volume, liquidity
      - endDate
      - clobTokenIds

    Note: Gamma API returns markets in approximate volume-desc order.
    Polymarket has tens of thousands of active markets; the commodity
    markets we care about (WTI "hit $X in April", Crude Oil (CL) settle
    buckets) land at offsets 5,000-28,000 — well past the old
    ``max_pages=10`` cap. Bumped to 60 (30k markets) to reach them.
    The volume filter still drops junk, so the candidate universe we
    actually hand to the matcher stays manageable.

    Page-size cap (2026-05-14 regression):
        Polymarket silently capped their /markets endpoint to a max
        100 results per page. Confirmed by direct probe: limit=10
        returns 10, limit=500 returns 100. With page_size>100 the
        pagination loop's `len(page) < page_size` termination clause
        fires after the first page, capping the entire universe at
        100 markets. We now use page_size=100 (matching the cap) and
        bumped max_pages from 60 to 300 to keep the effective depth
        at the same ~30k markets where commodity contracts live.
        Adds ~60s of per-page sleep to discovery runtime (300 pages
        x 0.2s), still well under the 30-min qsub h_rt.
    """
    import requests  # lazy

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    markets: list[dict] = []
    offset = 0

    for _ in range(max_pages):
        params = {
            "active": "true",
            "closed": "false",
            "archived": "false",
            "limit": page_size,
            "offset": offset,
        }
        try:
            r = session.get(POLY_GAMMA_MARKETS_URL, params=params, timeout=30)
            r.raise_for_status()
            page = r.json() or []
        except Exception as e:
            logger.warning("Polymarket fetch failed: %s", e)
            break
        if not page:
            break
        markets.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.2)

    logger.info("fetch_active_poly_markets: %d markets", len(markets))
    return markets


# ----------------------------------------------------------------------
# Matching (lazy-imports sentence_transformers)
# ----------------------------------------------------------------------


def _kalshi_mid(m: dict) -> float:
    """Best-effort price midpoint from a Kalshi market record.

    The live /markets endpoint returns string-encoded dollar fields
    (yes_bid_dollars, yes_ask_dollars, last_price_dollars). Older code
    paths may also provide numeric yes_bid/yes_ask.
    """
    def _to_float(v) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Newer live schema (strings in dollars)
    bid = _to_float(m.get("yes_bid_dollars"))
    ask = _to_float(m.get("yes_ask_dollars"))
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return (bid + ask) / 2.0

    # Older numeric schema
    bid = _to_float(m.get("yes_bid"))
    ask = _to_float(m.get("yes_ask"))
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        return (bid + ask) / 2.0

    # Last trade fallback
    for field in ("last_price_dollars", "last_price"):
        v = _to_float(m.get(field))
        if v is not None and v > 0:
            return v

    return 0.0


# Kalshi ticker prefixes we always skip during discovery.
# KXMVE* are multi-variable "parlay" markets with comma-separated titles
# that can't map 1:1 onto a single Polymarket question. Matching them
# just produces noise.
_KALSHI_SKIP_PREFIXES = ("KXMVE",)


def _is_matchable_kalshi_market(m: dict) -> bool:
    """Return True if a Kalshi market is worth trying to match.

    Rejects:
      - Multi-leg parlay markets (KXMVE*) with comma-separated titles
      - Markets with empty/missing titles
      - Markets with no trading activity (vol==0 and open_interest==0)
    """
    ticker = (m.get("ticker") or "").upper()
    if any(ticker.startswith(p) for p in _KALSHI_SKIP_PREFIXES):
        return False
    title = (m.get("title") or "").strip()
    if not title:
        return False
    # Comma-joined multi-leg titles ("yes X,no Y,yes Z") — skip
    if title.count(",") >= 2 and "yes" in title.lower() and "no" in title.lower():
        return False
    return True


def _poly_yes_price(m: dict) -> float:
    """Extract the YES outcome price from a Polymarket market record."""
    raw = m.get("outcomePrices")
    if not raw:
        return 0.0
    try:
        if isinstance(raw, str):
            prices = json.loads(raw)
        else:
            prices = raw
        if prices:
            return float(prices[0])
    except (json.JSONDecodeError, TypeError, ValueError, IndexError):
        pass
    return 0.0


# Minimum Polymarket volume to consider a market. Gamma API returns tens
# of thousands of markets, most with near-zero volume (speculative/joke
# markets). We only care about markets we could actually trade on, so
# drop anything under $5k cumulative volume.
MIN_POLY_VOLUME = 5000.0


def _poly_volume(m: dict) -> float:
    """Extract volume from a Polymarket market record.

    Gamma uses a few different volume fields depending on the market
    type. Take the max of whatever's available.
    """
    candidates = (
        m.get("volume"),
        m.get("volumeNum"),
        m.get("volume24hr"),
        m.get("liquidity"),
    )
    best = 0.0
    for c in candidates:
        if c is None:
            continue
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        if v > best:
            best = v
    return best


def _extract_outcome_tokens(m: dict) -> tuple[str, str]:
    """Return (yes_token_id, no_token_id) decimal strings from a Gamma record.

    Polymarket's CLOB requires per-outcome ERC-1155 token IDs (not just
    the market-level conditionId) for order placement. They live in
    ``clobTokenIds`` as a JSON-encoded length-2 array, with index order
    matching the ``outcomes`` array (typically ["Yes","No"], but we
    don't trust that — we read ``outcomes`` and pick the actual YES side).

    Returns ("", "") if the market isn't binary or the field is malformed.
    """
    tokens_raw = m.get("clobTokenIds")
    outcomes_raw = m.get("outcomes")
    if not tokens_raw or not outcomes_raw:
        return ("", "")
    try:
        tokens = (json.loads(tokens_raw)
                  if isinstance(tokens_raw, str) else tokens_raw)
        outcomes = (json.loads(outcomes_raw)
                    if isinstance(outcomes_raw, str) else outcomes_raw)
    except (json.JSONDecodeError, TypeError):
        return ("", "")
    if not isinstance(tokens, list) or not isinstance(outcomes, list):
        return ("", "")
    if len(tokens) != 2 or len(outcomes) != 2:
        return ("", "")
    yes_idx = next(
        (i for i, o in enumerate(outcomes)
         if str(o).strip().lower() in ("yes", "true", "y")),
        0,
    )
    no_idx = 1 - yes_idx
    return (str(tokens[yes_idx]), str(tokens[no_idx]))


def _filter_poly_by_volume(
    poly_markets: list[dict], min_volume: float = MIN_POLY_VOLUME
) -> list[dict]:
    """Drop Polymarket markets below a volume floor.

    Dramatically reduces the candidate space without losing anything we
    could realistically trade on.
    """
    kept = [m for m in poly_markets if _poly_volume(m) >= min_volume]
    logger.info(
        "Polymarket volume filter: %d -> %d (min_volume=%s)",
        len(poly_markets), len(kept), min_volume,
    )
    return kept


def match_markets(
    kalshi_markets: list[dict],
    poly_markets: list[dict],
    similarity_threshold: float = 0.70,
    top_k: int = 3,
    min_poly_volume: float = MIN_POLY_VOLUME,
) -> list[dict]:
    """Produce high-similarity active_matches-schema records.

    Uses an encode-once-then-matmul strategy:
      1. Volume-filter Polymarket so we don't waste compute on junk.
      2. Encode every Kalshi and Polymarket text exactly once.
      3. Compute a full similarity matrix via vectorized numpy
         (normalized embeddings + dot product = cosine).
      4. For each Kalshi market, pick the single best Polymarket match
         above the similarity threshold.
      5. Apply the structural quality filter on the resulting pairs.

    This replaces a naive keyword-overlap candidate generator that
    produced ~4M candidate pairs on a real run and would have taken
    ~6 hours to score. The matrix approach runs in ~60-90s for the
    same inputs and scales cleanly.

    Args:
        kalshi_markets: Raw Kalshi market dicts from /series->/events.
        poly_markets: Raw Polymarket market dicts from Gamma API.
        similarity_threshold: Minimum cosine similarity to accept.
        top_k: Unused today (kept 1-best per Kalshi). Reserved for
            future multi-candidate workflows.
        min_poly_volume: Minimum Polymarket cumulative volume to
            consider. Default drops tens of thousands of junk markets.

    Returns:
        List of active_matches-schema dicts surviving the quality filter.
    """
    import numpy as np  # lazy

    from src.matching.quality_filter import filter_active_match
    from src.matching.semantic_matcher import SemanticMatcher  # lazy

    # ------------------------------------------------------------------
    # 1. Input filtering
    # ------------------------------------------------------------------
    kalshi_markets = [m for m in kalshi_markets if _is_matchable_kalshi_market(m)]
    poly_markets = _filter_poly_by_volume(poly_markets, min_volume=min_poly_volume)

    if not kalshi_markets or not poly_markets:
        logger.info(
            "match_markets: empty input after filtering (kalshi=%d poly=%d)",
            len(kalshi_markets), len(poly_markets),
        )
        return []

    # ------------------------------------------------------------------
    # 2. Encode once (this is the expensive step)
    # ------------------------------------------------------------------
    matcher = SemanticMatcher()
    k_texts = [m.get("title", "") for m in kalshi_markets]
    p_texts = [m.get("question", "") for m in poly_markets]

    logger.info(
        "Encoding embeddings: %d kalshi + %d poly = %d texts",
        len(k_texts), len(p_texts), len(k_texts) + len(p_texts),
    )
    k_emb = matcher.encode_batch(k_texts).astype(np.float32)  # (N_k, d)
    p_emb = matcher.encode_batch(p_texts).astype(np.float32)  # (N_p, d)

    # ------------------------------------------------------------------
    # 3. Vectorized cosine similarity via normalized dot product
    # ------------------------------------------------------------------
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # avoid /0
        return x / norms

    k_emb = _normalize(k_emb)
    p_emb = _normalize(p_emb)

    # Similarity matrix: (N_k, N_p). 1065 x 4000 ~ 17MB, fits in RAM easy.
    sim_matrix = k_emb @ p_emb.T

    # ------------------------------------------------------------------
    # 4. Best Polymarket per Kalshi above threshold
    # ------------------------------------------------------------------
    best_idx = np.argmax(sim_matrix, axis=1)  # (N_k,) index into poly
    best_sim = sim_matrix[np.arange(len(kalshi_markets)), best_idx]  # (N_k,)

    best: dict[str, tuple[float, dict]] = {}
    for i, km in enumerate(kalshi_markets):
        sim_val = float(best_sim[i])
        if sim_val < similarity_threshold:
            continue
        kt = km.get("ticker", "")
        if not kt:
            continue
        pm = poly_markets[int(best_idx[i])]
        k_mid = _kalshi_mid(km)
        p_price = _poly_yes_price(pm)
        # Capture YES/NO outcome ERC-1155 token IDs from clobTokenIds.
        # Required for placing real orders on Polymarket CLOB. Format is
        # a JSON-string array [yes_token, no_token] of decimal strings.
        # Index ordering is determined by `outcomes`: ["Yes","No"].
        yes_token, no_token = _extract_outcome_tokens(pm)
        match = {
            "kalshi_ticker": kt,
            "kalshi_title": km.get("title", ""),
            "kalshi_event": km.get("event_ticker", ""),
            "kalshi_mid": k_mid,
            "kalshi_vol": km.get("volume", 0),
            "poly_id": pm.get("conditionId", ""),
            "poly_title": pm.get("question", ""),
            "poly_price": p_price,
            "poly_vol": _poly_volume(pm),
            "polymarket_yes_token_id": yes_token,
            "polymarket_no_token_id": no_token,
            "similarity": sim_val,
            "spread": round(k_mid - p_price, 6),
        }
        # If the same Kalshi ticker somehow appears twice, keep highest-sim
        if kt not in best or sim_val > best[kt][0]:
            best[kt] = (sim_val, match)

    # ------------------------------------------------------------------
    # 5. Structural quality filter
    # ------------------------------------------------------------------
    survivors: list[dict] = []
    rejected_reasons: dict[str, int] = {}
    for _, match in best.values():
        ok, reason = filter_active_match(match)
        if ok:
            survivors.append(match)
        else:
            key = (reason or "unknown").split(" ")[0]
            rejected_reasons[key] = rejected_reasons.get(key, 0) + 1

    logger.info(
        "match_markets: %d kalshi x %d poly -> %d semantic matches -> %d "
        "after quality filter (rejections=%s)",
        len(kalshi_markets), len(poly_markets), len(best), len(survivors),
        rejected_reasons,
    )
    return survivors


# ----------------------------------------------------------------------
# End-to-end discovery (wired by scripts/discover_markets.py)
# ----------------------------------------------------------------------


def _load_protected_indices(live_dir: Path) -> set[int]:
    """Return the set of array indices that must not be evicted.

    Reads open positions from ``positions.db`` and converts each
    ``pair_id`` (``live_NNNN``) into its integer index. Missing DB or
    missing table is not an error — discovery just has nothing to
    protect.
    """
    import sqlite3

    db_path = live_dir / "positions.db"
    if not db_path.exists():
        return set()
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            # The open-positions table is "positions" (closed rows live
            # in "closed_positions"). Only live_NNNN pair_ids map to
            # array indices; anything else (offline training pair_ids)
            # is silently skipped.
            cur.execute("SELECT pair_id FROM positions")
            rows = cur.fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("Could not read positions.db for protection set: %s", e)
        return set()

    protected: set[int] = set()
    for (pid,) in rows:
        if not isinstance(pid, str) or not pid.startswith("live_"):
            continue
        try:
            protected.add(int(pid.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if protected:
        logger.info(
            "Loaded %d protected pair_ids (open positions) for eviction guard",
            len(protected),
        )
    return protected


def run_discovery(
    live_dir: Path = Path("data/live"),
    similarity_threshold: float = 0.70,
) -> dict:
    """End-to-end discovery: fetch → match → filter → upsert.

    Returns a stats dict. Safe to call from cron / GitHub Actions.
    """
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    matches_path = live_dir / "active_matches.json"

    # 1. Fetch live markets
    kalshi = fetch_active_kalshi_markets()
    poly = fetch_active_poly_markets()

    # 2. Match + filter
    new_matches = match_markets(kalshi, poly, similarity_threshold=similarity_threshold)

    # 3. Load existing
    if matches_path.exists():
        with open(matches_path) as f:
            existing = json.load(f)
    else:
        existing = []

    # 4. Upsert (protecting indices referenced by open positions so
    #    eviction can't pull the rug out from under live trading).
    protected_indices = _load_protected_indices(live_dir)
    merged, stats = upsert_active_matches(
        existing,
        new_matches,
        protected_indices=protected_indices,
    )

    # 5. Save
    with open(matches_path, "w") as f:
        json.dump(merged, f, indent=2)

    stats["kalshi_fetched"] = len(kalshi)
    stats["poly_fetched"] = len(poly)
    stats["new_matches_this_run"] = len(new_matches)
    stats["protected_indices_count"] = len(protected_indices)
    return stats
