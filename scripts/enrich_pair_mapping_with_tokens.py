#!/usr/bin/env python3
"""Enrich pair_mapping.json with Polymarket YES/NO outcome token IDs.

Background: pair_mapping.json was generated before live trading was a
goal, so it stored only the market-level conditionId (`polymarket_market_id`).
But order placement on Polymarket CLOB requires the per-outcome
ERC-1155 token IDs (one for the YES outcome, one for NO). This script
fetches those from the Gamma API and persists them.

Output schema additions per pair_id:
    polymarket_yes_token_id   — decimal string (YES outcome ERC-1155 id)
    polymarket_no_token_id    — decimal string (NO outcome ERC-1155 id)
    polymarket_tokens_fetched_at — unix timestamp

Pairs whose conditionId returns no Gamma result are left untouched
(strategy.py safely skips them rather than placing one-sided orders).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
BATCH_SIZE = 100        # Gamma supports multi-id lookups via `condition_ids=`
REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_BATCHES = 0.25  # avoid 429s

MAPPING_PATH = Path("data/live/pair_mapping.json")


def fetch_tokens_for_condition_ids(condition_ids: list[str]) -> dict[str, dict]:
    """Return condition_id -> {yes_token, no_token} for the IDs Gamma knows.

    Batches into chunks of BATCH_SIZE. Missing condition_ids simply
    aren't in the result (caller decides what to do).
    """
    out: dict[str, dict] = {}
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    for i in range(0, len(condition_ids), BATCH_SIZE):
        chunk = condition_ids[i : i + BATCH_SIZE]
        # Gamma accepts repeated ?condition_ids= params
        params = [("condition_ids", cid) for cid in chunk] + [("limit", BATCH_SIZE)]
        try:
            resp = session.get(GAMMA_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [batch {i // BATCH_SIZE}] error: {e}", file=sys.stderr)
            time.sleep(1.0)
            continue
        if not isinstance(data, list):
            continue
        for market in data:
            cid = market.get("conditionId")
            if not cid:
                continue
            tokens_raw = market.get("clobTokenIds")
            outcomes_raw = market.get("outcomes")
            if not tokens_raw or not outcomes_raw:
                continue
            try:
                tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else tokens_raw
                outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
            except json.JSONDecodeError:
                continue
            if len(tokens) != 2 or len(outcomes) != 2:
                # Not a binary market — Polymarket also has multi-outcome
                # markets we shouldn't trade through this codepath.
                continue
            # Find which index corresponds to YES vs NO; Polymarket
            # convention is index 0 = YES, index 1 = NO, but verify.
            yes_idx = next(
                (k for k, o in enumerate(outcomes) if str(o).strip().lower() in ("yes", "true", "y")),
                0,
            )
            no_idx = 1 - yes_idx
            out[cid] = {
                "polymarket_yes_token_id": str(tokens[yes_idx]),
                "polymarket_no_token_id": str(tokens[no_idx]),
            }
        time.sleep(SLEEP_BETWEEN_BATCHES)
    return out


def main() -> int:
    if not MAPPING_PATH.exists():
        print(f"Missing {MAPPING_PATH}", file=sys.stderr)
        return 1

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    print(f"Loaded {len(mapping):,} pair entries from {MAPPING_PATH}")

    # Collect distinct conditionIds.
    cid_to_pairs: dict[str, list[str]] = defaultdict(list)
    for pair_id, entry in mapping.items():
        cid = entry.get("polymarket_market_id", "")
        if cid and isinstance(cid, str):
            cid_to_pairs[cid].append(pair_id)

    print(f"Distinct conditionIds: {len(cid_to_pairs):,}")

    # Skip pairs we've already enriched.
    needs_fetch = [cid for cid, pids in cid_to_pairs.items()
                   if not all(mapping[p].get("polymarket_yes_token_id") for p in pids)]
    print(f"Need to fetch: {len(needs_fetch):,} (skipping already-enriched)")
    if not needs_fetch:
        print("Nothing to do.")
        return 0

    print("Fetching from Gamma API in batches of "
          f"{BATCH_SIZE} ({len(needs_fetch) // BATCH_SIZE + 1} batches)...")
    tokens_by_cid = fetch_tokens_for_condition_ids(needs_fetch)
    print(f"Gamma returned tokens for {len(tokens_by_cid):,} / {len(needs_fetch):,} conditionIds")

    enriched_count = 0
    now_ts = int(time.time())
    for cid, pair_ids in cid_to_pairs.items():
        tokens = tokens_by_cid.get(cid)
        if not tokens:
            continue
        for pid in pair_ids:
            mapping[pid].update(tokens)
            mapping[pid]["polymarket_tokens_fetched_at"] = now_ts
            enriched_count += 1

    print(f"Enriched {enriched_count:,} pair entries")

    tmp_path = MAPPING_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(mapping, f)
    tmp_path.replace(MAPPING_PATH)
    print(f"Wrote {MAPPING_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
