"""Fetch settlement criteria from platform APIs for candidate pairs."""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import logging

from src.data.client import ResilientClient

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA_BASE = "https://gamma-api.polymarket.com"


def _fetch_kalshi_settlement(client: ResilientClient, market_id: str) -> str:
    """Fetch rules_primary + rules_secondary from Kalshi API."""
    try:
        resp = client.session.get(
            f"{client.base_url}/markets/{market_id}",
            timeout=client.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        market = data.get("market", {})
        primary = market.get("rules_primary", "")
        secondary = market.get("rules_secondary", "")
        parts = [p for p in [primary, secondary] if p]
        return " | ".join(parts) if parts else ""
    except Exception as e:
        logger.warning(f"Failed to fetch Kalshi settlement for {market_id}: {e}")
        return "FETCH_FAILED"


def _fetch_polymarket_settlement(client: ResilientClient, market_id: str) -> str:
    """Fetch description + resolutionSource from Polymarket Gamma API."""
    try:
        resp = client.session.get(
            f"{client.base_url}/markets",
            params={"id": market_id},
            timeout=client.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return ""
        market = data[0] if isinstance(data, list) else data
        desc = market.get("description", "")
        source = market.get("resolutionSource", "")
        parts = [p for p in [desc, source] if p]
        return " | ".join(parts) if parts else ""
    except Exception as e:
        logger.warning(f"Failed to fetch Polymarket settlement for {market_id}: {e}")
        return "FETCH_FAILED"


def enrich_settlement_criteria(candidates: list[dict]) -> list[dict]:
    """Add settlement criteria fields to each candidate dict.

    Fetches from Kalshi and Polymarket APIs on-demand (only for candidates,
    not all markets). Adds: kalshi_settlement, polymarket_settlement,
    settlement_aligned, settlement_notes.
    """
    kalshi_client = ResilientClient(base_url=KALSHI_BASE, requests_per_second=18.0)
    gamma_client = ResilientClient(base_url=GAMMA_BASE, requests_per_second=10.0)

    for candidate in candidates:
        k_settlement = _fetch_kalshi_settlement(
            kalshi_client, candidate["kalshi_market_id"]
        )
        p_settlement = _fetch_polymarket_settlement(
            gamma_client, candidate["polymarket_market_id"]
        )

        candidate["kalshi_settlement"] = k_settlement
        candidate["polymarket_settlement"] = p_settlement

        # Determine alignment -- if either fetch failed, mark as not aligned
        if k_settlement == "FETCH_FAILED" or p_settlement == "FETCH_FAILED":
            candidate["settlement_aligned"] = False
            candidate["settlement_notes"] = (
                "Settlement criteria could not be fetched from one or both platforms"
            )
        elif not k_settlement or not p_settlement:
            candidate["settlement_aligned"] = False
            candidate["settlement_notes"] = (
                "Settlement criteria missing from one or both platforms"
            )
        else:
            # Default to True -- human reviewer will verify during curation
            candidate["settlement_aligned"] = True
            candidate["settlement_notes"] = ""

    return candidates
