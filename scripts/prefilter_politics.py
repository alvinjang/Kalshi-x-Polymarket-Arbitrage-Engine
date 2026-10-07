"""Pre-filter politics metadata for matching.

Filters political markets on both platforms to reduce pairwise matching load.
"""
import json
import re
from pathlib import Path


def filter_kalshi_politics(metadata: list[dict]) -> list[dict]:
    """Keep unique political events, deduplicating by event prefix."""
    events = {}
    for m in metadata:
        mid = m.get("market_id", "")
        # Extract event prefix (strip strike suffix like -T4.3 or -YES)
        prefix = re.sub(r"-[TY][\d.A-Z]+$", "", mid)
        prefix = re.sub(r"-[A-Z]+\d+$", "", prefix)
        if prefix not in events:
            events[prefix] = m
    return list(events.values())


def filter_polymarket_politics(metadata: list[dict]) -> list[dict]:
    """Filter Polymarket for genuine political markets."""
    political_keywords = [
        "trump", "biden", "harris", "election", "senate", "house",
        "president", "presidential", "governor", "democrat", "republican",
        "congress", "nomination", "primary", "gop", "dnc",
    ]

    filtered = []
    seen = set()
    for m in metadata:
        q = m.get("question", "").lower()
        mid = m.get("market_id", "")

        if mid in seen:
            continue

        # Must have political keyword
        if not any(k in q for k in political_keywords):
            continue

        seen.add(mid)
        filtered.append(m)

    return filtered


def main():
    # Load
    kalshi = json.load(open("data/raw/kalshi/_politics_metadata.json"))
    poly = json.load(open("data/raw/polymarket/_metadata.json"))

    print(f"Loaded: Kalshi {len(kalshi)}, Polymarket {len(poly)}")

    # Filter
    kalshi_filtered = filter_kalshi_politics(kalshi)
    poly_filtered = filter_polymarket_politics(poly)

    print(f"Filtered: Kalshi {len(kalshi_filtered)}, Polymarket {len(poly_filtered)}")

    # Add category field for matching (the existing pipeline expects 'Economics' etc)
    for m in kalshi_filtered:
        m["category"] = "Politics"  # normalize for matching
    for m in poly_filtered:
        m["category"] = "politics"  # normalize

    # Save
    Path("data/raw/kalshi/_politics_metadata_filtered.json").write_text(
        json.dumps(kalshi_filtered, indent=2)
    )
    Path("data/raw/polymarket/_politics_metadata_filtered.json").write_text(
        json.dumps(poly_filtered, indent=2)
    )
    print("Saved filtered metadata files")
    print(f"Candidate comparisons: ~{len(kalshi_filtered) * len(poly_filtered):,}")


if __name__ == "__main__":
    main()
