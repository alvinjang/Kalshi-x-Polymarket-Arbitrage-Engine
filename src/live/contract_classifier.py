"""Contract classifier -- parse Kalshi tickers, extract resolution dates, classify into tiers.

Parses 8+ Kalshi ticker formats to extract resolution dates, falls back to
Kalshi API for unparseable tickers, and classifies each contract into a
monitoring tier (DAILY/WEEKLY/MONTHLY/QUARTERLY) based on time-to-resolution.

Tier determines bar collection interval:
  DAILY    (0-7 days)   -> 15 min bars (900s)
  WEEKLY   (7-30 days)  -> 1h bars (3600s)
  MONTHLY  (30-90 days) -> 4h bars (14400s)
  QUARTERLY (90+ days)  -> daily bars (86400s)
  UNKNOWN              -> 4h bars (14400s, conservative default)

Usage:
    python -m src.live.contract_classifier               # classify all 615 pairs
    python -m src.live.contract_classifier --no-api       # regex-only, skip API
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.

import argparse
import calendar
import json
import logging
import re
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

_BAR_INTERVALS = {
    "DAILY": 900,       # 15 min
    "WEEKLY": 3600,     # 1 hour
    "MONTHLY": 14400,   # 4 hours
    "QUARTERLY": 86400, # 1 day
    "UNKNOWN": 14400,   # default to 4h
}


class Tier(Enum):
    """Contract monitoring tier based on time to resolution."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    UNKNOWN = "UNKNOWN"

    @property
    def bar_interval_seconds(self) -> int:
        return _BAR_INTERVALS[self.value]


BAR_INTERVALS: dict[Tier, int] = {t: t.bar_interval_seconds for t in Tier}

# ---------------------------------------------------------------------------
# Month mapping
# ---------------------------------------------------------------------------

MONTH_MAP: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    # Extended variants found in data
    "JUNE": 6,
}

# ---------------------------------------------------------------------------
# Sports defaults: prefix -> (month, day)
# ---------------------------------------------------------------------------

SPORTS_DEFAULTS: dict[str, tuple[int, int]] = {
    # NBA
    "KXNBA": (6, 30),
    "KXNBAEAST": (6, 30),
    "KXNBAWEST": (6, 30),
    "KXNBAWINS": (6, 30),
    "KXNBATOPPICK": (6, 30),
    "KXTEAMSINNBAF": (6, 30),
    "KXTEAMSINNBAEF": (6, 30),
    # MLB
    "KXMLB": (10, 31),
    "KXMLBNL": (10, 31),
    # F1
    "KXF1": (12, 31),
    "KXF1CONSTRUCTORS": (12, 31),
    # World Cup
    "KXMENWORLDCUP": (7, 19),
    "KXWCGROUPWIN": (7, 19),
    "KXWCROUND": (7, 19),
    "KXWCCONGO": (7, 19),
    # UCL
    "KXUCL": (6, 1),
    "KXUCLGAME": (6, 1),
    "KXTEAMSINUCL": (6, 1),
    # UEL
    "KXUEL": (5, 31),
    # La Liga
    "KXLALIGA": (5, 31),
    "KXLALIGAGAME": (5, 31),
    "KXLALIGATOP4": (5, 31),
    "KXLALIGARELEGATION": (5, 31),
    # Liga MX
    "KXLIGAMX": (5, 31),
    # Eurovision
    "KXEUROVISION": (5, 31),
    "KXEUROVISIONPARTICIPANTS": (5, 31),
    "KXEUROVISIONTELEV": (5, 31),
    # Super Bowl
    "KXSB": (2, 15),
    # Copa del Rey
    "KXCOPADELREY": (4, 30),
}

# ---------------------------------------------------------------------------
# Politics defaults: prefix -> (month, day) or None (force API)
# ---------------------------------------------------------------------------

POLITICS_DEFAULTS: dict[str, tuple[int, int] | None] = {
    # Presidential -- person level
    "KXPRESPERSON": (11, 5),
    "KXPRESPARTY": (11, 5),
    "KXPRESELECTIONOCCUR": (11, 5),
    "KXPRESENDORSEMUSKD": (11, 5),
    # Republican nominee -> convention
    "KXPRESNOMR": (8, 31),
    "KXVPRESNOMR": (8, 31),
    # VP Dem nominee -> convention
    "KXVPRESNOMD": (8, 31),
    # Dem nominee -> force API (too many variants)
    "KXPRESNOMD": None,
    # Senate
    "KXAOCSENATE": (11, 5),
    "KXSENATENYD": (11, 5),
    "KXSENATEPAD": (11, 5),
    "KXSENATELAD": (11, 5),
    "SENATEWA": (11, 5),
    # 2028 Dem run
    "KX2028DRUN": (11, 5),
    # Primaries
    "KXMUSKPRIMARY": (11, 5),
    "KXFLPRIMARY": (11, 5),
    "KXGAPRIMARY": (11, 5),
    "KXNCPRIMARY": (11, 5),
    "KXORPRIMARY": (11, 5),
    # Other elections
    "KXBRPRES": (10, 31),
    "KXBRBALLOT": (10, 31),
    "KXFRENCHPRES": (5, 31),
    "KXHUNGARYPARLI": (4, 30),
    "KXCOLOMBIAPRES": (5, 31),
    "KXCOLOMBIACHAMBER": (5, 31),
    "KXCOLOMBIASENATE": (5, 31),
    "KXSECSTATEVISIT": (12, 31),
    "KXFEDGOVNOM": (12, 31),
    "KXLARGECUT": (12, 31),
    # Congressional district primaries (KX + state + district + party)
    "KXNY7D": (11, 5),
    "KXME02D": (11, 5),
}


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Full YYMMMDD: "26APR08" or "26MAR31" etc. embedded anywhere in ticker
# Captures: year (2 digits), month (3-letter), day (1-2 digits, possibly
# followed by extra chars that we ignore)
_RE_YYMMMDD = re.compile(
    r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{1,2})",
    re.IGNORECASE,
)

# Split format: -YY-MMMDD at end or before another dash
_RE_SPLIT_YYMMMDD = re.compile(
    r"-(\d{2})-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(?:$|-)",
    re.IGNORECASE,
)

# YYMM only: "26APR" followed by dash or end
_RE_YYMM = re.compile(
    r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(?=-|$)",
    re.IGNORECASE,
)

# Month-only at end after -YY-: KXBTCMAX100-26-APR or KXBTCMAX100-26-JUNE
_RE_YEAR_MONTH_NAME = re.compile(
    r"-(\d{2})-(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUNE?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)$",
    re.IGNORECASE,
)

# Year-only: 2-digit year after dash, followed by dash, end, or non-digit
# e.g., -26- or -26B- or -26FINAL- or -26CLA-
_RE_YEAR_ONLY = re.compile(r"-(\d{2})(?=[^0-9]|$)")

# 4-digit year variant: -2028-
_RE_YEAR_4DIGIT = re.compile(r"-(\d{4})-")

# Primary-style tickers: KXFLPRIMARY-23D26-JMOS (year embedded after D)
_RE_PRIMARY_YEAR = re.compile(r"PRIMARY-\d+D(\d{2})-", re.IGNORECASE)


def _last_day_of_month(year: int, month: int) -> int:
    """Return the last day of the given month."""
    return calendar.monthrange(year, month)[1]


def _month_name_to_num(name: str) -> int | None:
    """Convert month name (3+ letters) to number."""
    upper = name.upper()
    # Direct lookup
    if upper in MONTH_MAP:
        return MONTH_MAP[upper]
    # Try 3-letter prefix
    prefix = upper[:3]
    return MONTH_MAP.get(prefix)


# ---------------------------------------------------------------------------
# ContractClassifier
# ---------------------------------------------------------------------------


class ContractClassifier:
    """Classifies Kalshi contracts by resolution date and monitoring tier.

    Parses resolution dates from ticker strings using regex patterns,
    falls back to Kalshi API for unparseable tickers, and classifies
    each contract into a tier based on hours remaining.
    """

    KALSHI_MARKET_URL = (
        "https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}"
    )

    def __init__(self, cache_path: Path | None = None):
        self._api_cache: dict[str, str | None] = {}  # ticker -> iso_str or None
        self._cache_path = cache_path or Path("data/live/resolution_dates_cache.json")
        self._load_api_cache()

    def _load_api_cache(self) -> None:
        """Load cached API resolution dates from disk."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path) as f:
                    self._api_cache = json.load(f)
                logger.info(f"Loaded {len(self._api_cache)} cached resolution dates")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load API cache: {e}")
                self._api_cache = {}

    def _save_api_cache(self) -> None:
        """Persist API cache to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._api_cache, f, indent=2)
        logger.info(f"Saved {len(self._api_cache)} cached resolution dates")

    # ------------------------------------------------------------------
    # Ticker prefix extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_prefix(ticker: str) -> str:
        """Extract the alphabetic prefix from a ticker.

        Examples:
            KXWTI-26APR08-T105.99 -> KXWTI
            KXPRESNOMD-28-AOC -> KXPRESNOMD
            KX2028DRUN-28-CBOOK -> KX2028DRUN
            SENATEWA-28-D -> SENATEWA
            KXNBAWINS-SAS-25-T20 -> KXNBAWINS
        """
        # Split on first dash
        parts = ticker.split("-")
        prefix = parts[0]
        # Remove trailing digits from prefix (e.g., KXBTCMAX100 -> KXBTCMAX100,
        # but KXAMERICAPARTY2028 -> KXAMERICAPARTY2028 -- keep as is since
        # these are unique identifiers)
        return prefix

    # ------------------------------------------------------------------
    # Resolution date parsing
    # ------------------------------------------------------------------

    def parse_resolution_date(self, ticker: str) -> datetime | None:
        """Parse resolution date from a Kalshi ticker string.

        Tries multiple regex patterns in priority order:
        1. Full YYMMMDD embedded anywhere (26APR08, 26MAR31, etc.)
        2. Split YY-MMMDD (26-MAY15)
        3. Year-month name only (26-APR, 26-JUNE)
        4. YYMM only (26APR -> end of month)
        5. Year-only with sports/politics defaults
        6. Return None if nothing matches

        Args:
            ticker: Kalshi ticker string.

        Returns:
            Resolution datetime, or None if unparseable.
        """
        upper = ticker.upper()

        # Step 1: Try split format first (more specific): -YY-MMMDD
        m = _RE_SPLIT_YYMMMDD.search(upper)
        if m:
            year = 2000 + int(m.group(1))
            month_num = _month_name_to_num(m.group(2))
            day = int(m.group(3))
            if month_num and 1 <= day <= 31:
                try:
                    return datetime(year, month_num, day, 23, 59)
                except ValueError:
                    pass

        # Step 2: Try full YYMMMDD embedded anywhere
        m = _RE_YYMMMDD.search(upper)
        if m:
            year = 2000 + int(m.group(1))
            month_num = _month_name_to_num(m.group(2))
            day = int(m.group(3))
            if month_num:
                # Clamp day to valid range for the month
                max_day = _last_day_of_month(year, month_num)
                day = min(day, max_day)
                if day >= 1:
                    try:
                        return datetime(year, month_num, day, 23, 59)
                    except ValueError:
                        pass

        # Step 3: Try year + month name at end: KXBTCMAX100-26-APR
        m = _RE_YEAR_MONTH_NAME.search(upper)
        if m:
            year = 2000 + int(m.group(1))
            month_num = _month_name_to_num(m.group(2))
            if month_num:
                day = _last_day_of_month(year, month_num)
                return datetime(year, month_num, day, 23, 59)

        # Step 4: Try YYMM only: 26APR -> end of month
        m = _RE_YYMM.search(upper)
        if m:
            year = 2000 + int(m.group(1))
            month_num = _month_name_to_num(m.group(2))
            if month_num:
                day = _last_day_of_month(year, month_num)
                return datetime(year, month_num, day, 23, 59)

        # Step 5: Year-only with defaults
        return self._parse_year_only(ticker, upper)

    def _parse_year_only(self, ticker: str, upper: str) -> datetime | None:
        """Handle year-only tickers using sports/politics defaults.

        Args:
            ticker: Original ticker string.
            upper: Uppercased ticker.

        Returns:
            datetime from defaults, or None if no match.
        """
        # Extract year from ticker
        year = None

        # Try 4-digit year first (KXPRESPARTY-2028-D)
        m = _RE_YEAR_4DIGIT.search(upper)
        if m:
            year = int(m.group(1))

        # Try primary-style year (KXFLPRIMARY-23D26-JMOS)
        if year is None:
            m = _RE_PRIMARY_YEAR.search(upper)
            if m:
                year = 2000 + int(m.group(1))

        # Try 2-digit year
        if year is None:
            m = _RE_YEAR_ONLY.search(upper)
            if m:
                year = 2000 + int(m.group(1))

        if year is None:
            # No year found at all
            return None

        # Get prefix for defaults lookup
        prefix = self._get_prefix(upper)

        # Check sports defaults (try longest prefix match first)
        for sport_prefix, (month, day) in sorted(
            SPORTS_DEFAULTS.items(), key=lambda x: -len(x[0])
        ):
            if prefix.startswith(sport_prefix) or upper.startswith(sport_prefix):
                return datetime(year, month, day, 23, 59)

        # Check politics defaults
        for pol_prefix, default in sorted(
            POLITICS_DEFAULTS.items(), key=lambda x: -len(x[0])
        ):
            if prefix.startswith(pol_prefix) or upper.startswith(pol_prefix):
                if default is None:
                    return None  # Force API lookup
                month, day = default
                return datetime(year, month, day, 23, 59)

        # No default found
        return None

    # ------------------------------------------------------------------
    # API fallback
    # ------------------------------------------------------------------

    def fetch_resolution_from_api(self, ticker: str) -> datetime | None:
        """Fetch resolution date from Kalshi API for unparseable tickers.

        Results are cached in self._api_cache to avoid redundant calls.

        Args:
            ticker: Kalshi ticker string.

        Returns:
            Resolution datetime, or None on error/404.
        """
        # Check cache first
        if ticker in self._api_cache:
            cached = self._api_cache[ticker]
            if cached is None:
                return None
            try:
                return datetime.fromisoformat(cached.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
            except (ValueError, AttributeError):
                return None

        # Hit API
        url = self.KALSHI_MARKET_URL.format(ticker=ticker)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 404:
                self._api_cache[ticker] = None
                return None
            r.raise_for_status()

            data = r.json()
            market = data.get("market", {})
            close_time = market.get("close_time")
            if close_time:
                self._api_cache[ticker] = close_time
                dt = datetime.fromisoformat(close_time.replace("Z", "+00:00")).replace(
                    tzinfo=None
                )
                return dt
            else:
                self._api_cache[ticker] = None
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"API error for {ticker}: {e}")
            self._api_cache[ticker] = None
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"API parse error for {ticker}: {e}")
            self._api_cache[ticker] = None
            return None
        finally:
            time.sleep(0.06)  # Rate limit: ~16 req/s

    # ------------------------------------------------------------------
    # Tier classification
    # ------------------------------------------------------------------

    def classify_contract(
        self, resolution_date: datetime | None, now: datetime
    ) -> Tier:
        """Classify a contract into a monitoring tier.

        Args:
            resolution_date: When the contract resolves, or None.
            now: Current time for calculating days remaining.

        Returns:
            Tier enum value.
        """
        if resolution_date is None:
            return Tier.UNKNOWN

        days_remaining = (resolution_date - now).total_seconds() / 86400

        if days_remaining < 7:
            return Tier.DAILY
        elif days_remaining < 30:
            return Tier.WEEKLY
        elif days_remaining < 90:
            return Tier.MONTHLY
        else:
            return Tier.QUARTERLY

    # ------------------------------------------------------------------
    # Bulk classification
    # ------------------------------------------------------------------

    def classify_all_pairs(
        self,
        matches: list[dict],
        now: datetime | None = None,
        use_api: bool = True,
    ) -> dict[str, dict]:
        """Classify all matched pairs into monitoring tiers.

        For each match:
        1. Parse resolution date from kalshi_ticker
        2. If None and use_api: fetch from Kalshi API
        3. Classify into tier
        4. Store result

        Args:
            matches: List of match dicts with 'kalshi_ticker' key.
            now: Current time (defaults to datetime.utcnow()).
            use_api: Whether to call Kalshi API for unparseable tickers.

        Returns:
            Dict mapping pair_id (live_NNNN) to classification dict.
        """
        if now is None:
            now = datetime.now()

        results: dict[str, dict] = {}
        api_lookups = 0
        api_hits = 0

        from src.live.pair_ids import make_pair_id

        for match in matches:
            ticker = match.get("kalshi_ticker", "")
            pair_id = make_pair_id(ticker, match.get("poly_id", ""))
            if not pair_id:
                # Incomplete match record — skip classification.
                continue

            # Step 1: Parse from ticker
            res_date = self.parse_resolution_date(ticker)

            # Step 2: API fallback
            if res_date is None and use_api:
                api_lookups += 1
                res_date = self.fetch_resolution_from_api(ticker)
                if res_date is not None:
                    api_hits += 1

            # Step 3: Classify
            tier = self.classify_contract(res_date, now)

            # Step 4: Store
            days_remaining = None
            if res_date is not None:
                days_remaining = round(
                    (res_date - now).total_seconds() / 86400, 2
                )

            results[pair_id] = {
                "tier": tier.name,
                "resolution_date": res_date.isoformat() if res_date else None,
                "days_remaining": days_remaining,
                "bar_interval_seconds": tier.bar_interval_seconds,
                "kalshi_ticker": ticker,
            }

        if use_api and api_lookups > 0:
            logger.info(
                f"API lookups: {api_lookups} attempted, {api_hits} resolved"
            )
            self._save_api_cache()

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point: classify all active pairs and report distribution."""
    parser = argparse.ArgumentParser(
        description="Classify Kalshi contracts by resolution date and tier",
        prog="python -m src.live.contract_classifier",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip Kalshi API lookups (regex-only classification)",
    )
    parser.add_argument(
        "--matches-file",
        type=str,
        default="data/live/active_matches.json",
        help="Path to active matches JSON (default: data/live/active_matches.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/live/pair_classifications.json",
        help="Output path for classifications (default: data/live/pair_classifications.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load matches
    matches_path = Path(args.matches_file)
    if not matches_path.exists():
        print(f"ERROR: Matches file not found: {matches_path}")
        return

    with open(matches_path) as f:
        matches = json.load(f)
    print(f"Loaded {len(matches)} matched pairs from {matches_path}")

    # Classify
    classifier = ContractClassifier()
    now = datetime.utcnow()
    use_api = not args.no_api

    results = classifier.classify_all_pairs(matches, now=now, use_api=use_api)

    # Distribution report
    from collections import Counter

    tier_counts = Counter(v["tier"] for v in results.values())
    total = len(results)

    print(f"\n{'='*60}")
    print(f"Classification Report ({total} pairs)")
    print(f"{'='*60}")
    print(f"{'Tier':<15} {'Count':>6} {'Pct':>7}  Example tickers")
    print(f"{'-'*60}")

    for tier_name in ["DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "UNKNOWN"]:
        count = tier_counts.get(tier_name, 0)
        pct = 100 * count / total if total > 0 else 0
        examples = [
            v["kalshi_ticker"]
            for v in results.values()
            if v["tier"] == tier_name
        ][:3]
        ex_str = ", ".join(examples) if examples else "-"
        print(f"{tier_name:<15} {count:>6} {pct:>6.1f}%  {ex_str}")

    print(f"{'='*60}")

    # Show unknowns
    unknowns = [
        v["kalshi_ticker"] for v in results.values() if v["tier"] == "UNKNOWN"
    ]
    if unknowns:
        print(f"\nUNKNOWN tickers ({len(unknowns)}):")
        for t in sorted(set(unknowns)):
            print(f"  {t}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved classifications to {output_path}")


if __name__ == "__main__":
    main()
