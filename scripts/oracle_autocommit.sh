#!/bin/bash
# oracle_autocommit.sh -- Hourly auto-commit of trading results to GitHub.
#
# Stages live data files (positions.db, bars.parquet, position_history.jsonl,
# pair_classifications.json), commits with a timestamp and position count,
# and pushes to origin. Runs via cron at :05 past each hour.
#
# Cron entry (installed by setup_oracle.sh):
#   5 * * * * cd ~/DS340-Project && bash ~/DS340-Project/scripts/oracle_autocommit.sh >> ~/DS340-Project/logs/autocommit.log 2>&1

set -euo pipefail

# Navigate to repo root (handle both cron and manual invocation)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# Only commit if there are changes in data/live/
if git diff --quiet data/live/ 2>/dev/null && \
   git diff --cached --quiet data/live/ 2>/dev/null && \
   [ -z "$(git ls-files --others --exclude-standard data/live/ 2>/dev/null)" ]; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): no changes to commit"
    exit 0
fi

# Stage live data files (ignore errors for missing files)
git add data/live/bars.parquet \
        data/live/positions.db \
        data/live/position_history.jsonl \
        data/live/pair_classifications.json \
        data/live/active_matches.json \
        data/live/resolution_dates_cache.json \
        2>/dev/null || true

# Also stage any new untracked files in data/live/
git add data/live/ 2>/dev/null || true

# Check if anything was actually staged
if git diff --cached --quiet 2>/dev/null; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): nothing staged to commit"
    exit 0
fi

# Count open positions for commit message
OPEN_POS=$(python3 -c "
import sqlite3, os
db = 'data/live/positions.db'
if os.path.exists(db):
    conn = sqlite3.connect(db)
    try:
        count = conn.execute('SELECT COUNT(*) FROM positions').fetchone()[0]
    except Exception:
        count = 0
    conn.close()
    print(count)
else:
    print(0)
" 2>/dev/null || echo "?")

TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
git commit -m "auto: trading results ${TIMESTAMP} (${OPEN_POS} open positions)" || {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): commit failed"
    exit 0
}

# Push with one retry
git push origin master || git push origin main || {
    sleep 5
    git push origin master || git push origin main || {
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): push failed, will retry next hour"
        exit 0
    }
}

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): committed and pushed (${OPEN_POS} open positions)"
