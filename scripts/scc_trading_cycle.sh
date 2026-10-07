#!/bin/bash
#
# SCC cron entrypoint for the adaptive trading cycle.
#
# Called every 15 minutes from crontab:
#   */15 * * * * /usr4/ds340/iansabia/DS340-Project/scripts/scc_trading_cycle.sh
#
# Replaces the previous inline crontab one-liner which had two bugs:
#   1. Sourced /etc/profile.d/lmod.sh which does not exist on SCC login
#      nodes — every run died before reaching Python.
#   2. Shell operator precedence caused git commit to run unconditionally
#      after any failure, not just when there were changes to commit.
#
# Design:
#   - Uses a LOGIN shell to get SCC's environment (no manual module sourcing).
#   - set -u catches unset vars; we handle errors explicitly instead of set -e
#     so one bad step (e.g. transient git push failure) doesn't abort the
#     whole cron cycle without logging.
#   - Logs everything to $HOME/trading.log with ISO timestamps so you can
#     `tail -f ~/trading.log` to debug.
#   - Only commits when git diff --cached shows actual changes AND the
#     Python step succeeded.

set -u

PROJECT_DIR="/usr4/ds340/iansabia/DS340-Project"
LOG_FILE="$HOME/trading.log"
PY="$PROJECT_DIR/.venv/bin/python"

# Redirect all further output (stdout + stderr) to the log file.
exec >> "$LOG_FILE" 2>&1

# Timestamped section header
echo ""
echo "========================================================"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SCC cycle start"
echo "========================================================"

# Helper: log a line with timestamp
log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

# Step 1: cd into project directory
if ! cd "$PROJECT_DIR"; then
    log "ERROR: cannot cd to $PROJECT_DIR"
    exit 1
fi

# Step 2: verify python interpreter is reachable without loading modules.
# The .venv was created with `module load python3/3.12.4 && python -m venv .venv`
# so .venv/bin/python is a symlink to the absolute path of the 3.12 interpreter.
# This means the cron does NOT need `module load` at all — huge simplification.
if [ ! -x "$PY" ]; then
    log "ERROR: python interpreter missing at $PY"
    log "       Recreate with: module load python3/3.12.4 && python -m venv .venv"
    exit 1
fi

# Step 3: pull latest code from main branch.
# Binary files (positions.db, bars.parquet) can't auto-merge, so if
# pull --rebase hits a conflict the repo gets stuck in a broken state
# that persists across ALL future cycles. The fix: detect any dirty
# git state (rebase in progress, merge conflict, detached HEAD) and
# hard-reset to origin/master. We lose any uncommitted local data
# from this cycle but the NEXT cycle runs clean.
_git_recover() {
    log "WARN: git in bad state — recovering via hard reset to origin/master"
    git rebase --abort 2>/dev/null || true
    git merge --abort 2>/dev/null || true
    git checkout master 2>/dev/null || true
    git fetch -q origin 2>/dev/null || true
    git reset --hard origin/master 2>/dev/null || true
}

# Check for leftover broken state from a previous failed cycle
if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ] || [ -f .git/MERGE_HEAD ]; then
    _git_recover
fi

log "git pull --rebase --autostash"
if ! git pull -q --rebase --autostash; then
    _git_recover
fi

# Step 4: run the adaptive trading cycle.
# Captures its own return code so we can decide whether to commit.
log "running trading_cycle --cycle"
if "$PY" -m src.live.trading_cycle --cycle; then
    log "trading_cycle: success"
    CYCLE_OK=1
else
    log "trading_cycle: FAILED (exit $?)"
    CYCLE_OK=0
fi

# Step 4b: paper_trader REMOVED from login node (2026-04-12).
# With 10,750+ pairs the paper_trader's LSTM/GRU retraining burns
# >15 min CPU, which trips SCC's login-node watchdog. The live
# trading system (strategy.py in step 4) handles all position
# management independently. Paper trader research logging can run
# as an occasional batch job (qsub) when needed for the data-scaling
# study, but NOT every 15 minutes on the login node.

# Step 5: stage live data files.
# These are gitignored for local dev but force-added for collection runs.
# paper_trades*.jsonl covers BOTH the legacy archive (paper_trades.jsonl,
# frozen at ~81MB) and all per-UTC-day rotated files
# (paper_trades_YYYY-MM-DD.jsonl). Task #27.
log "staging data/live files"
git add -f data/live/bars.parquet 2>/dev/null || true
git add -f data/live/paper_trades*.jsonl 2>/dev/null || true
git add data/live/positions.db 2>/dev/null || true
git add data/live/position_history.jsonl 2>/dev/null || true
git add data/live/pair_classifications.json 2>/dev/null || true
git add data/live/pair_mapping.json 2>/dev/null || true

# Step 6: only commit if trading_cycle succeeded AND there are actual changes.
if [ "$CYCLE_OK" -eq 0 ]; then
    log "skipping commit — trading_cycle failed this run"
elif git diff --cached --quiet; then
    log "nothing to commit"
else
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if git commit -q -m "auto: scc update ${TIMESTAMP}"; then
        log "committed"
        # Rebase-retry push: GHA discovery and trading both push to
        # master, so non-fast-forward rejections are common. Retry
        # up to 3 times. If the rebase hits a binary conflict,
        # recover and abandon THIS cycle's commit (next cycle
        # will start clean).
        PUSHED=0
        for attempt in 1 2 3; do
            if git push -q 2>/dev/null; then
                log "pushed successfully (attempt $attempt)"
                PUSHED=1
                break
            fi
            log "push rejected (attempt $attempt) — rebasing"
            if ! git pull --rebase -q --autostash 2>/dev/null; then
                log "WARN: rebase conflict on push retry — recovering"
                _git_recover
                break
            fi
        done
        if [ "$PUSHED" -eq 0 ]; then
            log "WARN: push failed after retries — data saved locally, will retry next cycle"
        fi
    else
        log "WARN: git commit failed"
    fi
fi

log "cycle complete"
