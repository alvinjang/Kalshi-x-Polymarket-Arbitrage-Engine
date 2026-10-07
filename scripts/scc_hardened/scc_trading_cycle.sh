#!/bin/bash
#
# SCC cron entrypoint for the adaptive trading cycle. HARDENED VERSION.
#
# Changes vs scripts/scc_trading_cycle.sh (the original):
#
#   Layer 1 — write-side integrity guards (Step 5):
#       Before `git add`-ing bars.parquet or positions.db, verify each
#       file is non-zero and readable in its expected format (pyarrow
#       for parquet, sqlite PRAGMA integrity_check for the db). If a
#       file fails the check, do NOT stage it. The previous valid
#       version stays in the index from prior commits; no corrupted
#       payload reaches origin.
#
#   Layer 3 — recover-side guard (_git_recover):
#       After `git reset --hard origin/master`, verify that origin's
#       bars.parquet is itself non-zero. If origin is corrupt, exit
#       with code 2 and log loudly. This prevents the recover function
#       from silently propagating origin-side corruption back into the
#       working tree (which would then get re-committed under a new
#       "auto: scc update" message by Step 6 — the exact failure
#       pathway that broke production on 2026-05-20).
#
# Called every 15 minutes from crontab:
#   */15 * * * * /usr4/ds340/iansabia/DS340-Project/scripts/scc_trading_cycle.sh
#
# See scripts/scc_hardened/README.md for the full incident write-up.

set -u

PROJECT_DIR="/usr4/ds340/iansabia/DS340-Project"
LOG_FILE="$HOME/trading.log"
PY="$PROJECT_DIR/.venv/bin/python"

# Phase A canonical-oil deployment gates. Default state: shadow on,
# enabled off. To flip to A2 narrow-replacement, set CANONICAL_OIL_ENABLED=true
# in $HOME/.scc_env or override at the crontab line. See
# scripts/scc_hardened/phase_a_v3.md section (f) for the truth table.
export CANONICAL_OIL_ENABLED="${CANONICAL_OIL_ENABLED:-false}"
export CANONICAL_OIL_SHADOW="${CANONICAL_OIL_SHADOW:-true}"
[ -f "$HOME/.scc_env" ] && . "$HOME/.scc_env"

# Redirect all further output (stdout + stderr) to the log file.
exec >> "$LOG_FILE" 2>&1

# Timestamped section header
echo ""
echo "========================================================"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SCC cycle start (hardened)"
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
if [ ! -x "$PY" ]; then
    log "ERROR: python interpreter missing at $PY"
    log "       Recreate with: module load python3/3.12.4 && python -m venv .venv"
    exit 1
fi

# Step 3: pull latest code from main branch.
# Layer 3 integrated into _git_recover.
_git_recover() {
    log "WARN: git in bad state — recovering via hard reset to origin/master"
    git rebase --abort 2>/dev/null || true
    git merge --abort 2>/dev/null || true
    git checkout master 2>/dev/null || true
    git fetch -q origin 2>/dev/null || true
    git reset --hard origin/master 2>/dev/null || true

    # Layer 3: after recovery, verify origin/master itself is not corrupt.
    # If origin has a 0-byte bars.parquet, do NOT silently continue —
    # this is the exact pathway that propagated corruption indefinitely
    # in the 2026-05-20 incident. Exit with a distinct code so cron
    # logs make the cause obvious.
    if [ -f data/live/bars.parquet ] && [ ! -s data/live/bars.parquet ]; then
        log "FATAL: origin/master has 0-byte data/live/bars.parquet — refusing to continue"
        log "       Recovery (manual): git checkout <good-sha> -- data/live/bars.parquet"
        log "       Last known good (incident reference): 04625b8"
        exit 2
    fi
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
log "running trading_cycle --cycle"
if "$PY" -m src.live.trading_cycle --cycle; then
    log "trading_cycle: success"
    CYCLE_OK=1
else
    log "trading_cycle: FAILED (exit $?)"
    CYCLE_OK=0
fi

# Step 5: stage live data files, WITH LAYER 1 INTEGRITY GUARDS on binary files.
#
# Helper: stage a parquet file only if it is non-zero and pyarrow-readable.
# Returns 0 on success or "nothing to do" (missing file is fine);
# returns 1 if the file exists but failed validation (do not stage).
_stage_if_valid_parquet() {
    local path="$1"
    if [ ! -f "$path" ]; then
        return 0
    fi
    local size
    size=$(stat -c%s "$path" 2>/dev/null)
    if [ -z "$size" ]; then
        size=$(stat -f%z "$path" 2>/dev/null || echo 0)
    fi
    if [ "$size" -eq 0 ]; then
        log "GUARD: $path is 0 bytes — refusing to stage, keeping prior origin version"
        return 1
    fi
    if ! "$PY" -c "import pyarrow.parquet as pq; pq.read_metadata('$path')" 2>/dev/null; then
        log "GUARD: $path is non-zero but failed pyarrow read — refusing to stage"
        return 1
    fi
    git add -f "$path"
    log "staged: $path (${size} bytes, parquet OK)"
    return 0
}

# Helper: stage a sqlite db only if it is non-zero and passes integrity_check.
_stage_if_valid_sqlite() {
    local path="$1"
    if [ ! -f "$path" ]; then
        return 0
    fi
    local size
    size=$(stat -c%s "$path" 2>/dev/null)
    if [ -z "$size" ]; then
        size=$(stat -f%z "$path" 2>/dev/null || echo 0)
    fi
    if [ "$size" -eq 0 ]; then
        log "GUARD: $path is 0 bytes — refusing to stage"
        return 1
    fi
    local check
    check=$("$PY" -c "import sqlite3; c = sqlite3.connect('$path'); r = c.execute('PRAGMA integrity_check').fetchone(); c.close(); print(r[0] if r else 'fail')" 2>/dev/null)
    if [ "$check" != "ok" ]; then
        log "GUARD: $path failed sqlite integrity_check (got: ${check:-error}) — refusing to stage"
        return 1
    fi
    git add "$path"
    log "staged: $path (${size} bytes, sqlite OK)"
    return 0
}

# Layer 4 — refuse to stage files that are symlinks to off-tree paths
# (e.g. /projectnb, /scratch). These are Q1/Q2 off-tree live-state files;
# gitignore covers the day-to-day case but git add -f bypasses gitignore,
# so this guard is the belt-and-suspenders against accidents.
_is_offsite_symlink() {
    local path="$1"
    [ -L "$path" ] || return 1
    local target
    target=$(readlink "$path")
    case "$target" in
        /projectnb/*|/scratch/*|/share/*) return 0 ;;
        *) return 1 ;;
    esac
}

log "staging tracked data/live files (Layer 1 integrity + Layer 4 off-tree guards)"
# bars.parquet and active_matches.json are PERMANENTLY off-tree (Q1/Q2).
# Never stage them regardless of whether they're a symlink or a regular file
# on disk. Both states can occur: symlink = intended state, regular file =
# transient post-pull artifact where git removed the symlink during checkout
# and the collector recreated a local file before the operator could
# re-symlink. In either case, do not stage.
if [ -e data/live/bars.parquet ] || [ -L data/live/bars.parquet ]; then
    if _is_offsite_symlink data/live/bars.parquet; then
        log "LAYER 4: data/live/bars.parquet is off-tree symlink — not staging"
    else
        log "LAYER 4: data/live/bars.parquet is a regular file (transient post-pull artifact?) — not staging; bars live off-tree per Q1/Q2, re-symlink with: ln -sfn /projectnb/ds340/projects/iansabia/live_state/bars.parquet data/live/bars.parquet"
    fi
fi
if [ -e data/live/active_matches.json ] || [ -L data/live/active_matches.json ]; then
    if _is_offsite_symlink data/live/active_matches.json; then
        log "LAYER 4: data/live/active_matches.json is off-tree symlink — not staging"
    else
        log "LAYER 4: data/live/active_matches.json is a regular file (transient post-pull artifact?) — not staging; active_matches lives off-tree per Q1/Q2"
    fi
fi
git add -f data/live/paper_trades*.jsonl 2>/dev/null || true
_stage_if_valid_sqlite data/live/positions.db || true
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
        # Rebase-retry push with up to 3 attempts.
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
