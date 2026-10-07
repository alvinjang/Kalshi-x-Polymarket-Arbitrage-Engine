#!/bin/bash -l
#
# SCC batch job: run the market discovery pipeline on a compute node.
# HARDENED VERSION.
#
# Changes vs scripts/scc_discover_markets.sh (the original):
#
#   Layer 2 — defensive unstaging before commit (Step 5):
#       Discovery is intended to commit ONLY data/live/active_matches.json.
#       In the 2026-05-20 incident, a previous failed scc_trading_cycle.sh
#       left bars.parquet (and other files) staged in the index. When this
#       script ran `git commit`, the leftover staged files were included
#       under the misleading "data: scc discover markets" commit message.
#       The hardened version scans `git diff --cached --name-only` BEFORE
#       committing and unstages anything that isn't active_matches.json,
#       logging each unstaged file. This prevents cross-script corruption
#       commits regardless of what state Layer 1's guards left behind.
#
# Submitted from cron via qsub every 3 hours:
#   17 */3 * * * /bin/bash -lc 'qsub /usr4/ds340/iansabia/DS340-Project/scripts/scc_discover_markets.sh'
#
# See scripts/scc_hardened/README.md for the full incident write-up.

#$ -N discover-markets
#$ -j y
#$ -o $HOME/logs/
#$ -l h_rt=00:30:00
#$ -pe omp 4
#$ -l mem_per_core=4G

set -u

PROJECT_DIR="/usr4/ds340/iansabia/DS340-Project"
PY="$PROJECT_DIR/.venv/bin/python"

# Timestamped section header
echo ""
echo "========================================================"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SCC discovery batch start (hardened)"
echo "  JOB_ID:   ${JOB_ID:-unknown}"
echo "  HOSTNAME: ${HOSTNAME:-unknown}"
echo "  NSLOTS:   ${NSLOTS:-unknown}"
echo "========================================================"

log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

# Pin thread counts to match the slot allocation so PyTorch doesn't
# over-subscribe cores and get our job killed for CPU hoarding.
if [ -n "${NSLOTS:-}" ]; then
    export OMP_NUM_THREADS="$NSLOTS"
    export MKL_NUM_THREADS="$NSLOTS"
    export OPENBLAS_NUM_THREADS="$NSLOTS"
    log "threads capped at $NSLOTS"
fi

# Step 1: cd into project
if ! cd "$PROJECT_DIR"; then
    log "FATAL: cannot cd to $PROJECT_DIR"
    exit 1
fi

# Step 2: verify python
if [ ! -x "$PY" ]; then
    log "FATAL: python interpreter missing at $PY"
    exit 1
fi

# Step 3: pull latest code (also picks up filter rule updates)
log "git pull --rebase --autostash"
if ! git pull -q --rebase --autostash; then
    log "WARN: git pull failed — continuing with local code"
fi

# Step 4: run discovery (writes to data/live/active_matches.json)
log "running scripts/discover_markets.py"
if PYTHONPATH="$PROJECT_DIR" "$PY" scripts/discover_markets.py; then
    log "discovery: success"
    DISCOVERY_OK=1
else
    log "discovery: FAILED (exit $?)"
    DISCOVERY_OK=0
fi

# Step 5: commit + push if discovery succeeded and there are changes.
# LAYER 2 GUARD: discovery is only supposed to commit active_matches.json.
# Any other staged file in the index is a leftover from a previous failed
# trading-cycle run and would otherwise get committed under the wrong
# message. Unstage it loudly.
if [ "$DISCOVERY_OK" -eq 0 ]; then
    log "skipping commit — discovery failed this run"
else
    # Layer 2: scan and unstage anything that isn't active_matches.json
    UNEXPECTED_COUNT=0
    while IFS= read -r f; do
        if [ -n "$f" ] && [ "$f" != "data/live/active_matches.json" ]; then
            log "GUARD: unexpected staged file $f (leftover from prior run?) — unstaging before discover commit"
            git reset HEAD -- "$f" >/dev/null 2>&1 || true
            UNEXPECTED_COUNT=$((UNEXPECTED_COUNT + 1))
        fi
    done < <(git diff --cached --name-only 2>/dev/null)
    if [ "$UNEXPECTED_COUNT" -gt 0 ]; then
        log "GUARD: unstaged $UNEXPECTED_COUNT unexpected file(s); discover commit will include only active_matches.json"
    fi

    # Layer 4: active_matches.json is off-tree (Q1/Q2) at /projectnb via symlink.
    # Writes happen at /projectnb directly; gitignore + this guard prevent staging.
    if [ -L "data/live/active_matches.json" ]; then
        TARGET=$(readlink "data/live/active_matches.json")
        case "$TARGET" in
            /projectnb/*|/scratch/*|/share/*)
                log "LAYER 4: data/live/active_matches.json is off-tree symlink ($TARGET) — not staging"
                ;;
            *)
                log "WARN: data/live/active_matches.json symlinks to unexpected target ($TARGET) — staging"
                git add data/live/active_matches.json 2>/dev/null || true
                ;;
        esac
    else
        git add data/live/active_matches.json 2>/dev/null || true
    fi

    if git diff --cached --quiet; then
        log "no new pairs to commit"
    else
        TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        N=$("$PY" -c "import json; print(len(json.load(open('data/live/active_matches.json'))))" 2>/dev/null || echo "?")

        if git commit -q -m "data: scc discover markets ${TIMESTAMP} (${N} total pairs)"; then
            log "committed (${N} total pairs)"
            if git push -q; then
                log "pushed successfully"
            else
                log "WARN: git push failed — will retry next cycle"
            fi
        else
            log "WARN: git commit failed"
        fi
    fi
fi

log "batch job complete"
