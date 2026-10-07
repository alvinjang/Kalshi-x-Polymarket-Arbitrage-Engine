#!/bin/bash -l
#
# SCC batch job: run the market discovery pipeline on a compute node.
#
# Submitted from cron via qsub every 3 hours. Discovery is too heavy
# for login nodes (sentence-transformers BERT encoding burns ~2-5 min
# of CPU time across multiple cores), which trips SCC's 15-min CPU
# watchdog. The trading cycle stays on login-node cron (lightweight,
# ~10-20s CPU); discovery goes here.
#
# Submit manually with:
#   qsub /usr4/ds340/iansabia/DS340-Project/scripts/scc_discover_markets.sh
#
# Or from crontab (every 3 hours at :17, offset from :00 slot spike):
#   17 */3 * * * /bin/bash -lc 'qsub /usr4/ds340/iansabia/DS340-Project/scripts/scc_discover_markets.sh'
#
# Monitor with:
#   qstat                           # queue status
#   ls -lt ~/logs/discover.*.log    # recent run outputs
#   tail -f ~/logs/discover.LATEST_JOBID.log

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
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SCC discovery batch start"
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

# Step 5: commit + push if discovery succeeded and there are changes
if [ "$DISCOVERY_OK" -eq 0 ]; then
    log "skipping commit — discovery failed this run"
else
    git add data/live/active_matches.json 2>/dev/null || true

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
