#!/bin/bash
# setup_oracle.sh -- One-command Oracle VM deployment for DS340 trading system.
#
# Usage (from local machine):
#   ssh opc@129.80.4.112 'bash -s' < scripts/setup_oracle.sh
#
# Or copy and run on VM:
#   scp scripts/setup_oracle.sh opc@129.80.4.112:~/
#   ssh opc@129.80.4.112 'bash ~/setup_oracle.sh'
#
# This script is IDEMPOTENT -- safe to re-run.
# Memory budget: ~235MB Python footprint within 1GB VM.
# NO PyTorch, NO sentence-transformers, NO stable-baselines3.

set -euo pipefail

REPO_URL="https://github.com/Iansabia/DS340-Project.git"
REPO_DIR="$HOME/DS340-Project"
VENV_DIR="$REPO_DIR/.venv-oracle"
LOG_DIR="$REPO_DIR/logs"
PYTHON_BIN="python3.12"

echo "============================================"
echo "  DS340 Oracle VM Setup"
echo "  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================"

# ------------------------------------------------------------------
# Step 1: Install Python 3.12 + git + dependencies via dnf
# ------------------------------------------------------------------
echo ""
echo "[1/12] Installing system packages..."

# Enable EPEL and CRB for Python 3.12 on Oracle Linux 9
sudo dnf install -y epel-release 2>/dev/null || true
sudo dnf config-manager --set-enabled ol9_codeready_builder 2>/dev/null || true

# Install Python 3.12 + pip + git
sudo dnf install -y python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || {
    echo "  Trying alternative: python3.12 from AppStream..."
    sudo dnf module install -y python3.12 2>/dev/null || true
    sudo dnf install -y python3.12 python3.12-pip git gcc 2>/dev/null || true
}

# Verify Python is available
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    echo "ERROR: $PYTHON_BIN not found after installation."
    echo "  Trying python3..."
    PYTHON_BIN="python3"
    if ! command -v "$PYTHON_BIN" &>/dev/null; then
        echo "FATAL: No Python 3 found. Exiting."
        exit 1
    fi
fi

PYTHON_VERSION=$($PYTHON_BIN --version 2>&1)
echo "  Python: $PYTHON_VERSION"

# ------------------------------------------------------------------
# Step 2: Clone repo (or pull if already exists)
# ------------------------------------------------------------------
echo ""
echo "[2/12] Setting up repository..."

if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repo exists, pulling latest..."
    cd "$REPO_DIR"
    git fetch origin
    git reset --hard origin/master
else
    echo "  Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo "  Repo at: $REPO_DIR"
echo "  Branch: $(git branch --show-current)"

# ------------------------------------------------------------------
# Step 3: Create virtual environment
# ------------------------------------------------------------------
echo ""
echo "[3/12] Creating virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  venv exists, reusing $VENV_DIR"
else
    $PYTHON_BIN -m venv "$VENV_DIR"
    echo "  Created $VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "  Active Python: $(python --version) at $(which python)"

# ------------------------------------------------------------------
# Step 4: Install ONLY lightweight dependencies (no PyTorch!)
# ------------------------------------------------------------------
echo ""
echo "[4/12] Installing minimal pip dependencies..."

pip install --upgrade pip wheel setuptools 2>&1 | tail -1

# CRITICAL: Only install what the trading cycle needs.
# Total footprint ~235MB (well within 800MB available).
pip install \
    pandas \
    numpy \
    pyarrow \
    scikit-learn \
    xgboost \
    requests \
    2>&1 | tail -3

echo "  Installed packages:"
pip list --format=columns 2>/dev/null | grep -iE "pandas|numpy|scikit|xgboost|requests|pyarrow" || true

# ------------------------------------------------------------------
# Step 5: Verify models load correctly
# ------------------------------------------------------------------
echo ""
echo "[5/12] Verifying model loading..."

cd "$REPO_DIR"
python -c "
import sys
sys.path.insert(0, '.')
from src.models.base import BasePredictor

lr = BasePredictor.load('models/deployed/linear_regression.pkl')
print(f'  LR model loaded: {lr.name}')

xgb = BasePredictor.load('models/deployed/xgboost.pkl')
print(f'  XGBoost model loaded: {xgb.name}')

print('  Models: OK')
" || {
    echo "  WARNING: Model loading failed (will retry after full setup)"
}

# ------------------------------------------------------------------
# Step 6: Verify no PyTorch contamination
# ------------------------------------------------------------------
echo ""
echo "[6/12] Verifying no PyTorch installed..."

python -c "
import subprocess, sys
result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                       capture_output=True, text=True)
torch_pkgs = [l for l in result.stdout.splitlines() if 'torch' in l.lower()]
if torch_pkgs:
    print(f'  WARNING: PyTorch packages found: {torch_pkgs}')
    sys.exit(1)
else:
    print('  No PyTorch: OK (memory safe)')
" || echo "  WARNING: torch check failed"

# ------------------------------------------------------------------
# Step 7: Create logs directory
# ------------------------------------------------------------------
echo ""
echo "[7/12] Creating logs directory..."

mkdir -p "$LOG_DIR"
touch "$LOG_DIR/trading_cycle.log"
touch "$LOG_DIR/keepalive.log"
touch "$LOG_DIR/autocommit.log"
echo "  Logs at: $LOG_DIR"

# ------------------------------------------------------------------
# Step 8: Create data/live directory if missing
# ------------------------------------------------------------------
echo ""
echo "[8/12] Ensuring data directories exist..."

mkdir -p "$REPO_DIR/data/live"
echo "  data/live: OK"

# ------------------------------------------------------------------
# Step 9: Configure cron jobs (preserving existing entries)
# ------------------------------------------------------------------
echo ""
echo "[9/12] Configuring cron jobs..."

# Build new crontab, preserving non-DS340 entries
EXISTING_CRON=$(crontab -l 2>/dev/null || echo "")
FILTERED_CRON=$(echo "$EXISTING_CRON" | grep -v "DS340" | grep -v "trading_cycle" | grep -v "oracle_keepalive" | grep -v "oracle_autocommit" | grep -v "^$" || echo "")

NEW_CRON="$FILTERED_CRON
# === DS340 Trading System ===
# Trading cycle: every 15 minutes
*/15 * * * * cd $REPO_DIR && $VENV_DIR/bin/python -m src.live.trading_cycle >> $LOG_DIR/trading_cycle.log 2>&1
# Keep-alive: every 10 minutes (prevent Oracle free-tier reclamation)
*/10 * * * * bash $REPO_DIR/scripts/oracle_keepalive.sh >> $LOG_DIR/keepalive.log 2>&1
# Auto-commit: hourly at :05
5 * * * * cd $REPO_DIR && bash $REPO_DIR/scripts/oracle_autocommit.sh >> $LOG_DIR/autocommit.log 2>&1
"

echo "$NEW_CRON" | crontab -
echo "  Cron jobs installed:"
crontab -l | grep -E "DS340|trading_cycle|keepalive|autocommit" | head -6

# ------------------------------------------------------------------
# Step 10: Create log files with correct ownership
# ------------------------------------------------------------------
echo ""
echo "[10/12] Setting file permissions..."

chmod +x "$REPO_DIR/scripts/oracle_keepalive.sh" 2>/dev/null || true
chmod +x "$REPO_DIR/scripts/oracle_autocommit.sh" 2>/dev/null || true
chmod +x "$REPO_DIR/scripts/setup_oracle.sh" 2>/dev/null || true

# Ensure log files are owned by opc
chown -R "$(whoami)" "$LOG_DIR" 2>/dev/null || true
echo "  Permissions: OK"

# ------------------------------------------------------------------
# Step 11: Configure git for auto-commits
# ------------------------------------------------------------------
echo ""
echo "[11/12] Configuring git for auto-commits..."

cd "$REPO_DIR"
git config user.email "oracle-vm@ds340-trading.bot"
git config user.name "DS340 Trading Bot"
echo "  Git user: $(git config user.name) <$(git config user.email)>"

# ------------------------------------------------------------------
# Step 12: Run one dry-run cycle to verify everything works
# ------------------------------------------------------------------
echo ""
echo "[12/12] Running dry-run trading cycle..."

cd "$REPO_DIR"
python -m src.live.trading_cycle --dry-run 2>&1 | tail -15 || {
    echo "  WARNING: Dry-run had errors (check log above). This may be"
    echo "  expected if no active_matches.json or API connectivity issues."
}

# ------------------------------------------------------------------
# Done!
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "  DS340 Oracle VM Setup Complete!"
echo "============================================"
echo ""
echo "  Cron schedule:"
echo "    */15 * * * *  trading_cycle.py"
echo "    */10 * * * *  oracle_keepalive.sh"
echo "    5 * * * *     oracle_autocommit.sh"
echo ""
echo "  Monitor commands:"
echo "    tail -f $LOG_DIR/trading_cycle.log"
echo "    tail -f $LOG_DIR/keepalive.log"
echo "    tail -f $LOG_DIR/autocommit.log"
echo ""
echo "  Check status:"
echo "    cd $REPO_DIR && $VENV_DIR/bin/python -m src.live.trading_cycle --status"
echo ""
echo "  Memory usage:"
echo "    free -h"
echo ""
