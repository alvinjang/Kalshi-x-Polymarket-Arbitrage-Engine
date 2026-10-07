# Live (Real-Money) Trading Setup

This guide walks through the user-actionable steps to move the system from paper trading to real money on Kalshi + Polymarket.

> **Default state: OFF.** No real orders are placed unless `LIVE_TRADING=true` is set in the environment. If unset or set to anything else, the system continues to run as paper-trading only.

---

## 0. Canary config (target state)

| Lever | Value |
|---|---|
| Bankroll | $50 Kalshi + $50 Polymarket |
| Per-trade size | $2 per leg ($4 per arb trade) |
| Daily kill-switch | $10 realized loss (halts new entries only) |
| Per-exchange exposure cap | $50 |
| Position sizing model | Fixed-dollar (graduate to fractional Kelly after 50+ real trades) |

All defaults live in `src/live/risk_manager.py` and can be overridden per-call.

---

## 1. Kalshi account + API keys (~10 min, account already exists)

Kalshi uses RSA-PSS signed REST requests. You generate an RSA keypair in their portal; the public half stays with Kalshi, the private half lives only on your machine.

### Steps
1. Log into <https://kalshi.com/>
2. Go to **Account → API Keys** (or Settings → Developer → API)
3. Click **Generate New Key** → save the `Access Key ID` (UUID-like)
4. Kalshi will give you a **PEM-encoded RSA private key**. Save it to disk:
   ```bash
   mkdir -p ~/.kalshi
   chmod 700 ~/.kalshi
   # paste contents into this file:
   nano ~/.kalshi/private_key.pem
   chmod 600 ~/.kalshi/private_key.pem
   ```
5. **NEVER commit this file.** Verify it's outside the repo: `realpath ~/.kalshi/private_key.pem` should not be inside `DS340-Project/`.
6. Add the env vars to a local-only file (also not committed):
   ```bash
   # ~/.env.live  (or any file outside the repo)
   export LIVE_TRADING=true
   export KALSHI_API_KEY_ID=<your-access-key-id>
   export KALSHI_PRIVATE_KEY_PATH=$HOME/.kalshi/private_key.pem
   ```
7. **Sanity check** (paper mode auth, no order placed):
   ```bash
   source ~/.env.live
   .venv/bin/python -c "
   from src.live.kalshi_orders import KalshiOrderClient
   bal = KalshiOrderClient().get_balance()
   print(bal)
   "
   ```
   Expected output: `{'balance': N}` where N is your account balance in cents.

### Funding
Wire/ACH/debit USD into your Kalshi account. Start with **$50** for the canary. Don't fund more until you've validated end-to-end.

---

## 2. Polymarket wallet + USDC (~30-60 min, no wallet yet)

This is the bigger user-side task because we're crossing on-chain. The flow:
**(create wallet) → (acquire USDC on Polygon) → (export private key) → (set env vars)**

### Step 2a: create a wallet
**Recommended for first-time: MetaMask**
1. Install MetaMask browser extension: <https://metamask.io/>
2. Create a NEW wallet (don't reuse an existing one — this wallet will hold a private key in a file).
3. **Write down the seed phrase on paper.** Store in a safe place. Anyone with the seed phrase owns the funds.
4. In MetaMask, add the **Polygon** network if it's not already present (Settings → Networks → Add Polygon Mainnet, RPC `https://polygon-rpc.com`, chain ID `137`, currency `MATIC`).

### Step 2b: get USDC on Polygon
Polymarket trades on Polygon-USDC. You need both:
- A tiny bit of **MATIC** for gas (a few cents' worth, ~0.5 MATIC ≈ $0.30 plenty)
- **$50 USDC** for the canary bankroll

Options to acquire:
1. **Fiat on-ramp directly to Polygon** (simplest): MoonPay, Transak, or Ramp inside MetaMask. Select Polygon network, USDC currency, ~$55 (the extra covers fees).
2. **Bridge from Ethereum mainnet**: only if you already have USDC on mainnet. Use the official Polygon bridge at <https://portal.polygon.technology/>. This is ~$2-5 in mainnet gas + bridging time.
3. **Centralized exchange withdrawal**: Coinbase / Kraken support direct USDC withdrawal to Polygon. Cheapest if you already have an account.

### Step 2c: deposit USDC into Polymarket
Polymarket uses a Safe-based proxy wallet (signature_type=1 in our config). Funding flow:
1. Go to <https://polymarket.com/>, click **Deposit** in the top right.
2. Polymarket will show your **funder address** (this is your Safe wallet on Polygon, derived from your MetaMask EOA).
3. Send USDC from MetaMask to that funder address.
4. Wait ~1 minute for confirmation.

**Save your funder address** — you'll need it for `POLYMARKET_FUNDER_ADDRESS`.

### Step 2d: export private key for the API
1. In MetaMask: click your account → **Account Details → Show Private Key**. Enter your password.
2. **Copy and save**. This is your `POLYMARKET_PRIVATE_KEY` (0x-prefixed hex).
3. **Treat this like your Kalshi PEM:** never commit, restrict file permissions.

### Step 2e: set env vars
```bash
# Append to ~/.env.live
export POLYMARKET_PRIVATE_KEY=0x...your...key
export POLYMARKET_FUNDER_ADDRESS=0x...your...funder
export POLYMARKET_CHAIN_ID=137
```

### Sanity check
```bash
source ~/.env.live
.venv/bin/pip install py-clob-client>=0.18
.venv/bin/python -c "
from src.live.polymarket_orders import PolymarketOrderClient
bal = PolymarketOrderClient().get_balance_usdc()
print(f'USDC balance: \${bal:.2f}')
"
```
Expected: `USDC balance: $50.00` (or whatever you deposited).

---

## 3. Activation checklist

Before flipping `LIVE_TRADING=true` on SCC:

- [ ] Kalshi account funded with $50, API keys work
- [ ] Polymarket wallet funded with $50 USDC, API works
- [ ] Local sanity check both pass
- [ ] **One known gap**: pair_mapping.json does NOT yet contain Polymarket YES/NO token IDs (only the market-level ID). The strategy will skip every live entry until this is enriched — this is a SAFE default but means no real trades will happen yet. To fix: extend `src/live/market_discovery.py` to fetch and store `polymarket_yes_token_id` + `polymarket_no_token_id` per pair (Polymarket CLOB has a `/markets/{condition_id}` endpoint that returns both outcome tokens).
- [ ] Decide: run live locally first (~30 min observation) or push straight to SCC?

### To arm live trading on SCC
```bash
ssh scc1.bu.edu
cd ~/DS340-Project
# Append to wherever your cron's env comes from (e.g. ~/.profile or
# a sourced file referenced in scripts/scc_run_cycle.sh)
cat >> ~/.profile.live <<'EOF'
export LIVE_TRADING=true
export KALSHI_API_KEY_ID=...
export KALSHI_PRIVATE_KEY_PATH=$HOME/.kalshi/private_key.pem
export POLYMARKET_PRIVATE_KEY=0x...
export POLYMARKET_FUNDER_ADDRESS=0x...
EOF
chmod 600 ~/.profile.live
```

Then modify the cron-invoked script to `source ~/.profile.live` before running.

### Emergency halt
At any time, on the box where the cron runs:
```bash
export EMERGENCY_HALT=true
```
This overrides `LIVE_TRADING=true` and blocks all new entries instantly. Existing positions exit per their normal rules — DO NOT force-flatten unless you mean to lock in current P&L.

---

## 4. After 50+ live trades

Re-evaluate:
- Per-trade size: graduate from $2 fixed to fractional Kelly using observed win rates
- Daily kill-switch: tune based on actual variance
- Exposure cap: raise toward 2x bankroll if performance justifies it

---

## Architecture summary

```
src/live/risk_manager.py        # gate: LIVE_TRADING + size + daily-loss + exposure
src/live/kalshi_orders.py       # Kalshi REST client (RSA-PSS signed)
src/live/polymarket_orders.py   # Polymarket CLOB client (py-clob-client wrapper)
src/live/order_executor.py      # two-leg coordinator + atomicity recovery
src/live/strategy.py            # wires the above into the entry loop
```

Tests: `tests/test_risk_manager.py`, `tests/test_kalshi_orders.py`, `tests/test_polymarket_orders.py`, `tests/test_order_executor.py` — 26 tests covering the full live stack without network/credentials.
