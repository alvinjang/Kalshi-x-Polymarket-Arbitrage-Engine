# Deployment runbook — scc_hardened scripts to SCC

**Operator: Ian Sabia.** Run all commands from your **Mac terminal**
(verify with `hostname` — should NOT return `scc1`). Do not run any of
these through Claude. The hardened script files are in
`scripts/scc_hardened/` in this repo; they need to land at
`/usr4/ds340/iansabia/DS340-Project/scripts/` on SCC.

Total time: ~10-15 minutes including the manual smoke test.

---

## Pre-flight

### Verify you're on your Mac

```bash
hostname
```

Should return your Mac's name (e.g. `Ians-MacBook-Pro.local`), not
`scc1`. If it returns `scc1`, type `exit` first.

### Verify SSH alias works

```bash
ssh scc 'hostname && pwd'
```

Should print `scc1` and `/usr4/ds340/iansabia`. If it prompts for a
password and you don't want to type it again, the `ControlMaster`
multiplex in `~/.ssh/config` should keep it cached for 4 hours.

### Verify the hardened scripts exist locally

```bash
ls -la "/Users/iansabia/Desktop/DS340 Project/scripts/scc_hardened/"
```

Should list `scc_trading_cycle.sh`, `scc_discover_markets.sh`,
`README.md`, `deploy.md`.

---

## Step 1: Pull latest code on SCC (gets the hardened scripts into the repo on SCC)

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project && git pull --rebase --autostash'
```

This brings the `scripts/scc_hardened/` directory onto SCC alongside
the originals at `scripts/`. You'll see the new files but the cron is
still calling the originals — that's the intended state until you swap
them.

If the pull fails with a conflict on `data/live/bars.parquet` or
similar binary file, the local SCC checkout has uncommitted dirty data.
Recover with:

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project && git fetch origin && git reset --hard origin/master'
```

(Same `git reset --hard` the cron script does in `_git_recover`.)

---

## Step 2: Back up the existing scripts on SCC

Date-stamped backups in the same directory. If anything goes wrong,
restore via `cp <bak> <original>`.

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project/scripts && cp scc_trading_cycle.sh scc_trading_cycle.sh.bak.$(date +%Y-%m-%d) && cp scc_discover_markets.sh scc_discover_markets.sh.bak.$(date +%Y-%m-%d) && ls -la scc_trading_cycle.sh* scc_discover_markets.sh*'
```

Should show four files total: the two originals and two `.bak.YYYY-MM-DD`
copies.

---

## Step 3: Copy the hardened versions in place

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project && cp scripts/scc_hardened/scc_trading_cycle.sh scripts/scc_trading_cycle.sh && cp scripts/scc_hardened/scc_discover_markets.sh scripts/scc_discover_markets.sh'
```

### Verify the swap

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project && head -10 scripts/scc_trading_cycle.sh && echo "---" && head -10 scripts/scc_discover_markets.sh'
```

Both should show the `HARDENED VERSION` comment in the file header. If
you see the old headers (no "HARDENED" mention) the copy didn't take —
re-run Step 3.

---

## Step 4: Set executable permissions

`cp` typically preserves the existing file's mode, but verify anyway.

```bash
ssh scc 'chmod +x /usr4/ds340/iansabia/DS340-Project/scripts/scc_trading_cycle.sh /usr4/ds340/iansabia/DS340-Project/scripts/scc_discover_markets.sh && ls -la /usr4/ds340/iansabia/DS340-Project/scripts/scc_trading_cycle.sh /usr4/ds340/iansabia/DS340-Project/scripts/scc_discover_markets.sh'
```

Should show `-rwxr-xr-x` (or similar with `x` for the owner) on both
files.

---

## Step 5: Manual smoke test — run scc_trading_cycle.sh once

This is the critical step. Run the hardened script manually OUTSIDE
the cron schedule and verify the new log markers appear.

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project && bash scripts/scc_trading_cycle.sh; echo "EXIT_CODE=$?"'
```

Expected behavior:
- The script runs to completion (one full cycle, typically 30-90 sec)
- `EXIT_CODE=0` (success) or `EXIT_CODE=2` (origin-side corruption
  detected by Layer 3, which would mean the read-side fix to GHA
  hasn't synced or origin re-corrupted somehow)

### Check the log for new markers

```bash
ssh scc 'tail -50 $HOME/trading.log'
```

Look for these new log lines (proves the hardened script is what ran):

```
[...] SCC cycle start (hardened)             ← header changed
[...] staging data/live files (with Layer 1 integrity checks)
[...] staged: data/live/bars.parquet (NNN bytes, parquet OK)
[...] staged: data/live/positions.db (NNN bytes, sqlite OK)
```

If you instead see (or any combination):

```
[...] GUARD: data/live/bars.parquet is 0 bytes — refusing to stage, keeping prior origin version
[...] GUARD: data/live/bars.parquet is non-zero but failed pyarrow read — refusing to stage
[...] GUARD: data/live/positions.db failed sqlite integrity_check ...
```

…then the guards are doing their job: a corrupt file was detected and
not staged. The cycle continues without pushing corruption.

If you see `FATAL: origin/master has 0-byte data/live/bars.parquet`
the Layer 3 guard fired. Origin must have re-corrupted somehow. STOP,
do not roll forward, and ping Claude to diagnose.

### Optional: verify no unexpected commits were pushed by the smoke test

If the smoke test cycle succeeded AND the data files actually changed,
the script will have pushed a new `auto: scc update ...` commit. Check:

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project && git log --oneline -3'
```

The most recent commit should be either a clean `auto: scc update ...`
(if data changed) or whatever was at the head before (if nothing
changed). If you see a commit touching `bars.parquet` with size 0 in
its diff, **do not proceed**. Roll back via Step 7.

---

## Step 6: Let the cron pick up the new scripts

Nothing to do — cron reads the script files each run, so the next
scheduled cycle (`*/15 * * * *` for trading, next at the next quarter
hour; `17 */3 * * *` for discover, next at the next `:17` of a 3-hour
slot) will use the hardened versions automatically.

### Watch the first few cron cycles

```bash
ssh scc 'tail -f $HOME/trading.log'
```

Or open a separate terminal and check periodically:

```bash
ssh scc 'tail -100 $HOME/trading.log | grep -E "cycle start|cycle complete|staged|GUARD|ERROR|FATAL"'
```

Six clean cycles in a row (about 90 minutes for trading) is a good
"stable" signal.

### Verify origin-side commits

From your Mac:

```bash
cd "/Users/iansabia/Desktop/DS340 Project"
git fetch origin && git log --oneline origin/master -5
```

You should see fresh `auto: scc update ...` commits with clean diffs
(non-zero file sizes, no unexpected files in `data: scc discover
markets` commits).

---

## Step 7: Rollback (if anything looks wrong)

Restore the originals from the backups created in Step 2:

```bash
ssh scc 'cd /usr4/ds340/iansabia/DS340-Project/scripts && cp scc_trading_cycle.sh.bak.$(date +%Y-%m-%d) scc_trading_cycle.sh && cp scc_discover_markets.sh.bak.$(date +%Y-%m-%d) scc_discover_markets.sh && chmod +x scc_trading_cycle.sh scc_discover_markets.sh && head -5 scc_trading_cycle.sh'
```

Last command's `head -5` confirms the rollback (should show the
original header, not "HARDENED VERSION").

The cron will pick up the rolled-back scripts on the next cycle.

---

## After deployment is verified stable (6-12 cycles)

Ping Claude with confirmation. We then:

1. Resume the canonical-oil deployment plan (Phase A shadow mode).
2. Optionally remove the backup files from SCC once you're confident
   (kept for at least a week as a safety net).
3. Optionally consider whether to also harden `scc_retrain_batch.sh`
   (the third cron job, lower-risk because it only modifies model
   pickles which are small and don't gate downstream readers).
