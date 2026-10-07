# SCC cron script hardening — three-layer corruption guards

Hardened replacements for the two SCC cron scripts that produced the
2026-05-20 corruption incident. These versions are checked in here for
review and **must be manually deployed** to SCC (see `deploy.md`).

The originals live at `scripts/scc_trading_cycle.sh` and
`scripts/scc_discover_markets.sh` in the repo, and on SCC at
`/usr4/ds340/iansabia/DS340-Project/scripts/`. The hardened versions
in this directory are byte-for-byte ready to copy over.

## The incident

- 2026-05-20 04:48 UTC: last successful Collect & Paper Trade GHA run
- 2026-05-20 04:49 UTC: auto-retrain commit (model pickles only, unrelated)
- 2026-05-20 07:28 UTC: commit `6b8f3c5 data: scc discover markets ...`
  pushed `data/live/bars.parquet` to origin at **0 bytes** (was 11.4 MB)
- 2026-05-20 08:41 UTC through 2026-05-23 12:39 UTC: **30+ consecutive**
  Collect & Paper Trade workflow failures with
  `pyarrow.lib.ArrowInvalid: Parquet file size is 0 bytes`
- 2026-05-23 ~14:00 UTC: `bars.parquet` restored from `04625b8`,
  read-side guard added to `.github/workflows/collect-and-trade.yml`,
  validation `workflow_dispatch` run succeeded

The read-side fix (in the GHA workflow) prevents future 0-byte payloads
from breaking the workflow loudly. The hardened scripts in this directory
are the **write-side** fix that prevents 0-byte payloads from reaching
origin in the first place.

## Root cause chain (verbatim from the triage)

1. A prior `scc_trading_cycle.sh` was interrupted (SCC CPU watchdog,
   write race, or sigkill) while the Python step was writing
   `bars.parquet`. Left a 0-byte file on disk.
2. Next `scc_trading_cycle.sh` ran. Step 5 staged the 0-byte file via
   `git add -f`. The cycle then failed/skipped commit (the Step 6
   `CYCLE_OK` check guards commits, but not the Step 5 staging).
3. The staged 0-byte file persisted in the index across cycles.
4. `scc_discover_markets.sh` ran, did its `git pull --rebase --autostash`,
   then staged `active_matches.json`, then `git commit`. The commit
   included everything in the index — including the stale 0-byte
   `bars.parquet` — under the misleading commit message
   `data: scc discover markets`.
5. `git push` to origin. Done. 30+ GHA failures over 3 days.
6. Even worse: `_git_recover()` in the trading cycle would `git reset
   --hard origin/master`, propagating the corruption back to SCC.
   Self-healing was impossible.

## The three layers

### Layer 1 — write-side integrity guards (in `scc_trading_cycle.sh` Step 5)

Helper functions `_stage_if_valid_parquet` and `_stage_if_valid_sqlite`
replace the unconditional `git add -f` calls for `bars.parquet` and
`positions.db`. Each helper:

- Returns silently (success) if the file is missing (nothing to do)
- Refuses to stage and logs `GUARD: ...` if the file is 0 bytes
- Refuses to stage and logs `GUARD: ...` if the file fails its
  format-specific integrity check (pyarrow `read_metadata` for parquet,
  sqlite `PRAGMA integrity_check` for the db)
- Stages the file and logs `staged: ... (NNN bytes, format OK)` only
  if both checks pass

When a corrupted file is refused, the previous valid version stays in
the index from prior commits. No bad payload reaches origin. The next
cycle that produces a valid file will eventually commit clean.

JSONL and JSON files are not guarded (they're append-only or
small-rewrite text; corruption manifests as garbled lines but does
not break readers the way a 0-byte parquet does).

### Layer 2 — defensive unstaging (in `scc_discover_markets.sh` Step 5)

Discovery only intends to commit `active_matches.json`. The hardened
version scans `git diff --cached --name-only` immediately before
committing and unstages any file that isn't `active_matches.json`,
logging each one as `GUARD: unexpected staged file ... unstaging`.

This is a defense-in-depth measure. Layer 1 should already prevent
corrupt files from being staged in the first place; Layer 2 ensures
that **even if a leftover staged file slips through** (e.g., a future
script bug, a manual `git add` left behind by a maintainer), the
discover commit cannot accidentally include it.

The commit-message provenance is restored: `data: scc discover markets`
commits will contain only `data/live/active_matches.json`.

### Layer 3 — recover-side guard (in `scc_trading_cycle.sh _git_recover`)

After `git reset --hard origin/master`, the function checks whether the
file freshly pulled from origin is itself corrupt. If origin's
`bars.parquet` is 0 bytes, the function exits the entire script with
code `2` and logs the recovery command for the operator:

```
git checkout <good-sha> -- data/live/bars.parquet
```

The incident reference commit (`04625b8`) is named in the log message
so a future operator hitting the same failure has a direct breadcrumb.

This prevents the self-propagating loop where: corrupted origin → local
reset to corrupted state → trading cycle runs and re-commits the same
corrupted file → push → origin stays corrupted → repeat.

## What changed vs the originals

```
scripts/scc_trading_cycle.sh:
  + Layer 1: _stage_if_valid_parquet, _stage_if_valid_sqlite helpers
  + Layer 1: replace `git add -f data/live/bars.parquet` and
             `git add data/live/positions.db` with helper calls
  + Layer 3: post-recovery integrity check in _git_recover, exit 2
             on bad origin

scripts/scc_discover_markets.sh:
  + Layer 2: pre-commit scan + unstage of unexpected files
```

The trading-cycle script grows from ~150 lines to ~180. The discover
script grows from ~95 to ~110. No external dependencies added; both
scripts use the existing project venv at `.venv/bin/python` for the
pyarrow and sqlite checks.

## What did NOT change

- Cron schedules (`*/15 * * * *` for trading, `17 */3 * * *` for
  discover via qsub)
- Output paths (`$HOME/trading.log`, `$HOME/logs/`)
- Commit-message templates (`auto: scc update ...`,
  `data: scc discover markets ...`)
- The qsub directives (`#$ -N`, `#$ -pe omp 4`, etc.) in the
  discover script
- Any Python or GHA workflow code

## Deployment

See `deploy.md` in this same directory. SCC deployment is a manual
operator step — not done automatically by anything in this repo, and
not done by any AI tool call. The operator (you) runs the SSH commands
in `deploy.md` from a local Mac terminal.

After deployment, the first signal that the fix is working will be a
clean `auto: scc update ...` commit from `scc_trading_cycle.sh`
containing the expected file set with no `GUARD: ...` log lines (or,
better, log lines showing `staged: ... (NNN bytes, parquet OK)`).
