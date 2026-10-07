#!/usr/bin/env python3
"""Interactive wizard for live-trading credential setup.

Walks the user through each step they MUST do themselves, opening the
right browser URLs at the right moments and writing a complete
~/.env.live file at the end.

This is the shortest path from "I want to trade live" to "all checks
green" — but it cannot do KYC, generate keys, or move money for you.
What it CAN do is hold your hand and make sure nothing's forgotten.

Usage:
    python scripts/setup_wizard.py
"""
from __future__ import annotations

import getpass
import os
import sys
import webbrowser
from pathlib import Path

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(msg: str) -> None:
    bar = "=" * 60
    print(f"\n{BOLD}{bar}\n{msg}\n{bar}{RESET}\n")


def prompt(question: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    if secret:
        val = getpass.getpass(f"{question}{suffix}: ").strip()
    else:
        val = input(f"{question}{suffix}: ").strip()
    return val or default


def confirm(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{question} {suffix} ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def open_url(url: str, label: str) -> None:
    print(f"  → Opening: {GREEN}{url}{RESET}")
    print(f"    ({label})")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"  {YELLOW}Couldn't auto-open browser ({e}). Click the URL above.{RESET}")


def step_intro() -> None:
    banner("LIVE TRADING SETUP WIZARD")
    print("This wizard will walk you through the user-side steps that")
    print("I (Claude) cannot do for you:")
    print()
    print("  1. Generate Kalshi API keys           (~5 min)")
    print("  2. Create a Polygon wallet            (~10 min)")
    print("  3. Fund the wallet with USDC          (~30-60 min, varies)")
    print("  4. Deposit USDC to Polymarket         (~2 min)")
    print("  5. Save credentials to ~/.env.live    (this script writes it)")
    print()
    print(f"{YELLOW}You will see and handle private keys + seed phrases.{RESET}")
    print(f"{YELLOW}Stay alone. Don't share your screen.{RESET}")
    print()
    if not confirm("Ready to begin?"):
        print("Exited. Re-run when you have time.")
        sys.exit(0)


def step_kalshi_demo_first() -> bool:
    banner("STEP 0: Kalshi demo first?")
    print("RECOMMENDED: start on Kalshi's demo environment (real markets,")
    print("paper money) so you can validate fills work end-to-end before")
    print("committing real USD.")
    print()
    return confirm("Set up demo credentials first?", default=True)


def step_kalshi_keys(env: str) -> tuple[str, str]:
    banner(f"STEP 1: Kalshi API keys ({env})")
    portal_url = ("https://demo.kalshi.co/account/api-keys"
                  if env == "demo"
                  else "https://kalshi.com/account/api-keys")
    open_url(portal_url, "Kalshi API keys portal")
    print()
    print("In the portal:")
    print(f"  1. Click {BOLD}Generate New Key{RESET}")
    print(f"  2. Copy the {BOLD}Access Key ID{RESET} (a UUID).")
    print(f"  3. Download the {BOLD}private key (PEM file){RESET}.")
    print()
    key_id = prompt("Paste the Access Key ID")
    if not key_id:
        print(f"{RED}No key ID provided. Skipping Kalshi.{RESET}")
        return "", ""

    default_path = str(Path.home() / ".kalshi" / "private_key.pem")
    print()
    print(f"Save the PEM file you downloaded to:  {default_path}")
    print(f"Recommended commands:")
    print(f"  mkdir -p {Path(default_path).parent}")
    print(f"  chmod 700 {Path(default_path).parent}")
    print(f"  # move the downloaded file:")
    print(f"  mv ~/Downloads/*.pem {default_path}")
    print(f"  chmod 600 {default_path}")
    print()
    key_path = prompt("Path to your saved PEM file", default=default_path)
    expanded = os.path.expanduser(key_path)
    if not Path(expanded).exists():
        print(f"{YELLOW}Warning: {expanded} doesn't exist yet. "
              f"Make sure you save the PEM there before running live_check.{RESET}")
    else:
        print(f"{GREEN}✓ Found PEM at {expanded}{RESET}")
    return key_id, key_path


def step_polymarket_wallet() -> tuple[str, str]:
    banner("STEP 2-4: Polymarket wallet + USDC")
    print("Polymarket lives on the Polygon blockchain. You need:")
    print(f"  - A {BOLD}Polygon wallet{RESET}")
    print(f"  - At least ${BOLD}50 USDC{RESET} bridged to that wallet")
    print(f"  - A small amount of {BOLD}MATIC{RESET} for gas (~$0.50)")
    print()
    print("Easiest path for a first-time user: MetaMask + Polymarket on-ramp.")
    print()
    open_url("https://metamask.io/", "Install MetaMask")
    print("After installing, create a NEW wallet (don't reuse an existing one).")
    print(f"{YELLOW}WRITE DOWN THE 12-WORD SEED PHRASE ON PAPER.{RESET}")
    print(f"{YELLOW}Anyone with the seed phrase owns the funds — store safely.{RESET}")
    print()
    if not confirm("Have you created the wallet and saved the seed phrase?"):
        print("Pause here. Re-run when you're ready.")
        return "", ""

    open_url("https://polymarket.com/", "Polymarket (click Deposit)")
    print()
    print("On Polymarket:")
    print(f"  1. Connect the MetaMask wallet you just created")
    print(f"  2. Click {BOLD}Deposit{RESET} (top right)")
    print(f"  3. Polymarket will show your {BOLD}funder address{RESET}")
    print(f"     (this is a proxy wallet derived from your EOA)")
    print(f"  4. Use a fiat on-ramp inside MetaMask, OR send USDC from a")
    print(f"     CEX (Coinbase/Kraken support Polygon USDC withdrawal) to")
    print(f"     the funder address. Start with $50.")
    print()
    funder = prompt("Paste your Polymarket funder address (0x...)")
    if not funder.startswith("0x") or len(funder) != 42:
        print(f"{RED}That doesn't look like a Polygon address (need 0x + 40 hex chars).{RESET}")
        funder = prompt("Try again", default=funder)

    print()
    print("Now export your EOA private key:")
    print(f"  In MetaMask: Account → {BOLD}Account Details → Show Private Key{RESET}")
    print(f"  {YELLOW}Treat this like a password. Never paste it into any website.{RESET}")
    print()
    privkey = prompt("Paste your private key (input is hidden)", secret=True)
    if privkey and not privkey.startswith("0x"):
        privkey = "0x" + privkey
    return privkey, funder


def write_env_file(creds: dict, env_path: Path) -> None:
    banner("WRITING ~/.env.live")
    contents = f"""# Generated by setup_wizard.py — fill in any blanks manually.
# Source this file before running live trading:  source ~/.env.live

export LIVE_TRADING=false          # flip to "true" only after live_check passes
export EMERGENCY_HALT=false
export KALSHI_ENVIRONMENT={creds.get('kalshi_env', 'demo')}

export KALSHI_API_KEY_ID={creds.get('kalshi_key_id', '')}
export KALSHI_PRIVATE_KEY_PATH={creds.get('kalshi_key_path', '')}

export POLYMARKET_PRIVATE_KEY={creds.get('poly_privkey', '')}
export POLYMARKET_FUNDER_ADDRESS={creds.get('poly_funder', '')}
export POLYMARKET_CHAIN_ID=137
"""
    env_path.write_text(contents)
    env_path.chmod(0o600)
    print(f"  Wrote {env_path}")
    print(f"  Permissions: 600 (owner-only)")
    print()
    print(f"Next:")
    print(f"  {BOLD}source {env_path}{RESET}")
    print(f"  {BOLD}python scripts/live_check.py{RESET}")
    print()
    print("When live_check is all-green, edit ~/.env.live and change")
    print(f"{BOLD}LIVE_TRADING=false{RESET} to {BOLD}LIVE_TRADING=true{RESET}, then run a")
    print("trading cycle. The next entry will fire a real order under the")
    print("canary limits ($2/leg, $10 daily kill).")


def main() -> int:
    step_intro()

    # Kalshi
    kalshi_env = "demo" if step_kalshi_demo_first() else "prod"
    kalshi_key_id, kalshi_key_path = step_kalshi_keys(kalshi_env)

    # Polymarket
    if confirm("\nProceed to Polymarket wallet setup?", default=True):
        poly_privkey, poly_funder = step_polymarket_wallet()
    else:
        poly_privkey, poly_funder = "", ""
        print("Skipped Polymarket — only single-exchange (Kalshi) trades will fire.")

    # Write file
    env_path = Path.home() / ".env.live"
    if env_path.exists():
        if not confirm(f"\n{env_path} exists. Overwrite?", default=False):
            env_path = Path.home() / ".env.live.new"
            print(f"Will write to {env_path} instead.")

    write_env_file(
        {
            "kalshi_env": kalshi_env,
            "kalshi_key_id": kalshi_key_id,
            "kalshi_key_path": kalshi_key_path,
            "poly_privkey": poly_privkey,
            "poly_funder": poly_funder,
        },
        env_path,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled.")
        sys.exit(1)
