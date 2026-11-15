"""Post training progress to Discord every N minutes until the run ends.

The webhook is a secret. It is read from the DISCORD_WEBHOOK environment variable, or on Kaggle
from Add-ons -> Secrets (name DISCORD_WEBHOOK). It is never stored in this repo.

    python scripts/progress_discord.py --name four1 --log /kaggle/working/four1.log --every-min 15
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def webhook() -> str:
    url = os.environ.get("DISCORD_WEBHOOK", "")
    if not url:
        try:
            from kaggle_secrets import UserSecretsClient  # available inside Kaggle notebooks
            url = UserSecretsClient().get_secret("DISCORD_WEBHOOK")
        except Exception:
            url = ""
    if not url.startswith("https://discord.com/api/webhooks/"):
        raise SystemExit("no webhook: set DISCORD_WEBHOOK or add the Kaggle secret DISCORD_WEBHOOK")
    return url


def post(url: str, text: str) -> None:
    body = json.dumps({"content": text[:1900]}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "User-Agent": "laya-finetune-notify/1.0"})
    urllib.request.urlopen(req, timeout=30).read()


def running(name: str) -> bool:
    out = subprocess.run(["pgrep", "-f", f"train.py.*--name {name}"], capture_output=True, text=True)
    return bool(out.stdout.split())


def tail(path: Path, lines: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log yet)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name (the train.py --name)")
    ap.add_argument("--log", required=True)
    ap.add_argument("--every-min", type=float, default=15)
    ap.add_argument("--lines", type=int, default=8)
    args = ap.parse_args()

    url = webhook()
    log = Path(args.log)
    started = time.strftime("%H:%M")
    post(url, f"Laya `{args.name}` progress reporter started {started}; updates every {args.every_min:g} min.")
    while True:
        time.sleep(args.every_min * 60)
        if not running(args.name):
            break
        post(url, f"Laya `{args.name}` at {time.strftime('%H:%M')}:\n```\n{tail(log, args.lines)}\n```")
    report = None
    for p in Path(log.parent).rglob("test_report.txt"):
        report = p
    if report:
        post(url, f"Laya `{args.name}` finished.\n```\n{report.read_text()}\n```")
    else:
        post(url, f"Laya `{args.name}` ended WITHOUT a test report. Last log lines:\n```\n{tail(log, 15)}\n```")


if __name__ == "__main__":
    main()
