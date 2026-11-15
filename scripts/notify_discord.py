"""Wait for a training run to end, then post its result to a Discord webhook.

The webhook URL is a secret: it is read from the DISCORD_WEBHOOK environment variable and is
never stored in this repo.

    DISCORD_WEBHOOK=https://discord.com/api/webhooks/... \
        python scripts/notify_discord.py --run-dir /content/runs/four1 --log /content/four1.log --wait four1

--wait NAME   poll until no process has "--name NAME" in its command line (the training), then post.
Without --wait it posts right away (e.g. to test the webhook: add --message "test").
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def post(url: str, text: str) -> None:
    for start in range(0, len(text), 1900):          # Discord: 2000 chars per message
        chunk = text[start:start + 1900]
        body = json.dumps({"content": chunk}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "User-Agent": "laya-finetune-notify/1.0"})
        urllib.request.urlopen(req, timeout=30).read()
        time.sleep(1)


def running(name: str) -> bool:
    out = subprocess.run(["pgrep", "-f", f"train.py.*--name {name}"], capture_output=True, text=True)
    return bool(out.stdout.split())


def tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--log")
    ap.add_argument("--wait", help="run name to wait for")
    ap.add_argument("--message", help="post this text and exit")
    ap.add_argument("--poll", type=float, default=30)
    args = ap.parse_args()

    url = os.environ.get("DISCORD_WEBHOOK", "")
    if not url.startswith("https://discord.com/api/webhooks/"):
        sys.exit("set DISCORD_WEBHOOK to the webhook URL")
    if args.message:
        post(url, args.message)
        return

    if args.wait:
        time.sleep(20)                                # let the training process appear
        while running(args.wait):
            time.sleep(args.poll)
    run = Path(args.run_dir)
    report = run / "test_report.txt"
    if report.exists():
        post(url, f"Laya run `{run.name}` finished.\n```\n{report.read_text()}\n```")
    else:
        post(url, f"Laya run `{run.name}` ended WITHOUT a test report. Last log lines:\n"
                  f"```\n{tail(Path(args.log)) if args.log else '(no log)'}\n```")


if __name__ == "__main__":
    main()
