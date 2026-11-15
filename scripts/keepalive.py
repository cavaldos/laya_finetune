"""Keep an interactive Kaggle session busy while a training run is in progress.

Run it in the background from the notebook, next to train.py:

    python scripts/keepalive.py --every-sec 60 --until-run four1

It writes a heartbeat line to stdout every --every-sec seconds and stops when no train.py
process with "--name <run>" is left. It does not touch the browser: the browser is the
notebook UI, which only the page itself can drive.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def running(name: str) -> bool:
    out = subprocess.run(["pgrep", "-f", f"train.py.*--name {name}"], capture_output=True, text=True)
    return bool(out.stdout.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every-sec", type=float, default=60)
    ap.add_argument("--until-run", required=True, help="stop when this train.py --name is gone")
    args = ap.parse_args()

    beats = 0
    while running(args.until_run):
        beats += 1
        print(f"heartbeat {beats} {time.strftime('%H:%M:%S')}", flush=True)
        time.sleep(args.every_sec)
    print("train run ended; keepalive stops", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
