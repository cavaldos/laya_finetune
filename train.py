"""Fine-tune Laya, then report results on the test split. Two tasks, found from the data:

- win / loss of backtest trades: `dataset.jsonl` files of Bot_Trade's `src/evaluate/llm_dataset.py`
  (default: ./data/*/dataset.jsonl, .jsonl.gz also read);
- price direction (up / down + how far) after a moment: files of `src/evaluate/price_dataset.py`
  (e.g. --data data_price). Questions `direction` (choice) and `move` (score, 5 levels).

    python train.py --data data_price --profile smoke    # direction task, flow check on the Mac

    python train.py --profile smoke                 # Mac: tiny sample, flow check only
    python train.py --profile colab --name run1     # Colab T4: all trades
    python train.py --profile colab --name run1     # same command again = resume after a disconnect
    python train.py --eval-only runs/run1           # score the test split of a finished run again

See scripts/ for ready-made commands (Mac and Colab).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # ops MPS lacks fall back to CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset_prep as prep  # noqa: E402
import metrics as mx  # noqa: E402

PROFILES = {
    # Flow check on a laptop: a few trades, 1 epoch, short context. The numbers mean nothing.
    "smoke": dict(limit_train=24, limit_val=12, limit_test=12, epochs=1, micro_batch=1, grad_accum=4,
                  max_len=512, sections="auto", save_every_min=60.0, zero_shot=True),
    # Colab / any 16 GB GPU (T4): every trade, 2048 tokens, fp16 + gradient checkpointing.
    "colab": dict(limit_train=None, limit_val=None, limit_test=None, epochs=3, micro_batch=4, grad_accum=8,
                  max_len=2048, sections="auto", save_every_min=20.0, zero_shot=False),
}
CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


# ------------------------------------------------------------------------------------ model

def resolve_model(model: str, models_dir: Path) -> str:
    """Local checkpoint dir, or a Hub id downloaded once into `models_dir`."""
    if Path(model).is_dir():
        return str(Path(model))
    from huggingface_hub import snapshot_download
    target = models_dir / model.replace("/", "__")
    if not (target / "model.safetensors").exists():
        print(f"downloading {model} -> {target} (once) ...", flush=True)
        snapshot_download(model, local_dir=str(target), allow_patterns=CHECKPOINT_FILES)
    from laya.agent import _fix_tokenizer_config
    _fix_tokenizer_config(str(target))
    return str(target)


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    import torch
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"cuda ({p.name}, {p.total_memory / 2**30:.1f} GB)"
    return device


def token_report(model_dir: str, rows_path: Path, max_len: int, head_max_len: int) -> dict:
    """How much of each text fits: the text is cut from the END when it is too long."""
    from transformers import AutoTokenizer
    from laya.common import build_head
    from laya.train import to_internal

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    rows = prep.read_jsonl(rows_path)
    room = max_len
    for qid, question in rows[0]["questions"].items():     # the longest question head leaves the least room
        head_ids, _m, _s = build_head(tok, to_internal(qid, question), head_max_len)
        room = min(room, max_len - len(head_ids) - 2)
    lengths = sorted(len(tok(r["state"], add_special_tokens=False)["input_ids"]) for r in rows)
    cut = sum(n > room for n in lengths)
    return {"state_room_tokens": room, "tokens_median": lengths[len(lengths) // 2],
            "tokens_p95": lengths[int(len(lengths) * 0.95)], "tokens_max": lengths[-1],
            "truncated_rows": cut, "truncated_pct": round(100 * cut / len(lengths), 1)}


# ------------------------------------------------------------------------------- evaluation

def evaluate_run(run_dir: Path, device: str, base_model_dir: str | None, use_symbol_baseline: bool) -> dict:
    import numpy as np
    from trainer import predict_file

    data_dir = run_dir / "data"
    train_meta = prep.read_jsonl(data_dir / "train_meta.jsonl")
    test_meta = prep.read_jsonl(data_dir / "test_meta.jsonl")
    test_rows = prep.read_jsonl(data_dir / "test.jsonl")
    y = np.asarray([1 if m["label"] == "win" else 0 for m in test_meta])
    r = np.asarray([float(m["r"]) for m in test_meta])
    rr = np.asarray([float(m.get("rr") or 0.0) for m in test_meta])
    syms = [m["symbol"] for m in test_meta]
    train_wr = float(np.mean([m["label"] == "win" for m in train_meta]))

    preds = {"base_rate": np.full(len(y), train_wr),
             "logistic_tabular": mx.logistic_baseline(train_meta, test_meta, use_symbol_baseline)}
    if base_model_dir:
        print("scoring the test split with the base checkpoint (zero-shot) ...", flush=True)
        preds["laya_zero_shot"] = predict_file(base_model_dir, test_rows, device)
    print(f"scoring {len(test_rows)} test trades with the fine-tuned checkpoint ...", flush=True)
    preds["laya_finetuned"] = predict_file(str(run_dir / "model"), test_rows, device)

    results = {}
    for name, p in preds.items():
        ok = ~np.isnan(p)
        results[name] = mx.full_report(y[ok], p[ok], r[ok], rr[ok], [s for s, o in zip(syms, ok) if o])
        results[name]["scored"] = int(ok.sum())

    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "symbol", "side", "entry_time", "label", "r", "rr"] + [f"p_win_{k}" for k in preds])
        for i, m in enumerate(test_meta):
            w.writerow([m["id"], m["symbol"], m["side"], m["entry_time"], m["label"], m["r"], m["rr"]]
                       + [None if np.isnan(preds[k][i]) else round(float(preds[k][i]), 4) for k in preds])
    report = {"test_n": int(len(y)), "test_win_rate": round(float(y.mean()), 4),
              "train_win_rate": round(train_wr, 4), "models": results}
    (run_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = mx.format_report(report)
    (run_dir / "test_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return report


def evaluate_direction(run_dir: Path, device: str, base_model_dir: str | None, info: dict) -> dict:
    """Test report of a direction run: p(up) and the move levels vs base rate / indicator baseline."""
    import numpy as np
    from trainer import predict_probs

    data_dir = run_dir / "data"
    train_meta = prep.read_jsonl(data_dir / "train_meta.jsonl")
    test_meta = prep.read_jsonl(data_dir / "test_meta.jsonl")
    test_rows = prep.read_jsonl(data_dir / "test.jsonl")
    y = np.asarray([1 if m["label"] == "up" else 0 for m in test_meta])
    ret = np.asarray([float(m["ret_pct"]) for m in test_meta])
    syms = [m["symbol"] for m in test_meta]
    up_rate = float(np.mean([m["label"] == "up" for m in train_meta]))
    move = info.get("move")

    p_up = {"base_rate": np.full(len(y), up_rate), "logistic_features": mx.logistic_features(train_meta, test_meta)}
    move_probs: dict = {}
    if move:
        shares = np.bincount([m["move_level"] for m in train_meta], minlength=len(move["medians"])) / len(train_meta)
        move_probs["climatology"] = np.tile(shares, (len(y), 1))
    checkpoints = ([("laya_zero_shot", base_model_dir)] if base_model_dir else []) + \
        [("laya_finetuned", str(run_dir / "model"))]
    for name, path in checkpoints:
        print(f"scoring {len(test_rows)} test rows with {name} ...", flush=True)
        out = predict_probs(path, test_rows, device)
        if "direction" in out:
            pos = list(prep.DIRECTION_QUESTION["criteria"]).index("up")
            p_up[name] = np.asarray([np.nan if p is None else float(p[pos]) for p in out["direction"]])
        if move and "move" in out:
            k = len(move["medians"])
            move_probs[name] = np.asarray([np.full(k, np.nan) if p is None else p for p in out["move"]])

    results = {}
    for name, p in p_up.items():
        ok = ~np.isnan(p)
        results[name] = mx.direction_report(y[ok], p[ok], ret[ok], [s for s, o in zip(syms, ok) if o])
    report = {"test_n": int(len(y)), "test_up_rate": round(float(y.mean()), 4), "train_up_rate": round(up_rate, 4),
              "horizon": info.get("horizon"), "abs_move_median_pct": round(float(np.median(np.abs(ret))), 4),
              "models": results}
    if move:
        levels = np.asarray([m["move_level"] for m in test_meta])
        value = np.asarray([float(m["move_value"]) for m in test_meta])
        atr = np.asarray([float(m.get("atr_pct") or 0.0) for m in test_meta])
        report["move"] = {"unit": move["unit"], "edges": move["edges"], "medians": move["medians"], "models": {}}
        for name, probs in move_probs.items():
            ok = ~np.isnan(probs).any(1)
            report["move"]["models"][name] = mx.move_report(probs[ok], levels[ok], move["medians"], value[ok],
                                                            atr[ok], move["unit"], ret[ok])

    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = list(move_probs)
        w.writerow(["id", "symbol", "timeframe", "entry_time", "label", "ret_pct", "atr_pct", "move_level"]
                   + [f"p_up_{k}" for k in p_up] + [f"expected_move_pct_{k}" for k in names])
        for i, m in enumerate(test_meta):
            exp = []
            for k in names:
                e = float(move_probs[k][i] @ np.asarray(move["medians"]))
                exp.append(round(e * float(m.get("atr_pct") or 0.0) if move["unit"] == "atr" else e, 4))
            w.writerow([m["id"], m["symbol"], m.get("timeframe"), m["entry_time"], m["label"], m["ret_pct"],
                        m.get("atr_pct"), m.get("move_level")]
                       + [None if np.isnan(p_up[k][i]) else round(float(p_up[k][i]), 4) for k in p_up] + exp)
    (run_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = mx.format_direction_report(report)
    (run_dir / "test_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return report


# ------------------------------------------------------------------------------------- main

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    ap.add_argument("--data", default=str(HERE / "data"),
                    help="folder searched for */dataset.jsonl(.gz), or one dataset file")
    ap.add_argument("--out", default=str(HERE / "runs"), help="runs are written to <out>/<name>/")
    ap.add_argument("--name", help="run name (default: time + profile). Same name again = resume")
    ap.add_argument("--model", default="convaiinnovations/laya",
                    help="Hub id or local checkpoint dir (also: convaiinnovations/laya-typed-decisions)")
    ap.add_argument("--models-dir", default=str(HERE / "models"), help="where Hub checkpoints are cached")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    # data
    ap.add_argument("--symbols", help="comma list, e.g. BTC,ETH (default: every coin found)")
    ap.add_argument("--test-frac", type=float, default=0.2, help="newest share of the timeline kept for test")
    ap.add_argument("--val-frac", type=float, default=0.1, help="share just before test: best epoch + temperature")
    ap.add_argument("--no-embargo", action="store_true", help="keep trades still open when the next split starts")
    ap.add_argument("--sections", help="auto | compact | facts | all | comma list of ## section names "
                                       "(auto: compact for win/loss, all for direction)")
    ap.add_argument("--horizon", help="direction task: label from this stored horizon, e.g. h4 (default: the main one)")
    ap.add_argument("--move-unit", choices=["atr", "pct"], default="atr",
                    help="direction task: move levels in ATR of the chart TF (default; coins / TFs comparable) or %%")
    ap.add_argument("--questions", default="direction,move", help="direction task: direction,move | direction | move")
    ap.add_argument("--anonymize", action="store_true", help="coin name -> COIN, prices -> R from entry")
    ap.add_argument("--coin-group", action="store_true", help="coin name -> major / large-cap alt / small-cap alt")
    ap.add_argument("--limit-train", type=int, help="random subset of train (0 = all)")
    ap.add_argument("--limit-val", type=int, help="random subset of val (0 = all)")
    ap.add_argument("--limit-test", type=int, help="random subset of test (0 = all)")
    # training
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--micro-batch", type=int, help="sequences per forward pass (lower it on out-of-memory)")
    ap.add_argument("--grad-accum", type=int, help="micro-batches per optimizer step")
    ap.add_argument("--max-len", type=int, help="tokens per sequence (question + text); text is cut beyond it")
    ap.add_argument("--head-max-len", type=int, help="token budget of the question part (default: checkpoint's)")
    ap.add_argument("--encoder-lr", type=float, default=2.5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["rlcd", "soft-ce"], default="rlcd")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-encoder", action="store_true", help="train only the head (fast, little memory)")
    ap.add_argument("--no-amp", action="store_true", help="CUDA: train in fp32 instead of fp16")
    ap.add_argument("--no-checkpointing", action="store_true", help="disable gradient checkpointing (more memory)")
    ap.add_argument("--save-every-min", type=float, help="minutes between resume saves")
    ap.add_argument("--save-optimizer", action="store_true", help="also save AdamW state (+3.4 GB per save)")
    ap.add_argument("--max-hours", type=float, default=0.0, help="stop cleanly after N hours (0 = no limit)")
    ap.add_argument("--seed", type=int, default=0)
    # modes
    ap.add_argument("--prepare-only", action="store_true", help="write the data + token report, no training")
    ap.add_argument("--zero-shot", action=argparse.BooleanOptionalAction, default=None,
                    help="also score the untrained checkpoint on test")
    ap.add_argument("--eval-only", metavar="RUN_DIR", help="only (re)score the test split of a finished run")
    args = ap.parse_args()

    for key, value in PROFILES[args.profile].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    for key in ("limit_train", "limit_val", "limit_test"):
        setattr(args, key, getattr(args, key) or None)
    return args


def sections_of(spec: str):
    if spec in prep.SECTION_PRESETS:
        return prep.SECTION_PRESETS[spec]
    return [s.strip() for s in spec.split(",") if s.strip()]


def main():
    args = parse_args()
    device = pick_device(args.device)
    models_dir = Path(args.models_dir)

    if args.eval_only:
        run_dir = Path(args.eval_only)
        cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        base = resolve_model(cfg["model"], models_dir) if "--zero-shot" in sys.argv else None
        if cfg["data_info"].get("task") == "direction":
            evaluate_direction(run_dir, device, base, cfg["data_info"])
        else:
            evaluate_run(run_dir, device, base, use_symbol_baseline=not (cfg["anonymize"] or cfg["coin_group"]))
        return

    name = args.name or time.strftime("%Y%m%d-%H%M%S") + f"_{args.profile}"
    run_dir = Path(args.out) / name
    data_dir = run_dir / "data"
    cfg_path = run_dir / "run_config.json"
    resuming = (run_dir / "state" / "progress.json").exists()
    print(f"run dir: {run_dir}\ndevice: {describe_device(device)}", flush=True)

    if resuming and cfg_path.exists():
        # Same data and settings as the first session, whatever the command line says now.
        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        keep = {"device", "out", "name", "models_dir", "eval_only", "max_hours", "save_every_min", "data"}
        for key, value in saved.items():
            if key in vars(args) and key not in keep:
                setattr(args, key, value)
        info, tokens = saved["data_info"], saved["tokens"]
        print("resume: reusing the data split and settings of this run", flush=True)
        model_dir = resolve_model(args.model, models_dir)
        head_max_len = saved["head_max_len_used"]
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        info = prep.prepare(Path(args.data), data_dir, test_frac=args.test_frac, val_frac=args.val_frac,
                            sections=sections_of(args.sections), anon=args.anonymize, group=args.coin_group,
                            symbols=args.symbols.split(",") if args.symbols else None,
                            limits={"train": args.limit_train, "val": args.limit_val, "test": args.limit_test},
                            embargo=not args.no_embargo, seed=args.seed, horizon=args.horizon,
                            move_unit=args.move_unit, questions=args.questions.split(","))
        model_dir = resolve_model(args.model, models_dir)
        with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
            head_max_len = args.head_max_len or json.load(f).get("head_max_len", 192)
        tokens = token_report(model_dir, data_dir / "train.jsonl", args.max_len, head_max_len)

    print("data: %d trades from %d file(s), val from %s, test from %s, embargo dropped %d"
          % (info["total_trades"], len(info["files"]), info["val_cutoff"], info["test_cutoff"], info["embargo_dropped"]))
    positive = "up" if info.get("task") == "direction" else "win"
    for split in prep.SPLITS:
        s = info[split]
        if s["n"]:
            print("  %-5s %5d rows, %s %.1f%%, %s -> %s" % (split, s["n"], positive, 100 * s["win_rate"], s["from"], s["to"]))
    if info.get("move"):
        print("move levels (%s): edges %s, medians %s" % (info["move"]["unit"], info["move"]["edges"], info["move"]["medians"]))
    print("tokens: text median %d, p95 %d, max %d; room for text %d -> %d train rows (%.1f%%) cut at the end"
          % (tokens["tokens_median"], tokens["tokens_p95"], tokens["tokens_max"], tokens["state_room_tokens"],
             tokens["truncated_rows"], tokens["truncated_pct"]), flush=True)

    run_cfg = {**vars(args), "device": device, "model_dir": model_dir, "head_max_len_used": head_max_len,
               "data_info": info, "tokens": tokens}
    cfg_path.write_text(json.dumps(run_cfg, indent=2, default=str), encoding="utf-8")
    if args.prepare_only:
        print(f"prepared only; data in {data_dir}")
        return

    from trainer import TrainSettings, train

    settings = TrainSettings(
        epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, loss=args.loss,
        label_smoothing=args.label_smoothing, freeze_encoder=args.freeze_encoder,
        amp=not args.no_amp, checkpointing=not args.no_checkpointing,
        max_len=args.max_len, head_max_len=head_max_len, seed=args.seed,
        save_every_min=args.save_every_min, save_optimizer=args.save_optimizer, max_hours=args.max_hours,
    )
    progress = train(model_dir, run_dir, prep.read_jsonl(data_dir / "train.jsonl"),
                     prep.read_jsonl(data_dir / "val.jsonl"), settings, device)
    run_cfg["progress"] = progress
    cfg_path.write_text(json.dumps(run_cfg, indent=2, default=str), encoding="utf-8")
    if not progress.get("finished"):
        print(f"not finished yet. Resume with the same command (--name {name}).")
        return

    if info.get("task") == "direction":
        evaluate_direction(run_dir, device, model_dir if args.zero_shot else None, info)
    else:
        evaluate_run(run_dir, device, model_dir if args.zero_shot else None,
                     use_symbol_baseline=not (args.anonymize or args.coin_group))
    print(f"saved in {run_dir}: model/ (checkpoint), test_report.txt/json, predictions.csv")


if __name__ == "__main__":
    main()
