"""Training loop for Laya on one GPU (Colab T4) or Apple Silicon, built from laya's own pieces.

Why not `laya.train.finetune`: on Colab a session can drop at any time, and finetune can only
start over. This loop adds:
- resume: weights (+ optionally the optimizer) and the exact position in the epoch are saved
  every `save_every_min` minutes and at every epoch end, written atomically (safe on Drive);
- `max_hours`: stop cleanly (state saved) before a session limit kills the process;
- a chronological validation split scored after every epoch; the best epoch (lowest val
  log-loss) is kept as the final model and its probability temperature is fitted on val;
- fp16 autocast for scoring too (laya scores in fp32, ~3x slower on a T4), batches sorted by length.

Same objective, encoding and head as laya 0.4.32 (`laya.train`): pin that version.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from laya.common import collate_items, temp_bucket
from laya.train import (
    _forward,
    draw_option_order,
    encode_item,
    fit_temperature_map,
    items_from_rows,
    load_checkpoint,
    rlcd_loss,
    save_checkpoint,
    sigma_at,
    soft_ce_loss,
)

import metrics as mx


@dataclass
class TrainSettings:
    epochs: int = 3
    micro_batch: int = 4
    grad_accum: int = 8
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    loss: str = "rlcd"                 # "rlcd" (laya default) or "soft-ce"
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    label_smoothing: float = 0.0
    freeze_encoder: bool = False
    amp: bool = True                   # fp16 autocast, CUDA only
    checkpointing: bool = True         # gradient checkpointing: much less memory, ~30% slower
    max_len: int = 2048
    head_max_len: int = 192
    seed: int = 0
    log_every: int = 20                # micro-batches
    save_every_min: float = 20.0
    save_optimizer: bool = False       # +3.4 GB per save; without it a resume restarts AdamW moments
    max_hours: float = 0.0             # 0 = no limit
    eval_batch: int = 16


# ------------------------------------------------------------------------------------ items

def load_rows(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_items(tok, rows: Sequence[dict], max_len: int, head_max_len: int, label_smoothing: float = 0.0):
    """`(items, index)`: one item per usable (row, question) and the row index it came from.
    A win/loss row has one question; a direction row two (direction + move)."""
    items, index = [], []
    for i, row in enumerate(rows):
        its, _skipped = items_from_rows(tok, [row], max_len, head_max_len, label_smoothing=label_smoothing)
        items += its
        index += [i] * len(its)
    return items, index


# ---------------------------------------------------------------------------------- scoring

@torch.no_grad()
def score(model, tok, items, device: torch.device, max_len: int, head_max_len: int,
          batch_size: int = 16, amp: bool = True) -> List[np.ndarray]:
    """Raw logits per item (options in canonical order), batches sorted by length."""
    was_training = model.training
    model.eval()
    use_amp = amp and device.type == "cuda"
    order = sorted(range(len(items)), key=lambda i: len(items[i]["state_ids"]))
    out: List[Optional[np.ndarray]] = [None] * len(items)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = collate_items([[encode_item(tok, items[i], max_len, head_max_len) for i in idx]], tok.pad_token_id)
        logits = _forward(model, batch, device, use_amp, detach_encoder=False).cpu().numpy()
        for row, i in zip(logits, idx):
            out[i] = row[: items[i]["k"]]
    if was_training:
        model.train()
    return out


POSITIVE = ("win", "up")   # the option whose probability a binary question reports


def item_probs(items, logits, temperature: Sequence[float], by_options: Dict[str, float]) -> List[np.ndarray]:
    """Option probabilities of each item (canonical option order) after temperature scaling."""
    out = []
    for item, z in zip(items, logits):
        t = float(by_options.get(temp_bucket(item["qtype"], item["k"]), temperature[item["qtype"]]))
        z = np.asarray(z, dtype=float) / max(t, 1e-4)
        e = np.exp(z - z.max())
        out.append(e / e.sum())
    return out


def positive_index(item) -> Optional[int]:
    crit = item["q"]["crit"]
    if isinstance(crit, dict):
        keys = list(crit.keys())
        for label in POSITIVE:
            if label in keys:
                return keys.index(label)
    return None


def win_probability(items, logits, temperature: Sequence[float], by_options: Dict[str, float]) -> np.ndarray:
    """p(win) / p(up) of binary items."""
    probs = item_probs(items, logits, temperature, by_options)
    return np.asarray([p[positive_index(it)] for it, p in zip(items, probs)])


def labels_of(items) -> np.ndarray:
    return np.asarray([1 if it["target"][positive_index(it)] > 0.5 else 0 for it in items])


def val_metrics(items, logits, temperature: Sequence[float], by_options: Dict[str, float]) -> dict:
    """Mean log-loss of the true option over every item (all questions), plus AUC / spread of the
    binary question (win or up). For a win/loss run this is the binary log-loss, as before."""
    probs = item_probs(items, logits, temperature, by_options)
    true = [int(np.argmax(it["target"])) for it in items]
    ll = float(np.mean([-np.log(max(float(p[t]), mx.EPS)) for p, t in zip(probs, true)]))
    binary = [j for j, it in enumerate(items) if positive_index(it) is not None]
    auc, spread = None, 0.0
    if binary:
        y = np.asarray([1 if true[j] == positive_index(items[j]) else 0 for j in binary])
        p = np.asarray([probs[j][positive_index(items[j])] for j in binary])
        auc, spread = mx.auc(y, p), float(p.max() - p.min())
    return {"log_loss": round(ll, 4), "auc": None if auc is None else round(auc, 4), "p_spread": round(spread, 4)}


def fit_temperature(items, logits) -> dict:
    records = [(it["qtype"], z, it["target"], it["k"]) for it, z in zip(items, logits)]
    return fit_temperature_map(records)


# ------------------------------------------------------------------------------- save / load

def _atomic_dir_replace(tmp: Path, final: Path) -> None:
    old = final.with_name(final.name + ".old")
    if final.exists():
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def save_state(run_dir: Path, model, optimizer, scheduler, scaler, progress: dict, save_optimizer: bool) -> None:
    from safetensors.torch import save_file
    t0 = time.time()
    tmp = run_dir / "state.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in model.state_dict().items()},
              str(tmp / "model.safetensors"))
    extra = {"scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None}
    if save_optimizer:
        extra["optimizer"] = optimizer.state_dict()
    torch.save(extra, tmp / "trainer.pt")
    (tmp / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    _atomic_dir_replace(tmp, run_dir / "state")
    print(f"  [saved resume state: epoch {progress['epoch']} step {progress['step']} "
          f"({time.time() - t0:.0f}s)]", flush=True)


def load_state(run_dir: Path, model):
    """Weights into `model`; returns `(progress, extra)` or `(None, None)` when there is no state."""
    state = run_dir / "state"
    if not (state / "progress.json").exists():
        return None, None
    from safetensors.torch import load_file
    model.load_state_dict(load_file(str(state / "model.safetensors")), strict=True)
    extra = torch.load(state / "trainer.pt", map_location="cpu", weights_only=False)
    return json.loads((state / "progress.json").read_text(encoding="utf-8")), extra


# ------------------------------------------------------------------------------------ train

def train(model_dir: str, run_dir: Path, train_rows: List[dict], val_rows: List[dict],
          s: TrainSettings, device_name: str) -> dict:
    """Train (or resume), keep the best epoch in `<run_dir>/model`, fit its temperature on val."""
    device = torch.device(device_name)
    model, tok, cfg = load_checkpoint(model_dir)
    progress, extra = load_state(run_dir, model)
    if progress and progress.get("finished"):
        print("training already finished for this run; delete <run>/state to train again")
        return progress
    if progress:
        print(f"resuming from epoch {progress['epoch'] + 1} step {progress['step']}", flush=True)

    t_items = time.time()
    train_items, _ = build_items(tok, train_rows, s.max_len, s.head_max_len, s.label_smoothing)
    val_items, _ = build_items(tok, val_rows, s.max_len, s.head_max_len) if val_rows else ([], [])
    if not train_items:
        raise ValueError("no usable training rows")
    print(f"tokenized {len(train_items)} train / {len(val_items)} val items in {time.time() - t_items:.0f}s", flush=True)

    amp = s.amp and device.type == "cuda"
    if s.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif s.checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = s.checkpointing
    model.to(device).train()
    if s.freeze_encoder:
        model.encoder.eval()

    groups = [{"params": [p for n, p in model.named_parameters()
                          if not n.startswith("encoder.") and p.requires_grad], "lr": s.head_lr}]
    if not s.freeze_encoder:
        groups.insert(0, {"params": [p for n, p in model.named_parameters()
                                     if n.startswith("encoder.") and p.requires_grad], "lr": s.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=s.weight_decay)
    steps_per_epoch = math.ceil(len(train_items) / s.micro_batch)
    updates_per_epoch = math.ceil(steps_per_epoch / s.grad_accum)
    total_updates = updates_per_epoch * s.epochs
    warmup = max(1, int(total_updates * s.warmup_frac))
    base_lrs = [g["lr"] for g in groups]

    def lr_factor(update: int) -> float:
        if update < warmup:
            return (update + 1) / warmup
        frac = (update - warmup) / max(1, total_updates - warmup)
        return 1.0 - (1.0 - s.min_lr / max(base_lrs)) * 0.5 * (1 - math.cos(math.pi * min(1.0, frac)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    if extra:
        scheduler.load_state_dict(extra["scheduler"])
        for group, base in zip(optimizer.param_groups, base_lrs):   # LambdaLR applies it only on step()
            group["lr"] = base * lr_factor(scheduler.last_epoch)
        if scaler is not None and extra.get("scaler"):
            scaler.load_state_dict(extra["scaler"])
        if extra.get("optimizer"):
            optimizer.load_state_dict(extra["optimizer"])
        else:
            print("  (no optimizer state saved: AdamW restarts its moments)", flush=True)

    progress = progress or {"epoch": 0, "step": 0, "update": 0, "best_val_log_loss": None,
                            "best_epoch": None, "history": [], "finished": False}
    print(f"train {len(train_items)} items, {steps_per_epoch} micro-batches/epoch, {total_updates} updates "
          f"(effective batch {s.micro_batch * s.grad_accum}), device {device}, amp {amp}, "
          f"checkpointing {s.checkpointing}, loss {s.loss}", flush=True)

    started = time.time()
    last_save = time.time()
    deadline = started + s.max_hours * 3600 if s.max_hours > 0 else None
    order_rng = random.Random(s.seed)
    params = [p for g in groups for p in g["params"]]
    stopped_early = False

    for epoch in range(progress["epoch"], s.epochs):
        epoch_items = list(train_items)
        random.Random(s.seed + epoch).shuffle(epoch_items)     # same order after a resume
        sigma = sigma_at(epoch, s.epochs, s.sigma_start, s.sigma_end)
        first_step = progress["step"] if epoch == progress["epoch"] else 0
        torch.manual_seed(s.seed + epoch * 100003 + first_step)
        run_loss, run_n, t_window = 0.0, 0, time.time()
        epoch_loss, epoch_n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step in range(first_step, steps_per_epoch):
            chunk = [encode_item(tok, it, s.max_len, s.head_max_len, draw_option_order(it, order_rng, ()))
                     for it in epoch_items[step * s.micro_batch:(step + 1) * s.micro_batch]]
            batch = collate_items([chunk], tok.pad_token_id)
            logits = _forward(model, batch, device, amp, s.freeze_encoder)
            mask, target = batch["marker_mask"].to(device), batch["target"].to(device)
            if s.loss == "rlcd":
                loss = rlcd_loss(logits, target, mask, batch["qtype"].to(device), sigma,
                                 s.rl_samples, s.w_sph, s.w_rps)
            else:
                loss = soft_ce_loss(logits, target, mask)
            window_start = (step // s.grad_accum) * s.grad_accum
            window = min(s.grad_accum, steps_per_epoch - window_start)
            (scaler.scale(loss / window) if scaler else loss / window).backward()

            boundary = (step + 1) % s.grad_accum == 0 or step + 1 == steps_per_epoch
            if boundary:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, s.grad_clip)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress["update"] += 1

            value = float(loss.item())
            run_loss, run_n = run_loss + value, run_n + 1
            epoch_loss, epoch_n = epoch_loss + value, epoch_n + 1
            if s.log_every and (step + 1) % s.log_every == 0:
                dt = time.time() - t_window
                per_step = dt / max(1, run_n)
                left = (steps_per_epoch - step - 1) + (s.epochs - epoch - 1) * steps_per_epoch
                print(f"epoch {epoch + 1}/{s.epochs} step {step + 1}/{steps_per_epoch} loss {run_loss / run_n:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | {per_step:.2f}s/step, ETA {left * per_step / 3600:.2f}h",
                      flush=True)
                run_loss, run_n, t_window = 0.0, 0, time.time()

            if boundary and step + 1 < steps_per_epoch:
                out_of_time = deadline is not None and time.time() > deadline
                if out_of_time or time.time() - last_save > s.save_every_min * 60:
                    progress.update(epoch=epoch, step=step + 1)
                    save_state(run_dir, model, optimizer, scheduler, scaler, progress, s.save_optimizer)
                    last_save = time.time()
                if out_of_time:
                    print(f"max_hours {s.max_hours} reached: stopped, run again with the same name to resume")
                    stopped_early = True
                    break
        if stopped_early:
            break

        # ---- epoch end: val score, keep the best epoch
        entry = {"epoch": epoch + 1, "train_loss": round(epoch_loss / max(1, epoch_n), 4)}
        if val_items:
            logits_val = score(model, tok, val_items, device, s.max_len, s.head_max_len, s.eval_batch, amp)
            raw = val_metrics(val_items, logits_val, [1.0, 1.0, 1.0], {})
            # Pick epochs on log-loss AFTER temperature scaling: an over-confident epoch that ranks
            # trades better should not lose to a timid one just because of its scale.
            fitted = fit_temperature(val_items, logits_val)
            m = val_metrics(val_items, logits_val, fitted["temperature"], fitted.get("temperature_by_options") or {})
            entry.update(val_log_loss=m["log_loss"], val_log_loss_raw=raw["log_loss"], val_auc=m["auc"],
                         val_p_spread=raw["p_spread"])
            better = progress["best_val_log_loss"] is None or m["log_loss"] < progress["best_val_log_loss"]
            auc_txt = "-" if m["auc"] is None else f"{m['auc']:.4f}"
            print(f"== epoch {epoch + 1}: train loss {entry['train_loss']:.4f} | val logloss {m['log_loss']:.4f} "
                  f"(raw {raw['log_loss']:.4f}) AUC {auc_txt} p-spread {raw['p_spread']:.3f}"
                  f"{'  (best so far)' if better else ''}", flush=True)
        else:
            better = True
            print(f"== epoch {epoch + 1}: train loss {entry['train_loss']:.4f} (no val split)", flush=True)
        if better:
            progress.update(best_val_log_loss=entry.get("val_log_loss"), best_epoch=epoch + 1)
            save_checkpoint(model, tok, dict(cfg, max_len=s.max_len, head_max_len=s.head_max_len),
                            str(run_dir / "model"))
        progress["history"].append(entry)
        progress.update(epoch=epoch + 1, step=0)
        save_state(run_dir, model, optimizer, scheduler, scaler, progress, s.save_optimizer)
        last_save = time.time()
        model.train()
        if s.freeze_encoder:
            model.encoder.eval()

    if stopped_early:
        return progress

    # ---- finish: best epoch weights, temperature fitted on val
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best, tok, cfg = load_checkpoint(str(run_dir / "model"))
    best.to(device)
    temperature, by_options = [1.0, 1.0, 1.0], {}
    if val_items:
        logits_val = score(best, tok, val_items, device, s.max_len, s.head_max_len, s.eval_batch, amp)
        fitted = fit_temperature(val_items, logits_val)
        temperature, by_options = list(fitted["temperature"]), fitted.get("temperature_by_options") or {}
    out_cfg = dict(cfg, max_len=s.max_len, head_max_len=s.head_max_len, fine_tuned=True, temperature=temperature)
    out_cfg.pop("temperature_by_options", None)
    if by_options:
        out_cfg["temperature_by_options"] = by_options
    out_cfg["training"] = dict(out_cfg.get("training") or {}, laya_finetune=asdict(s),
                               best_epoch=progress["best_epoch"], history=progress["history"])
    with open(run_dir / "model" / "rl_agent_config.json", "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, indent=2)
    progress.update(finished=True, temperature=temperature, temperature_by_options=by_options,
                    train_hours=round((time.time() - started) / 3600, 3))
    (run_dir / "state" / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    # The resume weights are no longer needed once finished (1.7 GB); progress.json stays.
    for name in ("model.safetensors", "trainer.pt"):
        try:
            os.remove(run_dir / "state" / name)
        except OSError:
            pass
    print(f"best epoch {progress['best_epoch']}, temperature {[round(t, 3) for t in temperature]}", flush=True)
    del best
    return progress


def predict_probs(model_dir: str, rows: List[dict], device_name: str, batch_size: int = 16) -> Dict[str, list]:
    """{question id: [option probabilities (canonical order) or None, one per row]}, with the
    checkpoint's temperature. A row laya cannot encode gets None."""
    device = torch.device(device_name)
    model, tok, cfg = load_checkpoint(model_dir)
    model.to(device).eval()
    max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    out: Dict[str, list] = {}
    qids = sorted({q for r in rows for q in r.get("questions", {})})
    for qid in qids:
        single = [{"state": r["state"], "questions": {qid: r["questions"][qid]},
                   "expected": {qid: r["expected"][qid]}} if qid in r.get("questions", {}) else
                  {"state": "", "questions": {}} for r in rows]
        items, index = build_items(tok, single, max_len, head_max_len)
        logits = score(model, tok, items, device, max_len, head_max_len, batch_size, amp=True)
        probs = item_probs(items, logits, cfg.get("temperature") or [1.0, 1.0, 1.0],
                           cfg.get("temperature_by_options") or {})
        column: list = [None] * len(rows)
        for i, p in zip(index, probs):
            column[i] = p
        out[qid] = column
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return out


def predict_file(model_dir: str, rows: List[dict], device_name: str, batch_size: int = 16) -> np.ndarray:
    """p(win) for each row (NaN for a row laya cannot encode), with the checkpoint's temperature."""
    device = torch.device(device_name)
    model, tok, cfg = load_checkpoint(model_dir)
    model.to(device).eval()
    max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
    items, index = build_items(tok, rows, max_len, head_max_len)
    logits = score(model, tok, items, device, max_len, head_max_len, batch_size, amp=True)
    p = np.full(len(rows), np.nan)
    p[index] = win_probability(items, logits, cfg.get("temperature") or [1.0, 1.0, 1.0],
                               cfg.get("temperature_by_options") or {})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return p
