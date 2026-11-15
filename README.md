# laya_finetune

Fine-tune [Laya](https://github.com/NandhaKishorM/laya) (ModernBERT-large encoder + decision head, 421M) to
predict **win / loss** of a backtest trade from the AI-review text that Bot_Trade's
`src/evaluate/llm_dataset.py` writes, then score the test split.

```
train.py           CLI: prepare data -> train (resumable) -> score test -> report
trainer.py         training loop on laya's pieces: fp16, resume, periodic saves, best epoch on val
dataset_prep.py    load dataset.jsonl files, one global time split, text transforms, Laya rows
metrics.py         log-loss / Brier / AUC (overall + per coin), R if filtered, tabular baseline
colab_train.ipynb  Colab notebook (T4)
scripts/
  build_dataset.sh   Mac: build the datasets from Bot_Trade backtests into data/
  pack_data.sh       Mac: data/ -> laya_data.tar.gz (upload it to Google Drive)
  smoke_mac.sh       Mac: 2-minute flow check
  colab_setup.sh     Colab: install deps, check GPU
  colab_train.sh     Colab: unpack data from Drive, train / resume, report
data/  runs/  models/  (gitignored)
```

## 1. Data (Mac, inside Bot_Trade)

```bash
bash laya_finetune/scripts/build_dataset.sh                  # every coin not built yet (res/backtest/200k)
bash laya_finetune/scripts/build_dataset.sh BTCUSDT ETHUSDT  # only these
BT=res/backtest/100k bash laya_finetune/scripts/build_dataset.sh
```

Text only (the model never reads the chart image), no split: `train.py` makes one time split over all coins.
Re-running continues where it stopped (~0.1-0.2 s per trade, ~1650 trades per coin).

## 2. Flow check on the Mac

```bash
.venv/bin/pip install -r laya_finetune/requirements.txt     # once, into the Bot_Trade venv
bash laya_finetune/scripts/smoke_mac.sh
```

## 3. Train on Colab (T4)

```bash
bash laya_finetune/scripts/pack_data.sh    # -> laya_finetune/laya_data.tar.gz (~5 MB per coin)
```

Upload `laya_data.tar.gz` to Google Drive as `MyDrive/laya/laya_data.tar.gz`, open `colab_train.ipynb` in Colab
(File -> Open notebook -> GitHub -> cavaldos/laya_finetune), pick a T4 runtime, run all cells.

Or by hand in a Colab terminal / cell:

```bash
git clone https://github.com/cavaldos/laya_finetune.git /content/laya_finetune && cd /content/laya_finetune
bash scripts/colab_setup.sh
RUN_NAME=run1 bash scripts/colab_train.sh                 # profile "colab": all trades, 3 epochs, 2048 tokens
RUN_NAME=anon1 bash scripts/colab_train.sh --anonymize    # extra args go to train.py
```

- Runs go to `MyDrive/laya/runs/<RUN_NAME>/`. A resume state is saved every 20 minutes and at each epoch end;
  after a disconnect run the same command (same `RUN_NAME`) and it continues from there.
- `MAX_HOURS=3.5` stops cleanly (state saved) before a session limit.
- Out of GPU memory: `--micro-batch 2 --grad-accum 16` (same effective batch), or `--max-len 1024`.
- Speed: roughly 0.3-0.5 s per trade per epoch at 2048 tokens on a T4; the log prints s/step and an ETA.

## Split and selection

All coins share one timeline: oldest 70% train, next 10% val, newest 20% test (`--val-frac`, `--test-frac`).
A trade still open when the next split starts is dropped (embargo). After each epoch the model is scored on val;
the epoch with the lowest val log-loss (after temperature scaling) is kept and its temperature is fitted on val.
The test split is used once, at the end.

## Options (`python train.py -h`)

| option | meaning |
|---|---|
| `--sections compact\|facts\|all\|"PLAN,KEY FACTS"` | `##` sections of the text. `compact` = PLAN, KEY FACTS, SETUP, TREND, STRUCTURE (~1.5k tokens); `all` ~6-7k |
| `--max-len N` | tokens per sequence; a longer text is cut **at the end** (the run prints how many) |
| `--anonymize` | coin name -> `COIN`, prices -> `[+1.25R]` from the entry, distances -> % of price |
| `--coin-group` | coin name -> `major` / `large-cap alt` / `small-cap alt` (use with `--anonymize`) |
| `--symbols BTC,ETH` | only these coins |
| `--epochs`, `--micro-batch`, `--grad-accum`, `--encoder-lr`, `--head-lr` | training |
| `--loss rlcd\|soft-ce` | laya's objective (default) or plain cross-entropy |
| `--freeze-encoder` | train only the head: fast, little memory |
| `--limit-train/--limit-val/--limit-test N` | random subsets (quick tries) |
| `--zero-shot` | also score the untrained checkpoint |
| `--prepare-only` | only write the data + token report |
| `--eval-only RUN_DIR` | score the test split of a finished run again |

## Output (`runs/<name>/`)

- `data/` train / val / test rows exactly as the model saw them + trade metadata
- `model/` best checkpoint (`laya.load("runs/<name>/model")` works)
- `test_report.txt` / `.json`, `predictions.csv` (p(win) of every test trade for every model)
- `state/progress.json` loss and val metrics per epoch; `run_config.json` options + data stats

## Reading the report

Same test trades for every row:

- `base_rate`: always the train win rate. A model that does not beat its log-loss / Brier learned nothing.
- `logistic_tabular`: logistic regression on side, TF, zone, SL/TP source, RR, hour (no text).
- `laya_zero_shot` (with `--zero-shot`): the checkpoint before fine-tuning.
- `laya_finetuned`: the result.

`AUC/coin` is the AUC inside each coin, averaged. With coins mixed, the overall AUC can rise just by knowing
which coin had the better win rate in that period; `AUC/coin` cannot. The last block is gross R (fees not
included) if only the trades the model likes were taken, e.g. `p(win) > 1/(1+RR)` (break-even win rate at that RR).

Coin name or not: train once plain and once with `--anonymize`, compare `AUC/coin`. Equal -> keep the
anonymized model. Plain better only in overall AUC -> it learned per-coin win rates; reject it.

## Price direction task (up / down + how far)

Instead of win / loss of one SMC trade, predict the market: from the price path + indicators up to a
moment, is the price higher or lower `horizon` candles later, and by how much. Data from Bot_Trade's
`src/evaluate/price_dataset.py` (text: indicators of the chart TF + 1h + 4h, SMC structure, candles, day
context, every price as % from the current price, no coin name; ~1.75k tokens, fits `--max-len 2048`;
`--mode image` / `both` also writes chart PNGs for an image model).

```bash
bash laya_finetune/scripts/build_price_dataset.sh                 # Mac: every coin -> data_price/<coin>/
bash laya_finetune/scripts/build_price_dataset.sh BTCUSDT ETHUSDT # only these
python train.py --data data_price --profile smoke                 # Mac flow check (~2 min)
DATA_DIR=data_price RUN_NAME=dir1 bash scripts/colab_train.sh     # Kaggle / Colab
```

Default rows: a 15m chart every 3h, label after 12 candles (3h), so two rows never share a result.
Each row asks Laya two questions:
- `direction` (choice up / down) -> p(up);
- `move` (score, 5 ordered levels; edges = train quantiles of the move in ATR of the chart TF,
  `--move-unit pct` for raw %) -> expected move in % = sum of p(level) x level median x ATR %.

`--horizon h4` / `h24` relabels with another stored horizon; `--questions direction` trains only the
direction question (half the items, half the time).

The report compares Laya with `base_rate` and `logistic_features` (logistic regression on the
indicator numbers of the same text). Laya is only worth something if it beats `logistic_features` on
log-loss / AUC/coin. The trading block is gross % per trade (fees ~0.08-0.12% per round trip not taken off).
