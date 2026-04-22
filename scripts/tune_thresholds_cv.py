#!/usr/bin/env python3
"""
Cross-validated per-label threshold tuning for the multi-label need classifier.

Motivation
----------
The default `scripts/tune_thresholds.py` reads a single `val.csv` and picks a
threshold per label. When some labels have only 0-1 positives in that split
(psikolojik, altyapi in the current gold), the best threshold is pure noise.

This script instead:
1. Concatenates train + val into a "tuning pool" (test is never touched).
2. Runs multilabel-stratified K-fold (iterative stratification) so every fold
   keeps each rare label approximately balanced.
3. For each fold: fine-tunes BERTurk on the other K-1 folds, predicts on the
   held-out fold, and concatenates those predictions into a full out-of-fold
   (OOF) probability matrix for the entire pool.
4. Searches a threshold grid per label on the OOF probs using a *smoothed* F1
   (F1 with a small Laplace-style pseudocount) so rare labels do not overfit
   to a single example.
5. Also records the per-fold best thresholds and reports the median as a
   robustness check.

Outputs
-------
- thresholds JSON (mean across folds is deliberately NOT used; OOF-global is
  written by default; see --strategy).
- CV report text with per-label: OOF positives, OOF best_thr, smoothed-F1,
  fold-wise best_thr list, median.
- Optional metadata JSON with the exact split sources, label counts, fold
  distributions, threshold strategy, and CUDA device used.

Why smoothed F1
---------------
F1 = 2*TP / (2*TP + FP + FN). With 1-2 positives, a single TP can make F1
jump to 1.0 and a single FN can drop it to 0. We use
    F1_s = (2*TP + alpha) / (2*TP + FP + FN + alpha)
with alpha=1 (configurable) to damp this zero/one behavior and pick a more
defensible threshold.

Run location
------------
This script *does* fine-tune BERT K times (default K=5). Expect roughly
K * single-training cost. Use Colab with a GPU. For a small K=3 quick pass,
see --k 3.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NEED_LABEL_COLS: List[str] = [
    "arama_kurtarma",
    "saglik",
    "barinma",
    "gida_su",
    "altyapi",
    "guvenlik",
    "lojistik",
    "psikolojik",
    "bilgi_paylasimi",
]

TEXT_COL = "tweet_clean"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _coerce_binary_0_1(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip().replace({"": "0"})
    df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    if not df[col].isin([0, 1]).all():
        raise SystemExit(f"Invalid values in {col}: expected 0/1 only.")


def _load_concat(train_csv: Path, val_csv: Path, label_cols: List[str]) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    dfs = []
    for p in (train_csv, val_csv):
        df = pd.read_csv(p, encoding="utf-8-sig")
        missing = [c for c in [TEXT_COL] + label_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Missing columns in {p}: {missing}")
        for c in label_cols:
            _coerce_binary_0_1(df, c)
        df["__src__"] = p.name
        dfs.append(df)
    pool = pd.concat(dfs, ignore_index=True)
    texts = pool[TEXT_COL].astype("string").fillna("").tolist()
    y = pool[label_cols].astype(int).to_numpy(dtype=np.int64)
    return texts, y, pool


def _iterative_stratify(y: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    """
    Sechidis, Tsoumakas & Vlahavas (2011) - "On the Stratification of Multi-label Data".

    Assigns each sample to one of K folds so that each label's positive count
    is as close to uniform across folds as possible. Handles all-zero rows by
    routing them to the most-needed fold for the "none" pseudo-label.
    """
    n, L = y.shape
    rng = np.random.RandomState(seed)

    # Target number of positives per fold per label.
    pos_total = y.sum(axis=0).astype(float)
    desired_per_fold = np.tile(pos_total / k, (k, 1))  # shape (k, L)

    # Track remaining target per fold per label.
    remaining = desired_per_fold.copy()

    # Fold size targets.
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    remaining_slots = fold_sizes.copy()

    assignments = np.full(n, -1, dtype=int)

    # Process labels from rarest to most common (by total positive count).
    label_order = np.argsort(pos_total)  # rarest first
    # Index set of unassigned rows.
    unassigned = set(range(n))

    for lab in label_order:
        pos_idx = [i for i in unassigned if y[i, lab] == 1]
        rng.shuffle(pos_idx)
        for i in pos_idx:
            # Pick the fold with the largest remaining need for this label,
            # breaking ties by largest remaining slot, then random.
            cand_scores = remaining[:, lab].copy()
            valid = remaining_slots > 0
            if not valid.any():
                # Should not happen, but safe-guard.
                valid = np.ones(k, dtype=bool)
            cand_scores[~valid] = -np.inf

            max_val = cand_scores.max()
            winners = np.where(cand_scores == max_val)[0]
            if len(winners) > 1:
                slot_tiebreak = remaining_slots[winners]
                max_slot = slot_tiebreak.max()
                winners = winners[slot_tiebreak == max_slot]
            fold = int(rng.choice(winners))

            assignments[i] = fold
            remaining[fold, lab] -= 1
            remaining_slots[fold] -= 1
            unassigned.discard(i)

    # Remaining rows (all-zero label rows): distribute greedily by slot.
    leftovers = list(unassigned)
    rng.shuffle(leftovers)
    for i in leftovers:
        order = np.argsort(-remaining_slots)
        for fold in order:
            if remaining_slots[fold] > 0:
                assignments[i] = fold
                remaining_slots[fold] -= 1
                break

    if (assignments < 0).any():
        raise RuntimeError("Iterative stratification failed to assign some rows.")

    # Build per-fold index arrays.
    folds = [np.where(assignments == k_i)[0] for k_i in range(k)]
    return folds


def _smoothed_f1(tp: int, fp: int, fn: int, alpha: float) -> float:
    denom = 2 * tp + fp + fn + alpha
    if denom <= 0:
        return 0.0
    return (2 * tp + alpha) / denom


def _grid_search(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray, alpha: float) -> Tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    for thr in grid:
        yp = (probs >= float(thr)).astype(np.int64)
        tp = int(((y_true == 1) & (yp == 1)).sum())
        fp = int(((y_true == 0) & (yp == 1)).sum())
        fn = int(((y_true == 1) & (yp == 0)).sum())
        score = _smoothed_f1(tp, fp, fn, alpha)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def _get_runtime_device(*, require_cuda: bool) -> Dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise SystemExit("CUDA is required for this run, but torch.cuda.is_available() is False.")

    device_info: Dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        device_info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        major, minor = torch.cuda.get_device_capability(0)
        device_info["cuda_capability"] = f"{major}.{minor}"
    return device_info


def _train_fold(
    *,
    base_model: str,
    train_texts: List[str],
    train_y: np.ndarray,
    holdout_texts: List[str],
    holdout_y: np.ndarray,
    label_cols: List[str],
    seed: int,
    epochs: float,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    train_bs: int,
    eval_bs: int,
    max_length: int,
    use_pos_weight: bool,
    pos_weight_min: float,
    pos_weight_max: float,
    tmp_outdir: Path,
    fp16: bool,
) -> np.ndarray:
    """Train a fresh model on train split, return probs on holdout split."""

    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    enc_train = tokenizer(train_texts, truncation=True, max_length=max_length)
    enc_hold = tokenizer(holdout_texts, truncation=True, max_length=max_length)

    class _DS:
        def __init__(self, enc: Dict[str, List[List[int]]], y: np.ndarray):
            self.enc = enc
            self.y = y

        def __len__(self) -> int:
            return int(self.y.shape[0])

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[idx], dtype=torch.float32)
            return item

    ds_train = _DS(enc_train, train_y)
    ds_hold = _DS(enc_hold, holdout_y)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    pos_weight = None
    if use_pos_weight:
        n = int(train_y.shape[0])
        pos = train_y.sum(axis=0).astype(np.float32)
        neg = float(n) - pos
        weights = np.ones_like(pos, dtype=np.float32)
        for i in range(len(pos)):
            weights[i] = 1.0 if pos[i] <= 0 else float(neg[i] / pos[i])
        weights = np.clip(weights, pos_weight_min, pos_weight_max)
        weights = np.maximum(weights, 1.0)
        pos_weight = torch.tensor(weights, dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
    )

    steps_per_epoch = max(1, math.ceil(len(ds_train) / train_bs))
    warmup_steps = int(warmup_ratio * steps_per_epoch * epochs)

    tmp_outdir.mkdir(parents=True, exist_ok=True)
    tr_args = TrainingArguments(
        output_dir=str(tmp_outdir),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        fp16=fp16 and torch.cuda.is_available(),
        disable_tqdm=True,
        report_to=[],
        seed=seed,
    )

    class _WeightedTrainer(Trainer):
        def __init__(self, *a, pw=None, **kw):
            super().__init__(*a, **kw)
            self._pw = pw

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if self._pw is not None:
                pw = self._pw.to(logits.device)
                loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    trainer_kwargs = dict(
        model=model,
        args=tr_args,
        train_dataset=ds_train,
        eval_dataset=ds_hold,
        data_collator=collator,
        pw=pos_weight,
    )
    try:
        trainer = _WeightedTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError as e:
        if "processing_class" not in str(e):
            raise
        try:
            trainer = _WeightedTrainer(**trainer_kwargs, tokenizer=tokenizer)
        except TypeError as e2:
            if "tokenizer" not in str(e2):
                raise
            trainer = _WeightedTrainer(**trainer_kwargs)
    trainer.train()
    preds = trainer.predict(ds_hold)
    logits = np.asarray(preds.predictions, dtype=np.float32)
    probs = _sigmoid(logits)

    # Clean up checkpoint scratch
    try:
        shutil.rmtree(tmp_outdir, ignore_errors=True)
    except Exception:
        pass
    return probs


def main() -> int:
    p = argparse.ArgumentParser(description="Cross-validated per-label threshold tuning (multilabel-stratified).")
    p.add_argument("--train-csv", default="data/modeling/need_classification_gold_combined/train.csv")
    p.add_argument("--val-csv", default="data/modeling/need_classification_gold_combined/val.csv")
    p.add_argument("--labels-json", default=None, help="If omitted, uses default need label list.")
    p.add_argument("--k", type=int, default=5, help="Number of folds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-model", default="dbmdz/bert-base-turkish-cased")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--train-bs", type=int, default=16)
    p.add_argument("--eval-bs", type=int, default=32)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use-pos-weight", action="store_true")
    p.add_argument("--pos-weight-min", type=float, default=1.0)
    p.add_argument("--pos-weight-max", type=float, default=50.0)
    p.add_argument("--min-thr", type=float, default=0.05)
    p.add_argument("--max-thr", type=float, default=0.95)
    p.add_argument("--step", type=float, default=0.01)
    p.add_argument("--smoothing-alpha", type=float, default=1.0,
                   help="Laplace-style pseudocount added to smoothed F1 numerator and denominator.")
    p.add_argument("--strategy", choices=["oof_global", "fold_median"], default="oof_global",
                   help="How to pick the final threshold: search on concatenated OOF probs, or take median of fold-wise best thresholds.")
    p.add_argument("--out-thresholds", default=None, help="Output thresholds JSON path (required unless --dry-run).")
    p.add_argument("--out-report", default=None, help="Output report text path (required unless --dry-run).")
    p.add_argument("--out-meta", default=None, help="Optional metadata JSON path with tuning provenance.")
    p.add_argument("--out-oof", default=None, help="Optional CSV path for OOF probs (for later calibration).")
    p.add_argument("--scratch-dir", default="models/_cv_tuning_scratch")
    p.add_argument("--require-cuda", action="store_true", help="Fail fast if CUDA is not available.")
    p.add_argument("--dry-run", action="store_true", help="Only stratify folds and print label distributions, skip training.")
    args = p.parse_args()

    label_cols = NEED_LABEL_COLS
    if args.labels_json:
        label_cols = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))
        if not isinstance(label_cols, list) or not label_cols:
            raise SystemExit("label_columns.json must be a non-empty list.")
        label_cols = [str(x) for x in label_cols]

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    scratch = Path(args.scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)

    runtime_device = _get_runtime_device(require_cuda=bool(args.require_cuda))
    texts, y, pool = _load_concat(train_csv, val_csv, label_cols)
    n, L = y.shape
    pool_pos = {label_cols[i]: int(y[:, i].sum()) for i in range(L)}
    print(f"Runtime device: {runtime_device}")
    print(f"Pool rows: {n} (train+val).")
    print(f"Label positives (pool): {pool_pos}")

    folds = _iterative_stratify(y, k=int(args.k), seed=int(args.seed))
    print(f"Fold sizes: {[len(f) for f in folds]}")

    # Per-label positives per fold for the report.
    fold_pos_per_label: List[Dict[str, int]] = []
    for fi, idx in enumerate(folds):
        per = {label_cols[j]: int(y[idx, j].sum()) for j in range(L)}
        fold_pos_per_label.append(per)
        print(f"  fold {fi}: rows={len(idx)} positives={per}")

    if args.dry_run:
        print("--dry-run: skipping training/evaluation.")
        return 0

    if not args.out_thresholds or not args.out_report:
        raise SystemExit("--out-thresholds and --out-report are required (unless --dry-run).")

    oof_probs = np.zeros((n, L), dtype=np.float32)
    fold_best_thr: List[Dict[str, float]] = []

    thr_grid = np.arange(args.min_thr, args.max_thr + 1e-9, args.step, dtype=np.float32)
    thr_grid = np.clip(thr_grid, 0.0, 1.0)

    for fi in range(int(args.k)):
        hold_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(int(args.k)) if j != fi])
        train_texts = [texts[i] for i in train_idx]
        hold_texts = [texts[i] for i in hold_idx]
        train_y = y[train_idx]
        hold_y = y[hold_idx]
        print(f"\n=== Fold {fi+1}/{args.k}: train={len(train_idx)} hold={len(hold_idx)} ===")

        fold_scratch = scratch / f"fold_{fi}"
        probs = _train_fold(
            base_model=args.base_model,
            train_texts=train_texts,
            train_y=train_y,
            holdout_texts=hold_texts,
            holdout_y=hold_y,
            label_cols=label_cols,
            seed=int(args.seed) + fi,
            epochs=float(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            warmup_ratio=float(args.warmup_ratio),
            train_bs=int(args.train_bs),
            eval_bs=int(args.eval_bs),
            max_length=int(args.max_length),
            use_pos_weight=bool(args.use_pos_weight),
            pos_weight_min=float(args.pos_weight_min),
            pos_weight_max=float(args.pos_weight_max),
            tmp_outdir=fold_scratch,
            fp16=bool(args.fp16),
        )
        oof_probs[hold_idx] = probs

        # Per-fold per-label best threshold (for median robustness check).
        fold_map: Dict[str, float] = {}
        for j, lab in enumerate(label_cols):
            best_thr, _ = _grid_search(hold_y[:, j], probs[:, j], thr_grid, float(args.smoothing_alpha))
            fold_map[lab] = best_thr
        fold_best_thr.append(fold_map)
        print(f"  fold {fi} best_thr: {fold_map}")

    # Global OOF threshold search per label.
    global_best: Dict[str, Dict[str, float]] = {}
    thresholds_out: Dict[str, float] = {}
    for j, lab in enumerate(label_cols):
        yt = y[:, j]
        ps = oof_probs[:, j]
        best_thr, best_score = _grid_search(yt, ps, thr_grid, float(args.smoothing_alpha))
        folds_thrs = [fold_best_thr[fi][lab] for fi in range(int(args.k))]
        median_thr = float(np.median(folds_thrs))
        global_best[lab] = {
            "oof_pos": int(yt.sum()),
            "oof_best_thr": float(best_thr),
            "oof_smoothed_f1": float(best_score),
            "fold_best_thr": [float(t) for t in folds_thrs],
            "fold_median_thr": float(median_thr),
        }
        if args.strategy == "fold_median":
            thresholds_out[lab] = median_thr
        else:
            thresholds_out[lab] = best_thr

    out_thr = Path(args.out_thresholds)
    out_thr.parent.mkdir(parents=True, exist_ok=True)
    out_thr.write_text(json.dumps(thresholds_out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nWrote thresholds: {out_thr}")

    # Report
    out_rep = Path(args.out_report)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("CROSS-VALIDATED THRESHOLD TUNING REPORT\n")
    lines.append(f"train_csv: {train_csv.as_posix()}")
    lines.append(f"val_csv:   {val_csv.as_posix()}")
    lines.append(f"base_model: {args.base_model}")
    lines.append(f"use_pos_weight: {bool(args.use_pos_weight)}")
    lines.append(f"require_cuda: {bool(args.require_cuda)}")
    lines.append(f"runtime_device: {runtime_device}")
    lines.append(f"k: {args.k} | seed: {args.seed} | strategy: {args.strategy}")
    lines.append(f"smoothing_alpha: {args.smoothing_alpha}")
    lines.append(f"grid: [{args.min_thr}, {args.max_thr}] step={args.step}")
    lines.append(f"pool rows: {n}")
    lines.append(f"pool label positives: {pool_pos}")
    lines.append(f"fold sizes: {[len(f) for f in folds]}")
    lines.append("fold label positives:")
    for fi, per in enumerate(fold_pos_per_label):
        lines.append(f"  - fold_{fi}: {per}")
    lines.append("")
    lines.append("PER-LABEL RESULTS")
    for lab in label_cols:
        info = global_best[lab]
        lines.append(
            f"- {lab}: oof_pos={info['oof_pos']} "
            f"oof_best_thr={info['oof_best_thr']:.3f} "
            f"smoothed_f1={info['oof_smoothed_f1']:.3f} "
            f"fold_median_thr={info['fold_median_thr']:.3f} "
            f"fold_best_thr={[f'{t:.3f}' for t in info['fold_best_thr']]}"
        )
    out_rep.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote report: {out_rep}")

    if args.out_meta:
        out_meta = Path(args.out_meta)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "train_csv": str(train_csv.as_posix()),
            "val_csv": str(val_csv.as_posix()),
            "base_model": str(args.base_model),
            "labels": label_cols,
            "k": int(args.k),
            "seed": int(args.seed),
            "strategy": str(args.strategy),
            "smoothing_alpha": float(args.smoothing_alpha),
            "grid": {
                "min_thr": float(args.min_thr),
                "max_thr": float(args.max_thr),
                "step": float(args.step),
                "count": int(len(thr_grid)),
            },
            "use_pos_weight": bool(args.use_pos_weight),
            "pos_weight_min": float(args.pos_weight_min),
            "pos_weight_max": float(args.pos_weight_max),
            "require_cuda": bool(args.require_cuda),
            "runtime_device": runtime_device,
            "pool_rows": int(n),
            "pool_label_positives": pool_pos,
            "fold_sizes": [int(len(f)) for f in folds],
            "fold_label_positives": fold_pos_per_label,
            "thresholds_selected": thresholds_out,
            "per_label_results": global_best,
        }
        out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote meta: {out_meta}")

    if args.out_oof:
        out_oof = Path(args.out_oof)
        out_oof.parent.mkdir(parents=True, exist_ok=True)
        oof_df = pool[["id"] if "id" in pool.columns else []].copy()
        for j, lab in enumerate(label_cols):
            oof_df[f"prob_{lab}"] = oof_probs[:, j]
            oof_df[f"y_{lab}"] = y[:, j]
        oof_df.to_csv(out_oof, index=False, encoding="utf-8-sig")
        print(f"Wrote OOF probs: {out_oof}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
