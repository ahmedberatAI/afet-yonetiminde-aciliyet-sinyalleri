#!/usr/bin/env python3
"""
Train a multi-label need classifier (Turkish) using HuggingFace Transformers.

This repo prepares datasets/splits; training is intentionally NOT run by default.
Use this script when you're ready to start modeling.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


DEFAULT_CONFIG = "data/modeling/need_classification/training_config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        raise SystemExit("Missing dependency: pyyaml. Install it before running training.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Config YAML must parse to a dict.")
    return data


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically-stable sigmoid for logits.
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def _coerce_binary_0_1(df: pd.DataFrame, col: str) -> None:
    s = df[col].astype("string").fillna("").str.strip()
    s = s.replace({"": "0"})
    df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    if not df[col].isin([0, 1]).all():
        bad = df.loc[~df[col].isin([0, 1]), col].head(5).tolist()
        raise SystemExit(f"Invalid values in {col}: expected 0/1 only. Examples: {bad}")


def _load_xy(csv_path: Path, text_col: str, label_cols: List[str]) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in ([text_col] + label_cols) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {missing}")

    for c in label_cols:
        _coerce_binary_0_1(df, c)

    texts = df[text_col].astype("string").fillna("").tolist()
    y = df[label_cols].astype(int).to_numpy(dtype=np.int64)
    return texts, y


def _format_label_report(y: np.ndarray, label_cols: List[str]) -> Dict[str, int]:
    return {label_cols[i]: int(y[:, i].sum()) for i in range(len(label_cols))}


def _make_compute_metrics(threshold: float, label_cols: List[str]):
    def compute_metrics(eval_pred):
        try:
            logits = eval_pred.predictions
            y_true = eval_pred.label_ids
        except AttributeError:
            logits, y_true = eval_pred

        logits = np.asarray(logits)
        y_true = np.asarray(y_true).astype(int)
        probs = _sigmoid(logits)
        y_pred = (probs >= threshold).astype(int)

        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
        except ModuleNotFoundError:
            raise SystemExit("Missing dependency: scikit-learn. Install it before running training.")

        metrics: Dict[str, float] = {}
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))

        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, name in enumerate(label_cols):
            metrics[f"f1_{name}"] = float(per_label_f1[i])
        return metrics

    return compute_metrics


@dataclass
class _TorchDataset:
    encodings: Dict[str, List[List[int]]]
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch

        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def main() -> int:
    p = argparse.ArgumentParser(description="Train multi-label need classifier (HuggingFace Transformers).")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to training_config.yaml")
    p.add_argument("--output-dir", default="models/need_classification", help="Where to write model checkpoints.")
    p.add_argument("--threshold", type=float, default=None, help="Prediction threshold for metrics (overrides config).")
    p.add_argument(
        "--use-pos-weight",
        action="store_true",
        help="Use per-label pos_weight in BCEWithLogitsLoss to mitigate class imbalance.",
    )
    p.add_argument("--pos-weight-min", type=float, default=1.0, help="Lower bound for pos_weight (default: 1.0).")
    p.add_argument("--pos-weight-max", type=float, default=50.0, help="Upper bound for pos_weight (default: 50.0).")
    p.add_argument("--prep-only", action="store_true", help="Only validate data/config, do not train.")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)

    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    train_csv = Path(dataset_cfg.get("train_csv", ""))
    val_csv = Path(dataset_cfg.get("val_csv", ""))
    test_csv = Path(dataset_cfg.get("test_csv", ""))
    text_col = str(dataset_cfg.get("text_column", "tweet_clean"))

    label_cols = dataset_cfg.get("label_columns", NEED_LABEL_COLS)
    if not isinstance(label_cols, list) or not label_cols:
        raise SystemExit("dataset.label_columns must be a non-empty list.")
    label_cols = [str(x) for x in label_cols]

    base_model = str(model_cfg.get("base_model", "dbmdz/bert-base-turkish-cased"))
    max_length = int(model_cfg.get("max_length", 192))

    seed = int(train_cfg.get("seed", 42))
    epochs = float(train_cfg.get("epochs", 3))
    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.06))
    train_bs = int(train_cfg.get("train_batch_size", 16))
    eval_bs = int(train_cfg.get("eval_batch_size", 32))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    fp16 = bool(train_cfg.get("fp16", True))
    use_pos_weight = bool(train_cfg.get("use_pos_weight", False)) or bool(args.use_pos_weight)
    pos_weight_min = float(train_cfg.get("pos_weight_min", args.pos_weight_min))
    pos_weight_max = float(train_cfg.get("pos_weight_max", args.pos_weight_max))

    threshold = float(args.threshold) if args.threshold is not None else float(eval_cfg.get("threshold", 0.5))

    # Validate data (always).
    x_train, y_train = _load_xy(train_csv, text_col=text_col, label_cols=label_cols)
    x_val, y_val = _load_xy(val_csv, text_col=text_col, label_cols=label_cols)
    x_test, y_test = _load_xy(test_csv, text_col=text_col, label_cols=label_cols)

    print("DATA CHECKS")
    print(f"- train rows: {len(x_train)} | label positives: {_format_label_report(y_train, label_cols)}")
    print(f"- val rows:   {len(x_val)} | label positives: {_format_label_report(y_val, label_cols)}")
    print(f"- test rows:  {len(x_test)} | label positives: {_format_label_report(y_test, label_cols)}")
    print(f"- labels: {label_cols}")
    print(f"- threshold: {threshold}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "label_columns.json").write_text(json.dumps(label_cols, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.prep_only:
        print(f"Prep-only mode: wrote label_columns.json to {outdir.as_posix()}")
        return 0

    # Lazy imports so `--help` and `--prep-only` work without heavy deps.
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ModuleNotFoundError as e:
        raise SystemExit(f"Missing dependency for training: {e}. Install torch + transformers.")

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    enc_train = tokenizer(x_train, truncation=True, max_length=max_length)
    enc_val = tokenizer(x_val, truncation=True, max_length=max_length)

    ds_train = _TorchDataset(encodings=enc_train, labels=y_train)
    ds_val = _TorchDataset(encodings=enc_val, labels=y_val)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Optional: compute per-label pos_weight from training labels.
    pos_weight = None
    if use_pos_weight:
        n = int(y_train.shape[0])
        pos = y_train.sum(axis=0).astype(np.float32)
        neg = float(n) - pos
        # Standard heuristic: pos_weight = neg/pos (clipped). Do not down-weight positives (<1).
        weights = np.ones_like(pos, dtype=np.float32)
        for i in range(len(pos)):
            if pos[i] <= 0:
                weights[i] = 1.0
            else:
                weights[i] = float(neg[i] / pos[i])
        weights = np.clip(weights, float(pos_weight_min), float(pos_weight_max))
        weights = np.maximum(weights, 1.0)
        pos_weight = torch.tensor(weights, dtype=torch.float32)
        (outdir / "pos_weight.json").write_text(
            json.dumps({label_cols[i]: float(weights[i]) for i in range(len(label_cols))}, indent=2, ensure_ascii=True)
            + "\n",
            encoding="utf-8",
        )
        print("POS_WEIGHT (clipped)")
        print({label_cols[i]: float(weights[i]) for i in range(len(label_cols))})

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label_cols),
        problem_type="multi_label_classification",
    )

    steps_per_epoch = math.ceil(len(ds_train) / (train_bs * max(1, grad_accum)))
    warmup_steps = int(warmup_ratio * steps_per_epoch * epochs)

    args_tr = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=grad_accum,
        fp16=fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 5),
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    # Custom trainer to support class-imbalance weighting (pos_weight).
    class _WeightedTrainer(Trainer):
        def __init__(self, *t_args, pos_weight=None, **t_kwargs):
            super().__init__(*t_args, **t_kwargs)
            self._pos_weight = pos_weight

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            if self._pos_weight is not None:
                pw = self._pos_weight.to(logits.device)
                loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer_cls = _WeightedTrainer if pos_weight is not None else Trainer

    callbacks = []
    es_cfg = train_cfg.get("early_stopping", {}) if isinstance(train_cfg.get("early_stopping", {}), dict) else {}
    if bool(es_cfg.get("enabled", False)):
        patience = int(es_cfg.get("patience", 2))
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer_kwargs = dict(
        model=model,
        args=args_tr,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collator,
        compute_metrics=_make_compute_metrics(threshold=threshold, label_cols=label_cols),
        callbacks=callbacks,
    )
    extra_kwargs = {}
    if trainer_cls is _WeightedTrainer:
        extra_kwargs["pos_weight"] = pos_weight

    # Transformers Trainer API changed across versions:
    # - older: `tokenizer=...`
    # - newer: `processing_class=...` (and `tokenizer` removed)
    try:
        trainer = trainer_cls(**trainer_kwargs, processing_class=tokenizer, **extra_kwargs)
    except TypeError as e:
        if "processing_class" not in str(e):
            raise
        try:
            trainer = trainer_cls(**trainer_kwargs, tokenizer=tokenizer, **extra_kwargs)
        except TypeError as e2:
            if "tokenizer" not in str(e2):
                raise
            trainer = trainer_cls(**trainer_kwargs, **extra_kwargs)

    print("TRAINING")
    trainer.train()

    print("EVALUATION (VAL)")
    val_metrics = trainer.evaluate()
    (outdir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote: {(outdir / 'val_metrics.json').as_posix()}")

    # Optional: test evaluation (loads best model already).
    print("EVALUATION (TEST)")
    enc_test = tokenizer(x_test, truncation=True, max_length=max_length)
    ds_test = _TorchDataset(encodings=enc_test, labels=y_test)
    test_metrics = trainer.evaluate(eval_dataset=ds_test)
    (outdir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote: {(outdir / 'test_metrics.json').as_posix()}")

    trainer.save_model(str(outdir / "final"))
    tokenizer.save_pretrained(str(outdir / "final"))
    print(f"Wrote: {(outdir / 'final').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
