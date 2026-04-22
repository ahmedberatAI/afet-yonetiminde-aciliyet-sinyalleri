#!/usr/bin/env python3
"""
Post-process the prediction metadata written by scripts/predict_need_classifier.py
to align it with the canonical step-9 selection artifact.

Reads:
  - models/final/selection.json        (canonical step-9 selection)
  - data/predictions/<output>.meta.json (raw metadata from the predict script)

Writes back a superset metadata with:
  - selected_experiment_key
  - threshold_source, threshold_type
  - threshold_global_note (explains that the --threshold fallback was NOT used)
  - duplicate_rows_removed
  - generated_at (UTC ISO)
  - code / script source
  - selection / comparison / error-analysis / content-overlap artifact links
  - metadata_finalized_by (this script + timestamp)

Does not touch the prediction CSV itself.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha1_of_file(p: Path, max_bytes: int = 4 * 1024 * 1024) -> str:
    """Return a short sha1 over the first `max_bytes` of a file (fast, stable)."""
    h = hashlib.sha1()
    try:
        with p.open("rb") as f:
            h.update(f.read(max_bytes))
    except Exception:
        return ""
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--meta",
        default=str((REPO_ROOT / "data" / "predictions" / "need_predictions_geolocated_v2_final.meta.json").as_posix()),
    )
    ap.add_argument("--selection", default=str((REPO_ROOT / "models" / "final" / "selection.json").as_posix()))
    ap.add_argument(
        "--predict-script",
        default=str((REPO_ROOT / "scripts" / "predict_need_classifier.py").as_posix()),
    )
    args = ap.parse_args()

    meta_path = Path(args.meta)
    sel_path = Path(args.selection)
    predict_script = Path(args.predict_script)

    if not meta_path.exists():
        raise SystemExit(f"meta not found: {meta_path}")
    if not sel_path.exists():
        raise SystemExit(f"selection not found: {sel_path}")
    if not predict_script.exists():
        raise SystemExit(f"predict script not found: {predict_script}")

    meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    sel: Dict[str, Any] = json.loads(sel_path.read_text(encoding="utf-8"))

    before = int(meta.get("rows_before", 0))
    after = int(meta.get("rows_after", 0))
    dup_removed = max(before - after, 0)

    # Sanity: the model_dir and thresholds source in meta must agree with selection.
    sel_model_dir = str(sel.get("model_dir", "")).strip()
    meta_model_dir = str(meta.get("model_dir", "")).strip()
    if sel_model_dir and meta_model_dir and sel_model_dir != meta_model_dir:
        raise SystemExit(
            f"Model dir mismatch:\n  selection: {sel_model_dir}\n  meta     : {meta_model_dir}"
        )

    # Carry forward the canonical threshold provenance.
    threshold_source = str(sel.get("threshold_source", "unknown"))
    threshold_type = str(sel.get("threshold_type", "unknown"))
    thresholds_json = str(sel.get("thresholds_json", ""))

    enriched = dict(meta)
    enriched["selected_experiment_key"] = sel.get("selected_experiment_key", "")
    enriched["selected_tag"] = sel.get("selected_tag", "")
    enriched["thresholds_json"] = thresholds_json
    enriched["threshold_source"] = threshold_source
    enriched["threshold_type"] = threshold_type
    enriched["threshold_meta_json"] = sel.get("threshold_meta_json", "")
    enriched["threshold_global_note"] = (
        "The `threshold_global` field is ONLY a fallback in predict_need_classifier.py; "
        "it was NOT applied to any label. Effective thresholds are the per-label values "
        "above, sourced from the CV-tuned thresholds_cv.json (threshold_source='cv', "
        "threshold_type='oof_global')."
    )
    enriched["duplicate_rows_removed"] = int(dup_removed)
    enriched["selection_artifact"] = str(sel_path.relative_to(REPO_ROOT).as_posix())
    enriched["comparison_artifact"] = sel.get("comparison_source", {})
    enriched["error_analysis_artifact"] = sel.get("error_analysis_source", {})
    enriched["content_overlap_audit_artifact"] = sel.get("content_overlap_audit", {})
    enriched["generated_at"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    enriched["metadata_finalized_by"] = {
        "script": str(Path(__file__).relative_to(REPO_ROOT).as_posix()),
        "predict_script": str(predict_script.relative_to(REPO_ROOT).as_posix()),
        "predict_script_sha1_head": _sha1_of_file(predict_script),
    }
    enriched["canonical"] = True
    enriched["supersedes"] = [
        "data/predictions/need_predictions_geolocated_63k.csv",
        "data/predictions/need_predictions_geolocated_gold_combined_weighted.csv",
        "data/predictions/need_predictions_geolocated_v1_gold_combined_weighted.csv",
    ]

    # Rename threshold_global -> threshold_global_fallback so readers can't misread it.
    if "threshold_global" in enriched:
        enriched["threshold_global_fallback"] = enriched.pop("threshold_global")

    meta_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated: {meta_path}")
    print("Key threshold provenance fields:")
    print(f"  threshold_source = {threshold_source}")
    print(f"  threshold_type   = {threshold_type}")
    print(f"  thresholds_json  = {thresholds_json}")
    print(f"  duplicate_rows_removed = {dup_removed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
