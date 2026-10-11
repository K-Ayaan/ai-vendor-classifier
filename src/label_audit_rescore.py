"""Step 4b - re-score the classifiers against AI-judged is_pharmacy labels.

Usage: python src/label_audit_rescore.py
Reads data/annotation/label_audit_sample.csv (column expected_is_pharmacy: AI-judged
labels, NOT ground truth, judged by Claude Code from title and address only) and
data/processed/label_audit_key.csv (weak labels), refits the Step 4 models with the same
seed/split, and writes reports/label_audit_metrics.json.

The sample is stratified by weak label (all test weak positives + a random subset of weak
negatives), so every estimate for the test population is re-weighted by stratum:
w = N_stratum_in_test / n_stratum_in_sample. Unweighted sample figures are reported too.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logistic_regression_classification as L  # noqa: E402

SHEET = Path("data/annotation/label_audit_sample.csv")
KEY = Path("data/processed/label_audit_key.csv")
OUT = Path("reports/label_audit_metrics.json")
LABEL_NOTE = "AI-judged labels, not ground truth"


def metrics(y, p, w=None, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1], sample_weight=w).ravel()
    return {"confusion_matrix": {"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)},
            "accuracy": accuracy_score(y, pred, sample_weight=w),
            "precision": precision_score(y, pred, sample_weight=w, zero_division=0),
            "recall": recall_score(y, pred, sample_weight=w),
            "f1": f1_score(y, pred, sample_weight=w),
            "roc_auc": roc_auc_score(y, p, sample_weight=w),
            "pr_auc": average_precision_score(y, p, sample_weight=w)}


def main():
    sheet = pd.read_csv(SHEET, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    if "expected_is_pharmacy" not in sheet or (sheet.expected_is_pharmacy == "").any():
        raise SystemExit("expected_is_pharmacy missing or incomplete - nothing to re-score")
    ai = sheet[["vendor_id", "expected_is_pharmacy"]].assign(
        ai=lambda d: d.expected_is_pharmacy.astype(int))
    if not ai.ai.isin([0, 1]).all():
        raise ValueError("expected_is_pharmacy must be 0 or 1")
    key = pd.read_csv(KEY, encoding="utf-8", dtype={"vendor_id": str})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = L.run()
    test = r["test"].assign(p_l2=r["p_l2"], p_unpen=r["p_unpen"], p_small=r["small"]["p_test"])
    d = ai.merge(key[["vendor_id", "weak_is_pharmacy"]], on="vendor_id", validate="one_to_one")
    d = d.merge(test[["vendor_id", "y", "p_l2", "p_unpen", "p_small"]], on="vendor_id",
                validate="one_to_one")
    assert len(d) == len(sheet) and (d.weak_is_pharmacy == d.y).all(), "key/test mismatch"

    # stratum weights: test-set count / sample count, per weak label
    n_test = test.y.value_counts().to_dict()
    n_samp = d.y.value_counts().to_dict()
    weights = {int(k): n_test[k] / n_samp[k] for k in n_samp}
    w = d.y.map(weights).to_numpy()
    y_ai, y_weak = d.ai.to_numpy(), d.y.to_numpy()

    agree = {
        "n_sample": len(d),
        "ai_positive": int(y_ai.sum()), "ai_negative": int((1 - y_ai).sum()),
        "crosstab_weak_vs_ai": {f"weak={a},ai={b}": int(((y_weak == a) & (y_ai == b)).sum())
                                for a in (0, 1) for b in (0, 1)},
        "agreement_sample_unweighted": float(np.mean(y_ai == y_weak)),
        "agreement_test_reweighted": float(np.average(y_ai == y_weak, weights=w)),
        "share_weak_positives_ai_confirms": float(np.mean(y_ai[y_weak == 1] == 1)),
        "share_weak_negatives_ai_confirms": float(np.mean(y_ai[y_weak == 0] == 0)),
        "cohen_kappa_sample_unweighted": float(cohen_kappa_score(y_weak, y_ai)),
        "cohen_kappa_test_reweighted": float(cohen_kappa_score(y_weak, y_ai, sample_weight=w)),
        "estimated_ai_positive_rate_test_reweighted": float(np.average(y_ai, weights=w)),
        "weak_positive_rate_test": float(test.y.mean()),
    }
    # reported automatically if humans ever fill true_is_pharmacy (0/1)
    human = sheet[["vendor_id", "true_is_pharmacy"]].merge(d[["vendor_id", "ai", "y"]], on="vendor_id")
    human = human[human.true_is_pharmacy.str.strip() != ""]
    if len(human):
        h = human.true_is_pharmacy.str.strip().astype(int)
        if not h.isin([0, 1]).all():
            raise ValueError("true_is_pharmacy must be 0 or 1")
        human_agreement = {"n_filled": len(human),
                           "ai_vs_human_agreement": float(np.mean(human.ai == h)),
                           "ai_vs_human_cohen_kappa": float(cohen_kappa_score(h, human.ai)),
                           "weak_vs_human_agreement": float(np.mean(human.y == h))}
    else:
        human_agreement = {"n_filled": 0,
                           "status": "true_is_pharmacy not filled; human validation not done"}

    models = {"small_mle": "p_small", "l2_map": "p_l2", "full_unpenalised_invalid_mle": "p_unpen"}
    res = {}
    for name, col in models.items():
        p = d[col].to_numpy()
        res[name] = {"vs_ai_labels_test_reweighted": metrics(y_ai, p, w),
                     "vs_ai_labels_sample_unweighted": metrics(y_ai, p),
                     "vs_weak_labels_test_reweighted": metrics(y_weak, p, w)}
    out = {"label_note": LABEL_NOTE,
           "labelling": "Claude Code judged is_pharmacy from title and address only; did not "
                        "open the key, weak labels or predictions while labelling. Caveat: 12 "
                        "audit vendors' weak labels had been shown earlier in the session (Step 4 "
                        "error analysis).",
           "stratum_weights": {f"weak={k}": v for k, v in sorted(weights.items())},
           "test_counts": {f"weak={k}": int(v) for k, v in sorted(n_test.items())},
           "sample_counts": {f"weak={k}": int(v) for k, v in sorted(n_samp.items())},
           "agreement_ai_vs_weak": agree, "human_agreement": human_agreement,
           "classifiers": res}
    OUT.write_text(json.dumps(L.round_floats(out), indent=2), encoding="utf-8")

    print(f"[{LABEL_NOTE}] weights {out['stratum_weights']}")
    print("AI vs weak:", json.dumps(L.round_floats(agree), indent=1))
    for name, m in res.items():
        a, b = m["vs_ai_labels_test_reweighted"], m["vs_weak_labels_test_reweighted"]
        print(f"{name:30s} vs AI (reweighted): acc {a['accuracy']:.4f} P {a['precision']:.4f} "
              f"R {a['recall']:.4f} F1 {a['f1']:.4f} ROC {a['roc_auc']:.4f} PR {a['pr_auc']:.4f}"
              f" | vs weak (reweighted): F1 {b['f1']:.4f} ROC {b['roc_auc']:.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
