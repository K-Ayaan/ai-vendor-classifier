"""Shared helpers for the PBL-3 scripts: same palette and plot style as PBL-1/PBL-2, plus the
Phase 1 data in the exact Phase 1 splits (regression by requirement, classification stratified)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linear_regression_mle as lr  # noqa: E402
import logistic_regression_classification as L  # noqa: E402
from pbl2_style import (AQUA, BLUE, GRID, INK, INK2, LIGHT, MUTED, ORANGE, PALETTE,  # noqa: E402,F401
                        PURPLE, SEED, SURFACE, md_table, plt, save, write_json)

PBL3 = Path("pbl-3")
FEATURES = list(lr.FEATURES)
YELLOW, PINK = PALETTE[4], PALETTE[5]


def f4(x):
    return f"{x:.4f}"


def regression_data():
    """Phase 1 regression: 360 train / 90 test pairs, split by requirement (6 test buyers).
    Adds the requirement's radius (for EDA only, it is not a model input)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = lr.run(lr.load_annotations())
    req = pd.read_csv("data/processed/requirements.csv", encoding="utf-8")
    for k in ("train", "test"):
        r[k] = r[k].merge(req[["requirement_id", "max_distance_km"]], on="requirement_id",
                          how="left")
    r["Xn_tr"], r["Xn_te"] = r["X_tr"][:, 1:], r["X_te"][:, 1:]     # no ones column
    r["groups"] = r["train"].requirement_id.to_numpy()
    return r


def classification_data():
    """Phase 1 classification: 3,689 / 923 vendors, 441 binary columns + intercept."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = L.run()
    c["Xn_tr"], c["Xn_te"] = c["X_tr"][:, 1:], c["X_te"][:, 1:]
    c["names"] = c["fb"].names[1:]
    return c


def reg_metrics(y, yhat):
    rss = float(np.sum((y - yhat) ** 2))
    return {"RMSE": float(np.sqrt(rss / len(y))), "MAE": float(np.mean(np.abs(y - yhat))),
            "R2": float(1 - rss / np.sum((y - y.mean()) ** 2))}


def clf_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    return {"tn": int(((pred == 0) & (y == 0)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()), "tp": int(((pred == 1) & (y == 1)).sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)), "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))}
