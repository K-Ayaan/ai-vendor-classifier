"""Phase 2 Step 1 - save the Phase 1 models for the app.

Usage: python src/build_artifacts.py
Refits, with the same data, split, seed and code as Phase 1:
  * the linear match-score model (linear_regression_mle.run, 24 train requirements)
  * the L2 is_pharmacy classifier (FeatureBuilder + C chosen by 5-fold CV on train)
checks both against the Phase 1 reports, and writes artifacts/ (joblib + model_card.json).
It reads Phase 1 outputs only; it writes nothing outside artifacts/.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linear_regression_mle as lr  # noqa: E402
import logistic_regression_classification as L  # noqa: E402
from matching.artifacts import (ARTIFACTS, CARD_PATH, CLASSIFIER_KIND, CLASSIFIER_PATH,  # noqa: E402
                                LINEAR_KIND, LINEAR_PATH, load_classifier, load_linear_model,
                                pharmacy_probability)
from matching.features import FEATURES  # noqa: E402

CAVEAT = ("The match-score target is the AI-judged expected_score (Claude Code scoring each "
          "pair against data/annotation/RUBRIC.md). These are AI-judged labels, not human "
          "annotations and not ground truth; human validation was not done. Predictions are "
          "unbounded (some are below 0) and are clipped to 0-100 only for display.")
PHARMACY_CAVEAT = ("is_pharmacy is a weak label (the vendor appears in the pharmacy source "
                   "file). The probability informs the user; it never decides eligibility.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_linear():
    df = lr.load_annotations()
    r = lr.run(df)
    beta = np.asarray(r["beta"], float)
    phase1 = pd.read_csv("reports/regression_coefficients.csv")
    assert phase1.feature.tolist() == ["intercept"] + FEATURES
    max_diff = float(np.max(np.abs(phase1.beta.to_numpy() - beta)))
    assert max_diff < 1e-9, f"refit differs from Phase 1 coefficients by {max_diff}"
    model = {"kind": LINEAR_KIND, "feature_names": list(FEATURES),
             "columns": ["intercept"] + FEATURES, "beta": beta}
    card = {
        "kind": LINEAR_KIND,
        "target": lr.TARGET_LABEL,
        "training_data": {
            "files": {p: sha256(p) for p in [str(lr.SHEET).replace("\\", "/"),
                                             str(lr.PAIRS).replace("\\", "/")]},
            "split": "GroupShuffleSplit by requirement_id, seed 42",
            "train_requirements": sorted(r["train"].requirement_id.unique()),
            "test_requirements": sorted(r["test"].requirement_id.unique()),
            "n_train_pairs": len(r["y_tr"]), "n_test_pairs": len(r["y_te"])},
        "features": list(FEATURES),
        "coefficients": dict(zip(model["columns"], map(float, beta))),
        "train_metrics": r["train_metrics"], "test_metrics": r["test_metrics"],
        "sigma2_mle": r["mle"]["sigma2_mle"],
        "max_abs_diff_vs_reports_regression_coefficients_csv": max_diff,
        "caveat": CAVEAT,
    }
    return model, card


def build_classifier():
    v = L.load_vendors()
    train, test = L.split(v)
    fb = L.FeatureBuilder().fit(train)
    X_tr, X_te = fb.transform(train), fb.transform(test)
    y_tr, y_te = train.y.to_numpy(), test.y.to_numpy()
    best_c, _ = L.choose_c(X_tr, y_tr)
    model = L.sk_logreg(best_c, max_iter=10000, tol=1e-10).fit(X_tr[:, 1:], y_tr)
    clf = {"kind": CLASSIFIER_KIND, "feature_builder": fb, "model": model, "C": best_c,
           "feature_names": list(fb.names)}
    test_metrics = L.evaluate(y_te, pharmacy_probability(clf, test))
    phase1 = json.loads(Path("reports/classification_metrics.json").read_text(encoding="utf-8"))
    p1 = phase1["l2_model"]
    assert np.isclose(best_c, p1["C"]) and test_metrics["confusion_matrix"] == \
        p1["test_metrics"]["confusion_matrix"]
    for k in ["f1", "roc_auc", "pr_auc", "log_loss"]:
        assert np.isclose(test_metrics[k], p1["test_metrics"][k], rtol=1e-9), k
    coef = pd.read_csv("reports/classification_coefficients.csv")
    coef_diff = float(np.max(np.abs(coef.beta_sklearn_l2.to_numpy() - L.sk_beta(model))))
    assert coef.feature.tolist() == fb.names and coef_diff < 1e-9, coef_diff
    card = {
        "kind": CLASSIFIER_KIND,
        "label": "penalised likelihood (MAP), not MLE",
        "target": "is_pharmacy (weak label)",
        "training_data": {"files": {"data/processed/vendors.csv": sha256(L.VENDORS)},
                          "split": "stratified train_test_split 80/20, seed 42",
                          "n_train": len(y_tr), "n_test": len(y_te)},
        "C": best_c, "lambda_equiv": 1.0 / best_c,
        "feature_names": list(fb.names),
        "n_features_incl_intercept": len(fb.names),
        "coefficients": dict(zip(fb.names, map(float, L.sk_beta(model)))),
        "test_metrics": test_metrics,
        "max_abs_diff_vs_reports_classification_coefficients_csv": coef_diff,
        "caveat": PHARMACY_CAVEAT,
    }
    return clf, card


def main():
    linear, linear_card = build_linear()
    clf, clf_card = build_classifier()
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(linear, LINEAR_PATH)
    joblib.dump(clf, CLASSIFIER_PATH)
    card = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "created_by": "src/build_artifacts.py",
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                         "joblib": joblib.__version__},
            "linear_model": linear_card, "pharmacy_classifier": clf_card}
    # full precision (no rounding): the loader checks the coefficients exactly
    CARD_PATH.write_text(json.dumps(card, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    # round-trip through the validating loaders
    lin = load_linear_model()
    c = load_classifier()
    print(f"linear model: beta {dict(zip(lin['columns'], np.round(lin['beta'], 4)))}")
    print(f"  test R2 {linear_card['test_metrics']['R2']:.4f}, RMSE "
          f"{linear_card['test_metrics']['RMSE']:.4f}; max |beta - Phase 1 csv| "
          f"{linear_card['max_abs_diff_vs_reports_regression_coefficients_csv']:.2e}")
    tm = clf_card["test_metrics"]
    print(f"L2 classifier: C {c['C']:.4g}, {len(c['feature_names'])} features, test F1 "
          f"{tm['f1']:.4f}, cm {tm['confusion_matrix']}; max |beta - Phase 1 csv| "
          f"{clf_card['max_abs_diff_vs_reports_classification_coefficients_csv']:.2e}")
    print(f"wrote {LINEAR_PATH}, {CLASSIFIER_PATH}, {CARD_PATH}")


if __name__ == "__main__":
    main()
