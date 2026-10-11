"""Step 5 - linear regression of the AI-judged expected_score, fitted by Gaussian MLE /
least squares.

Target: the AI-judged expected_score column of data/annotation/annotation_sheet.csv.
These scores were judged by Claude Code pair by pair against RUBRIC.md (authorised by the
project owner in place of human raters). They are AI-judged labels, NOT human annotations
and NOT ground truth. Consistency is measured by a test-retest check on a fixed sample
(data/annotation/ai_rescore_check.csv).

Usage: python src/linear_regression_mle.py
Runs only when at least MIN_COMPLETE_ROWS rows have an expected_score; otherwise it records
the step as blocked and fits nothing. This script never generates or imputes scores.

Writes (when ready):  reports/regression_metrics.json, reports/regression_coefficients.csv,
                      data/processed/regression_predictions.csv
Writes (when blocked): reports/regression_metrics.json with status "blocked_pending_annotation"

Model:  y = X beta + eps,  eps ~ N(0, sigma^2 I)
        l(beta, sigma^2) = -n/2 log(2 pi sigma^2) - RSS(beta) / (2 sigma^2)
        For any sigma^2, maximising l over beta == minimising RSS(beta) = ||y - X beta||^2
        beta_hat = (X^T X)^{-1} X^T y,   sigma2_hat_MLE = RSS / n
        l_max = -n/2 (log(2 pi sigma2_hat) + 1)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching.features import FEATURES, design_matrix  # noqa: E402,F401  (shared with the app)

SEED = 42
SHEET = Path("data/annotation/annotation_sheet.csv")
PAIRS = Path("data/processed/pairs.csv")
OUT_METRICS = Path("reports/regression_metrics.json")
OUT_COEF = Path("reports/regression_coefficients.csv")
OUT_PRED = Path("data/processed/regression_predictions.csv")
RESCORE = Path("data/annotation/ai_rescore_check.csv")
RESCORE_BLIND = Path("data/annotation/ai_rescore_blind.csv")
MIN_COMPLETE_ROWS = 300
RESCORE_N = 45
TARGET = "expected_score"
TARGET_LABEL = "AI-judged expected_score"
N_TEST_REQUIREMENTS = 6
DISAGREE_THRESHOLD = 25
RATERS = ["rater_1_score", "rater_2_score"]
SCORE_COLS = RATERS + [TARGET]


# ---------------------------------------------------------------- annotations
def load_annotations():
    """Annotation sheet joined to pair features. Score columns (rater columns and the
    AI-judged expected_score) are coerced to numbers and range-checked."""
    s = pd.read_csv(SHEET, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = [c for c in ["pair_id", "requirement_id", "vendor_id"] + SCORE_COLS if c not in s]
    if missing:
        raise ValueError(f"annotation sheet is missing columns: {missing}")
    for c in SCORE_COLS:
        raw = s[c].str.strip()
        s[c] = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
        bad = raw.ne("") & s[c].isna()
        if bad.any():
            raise ValueError(f"{c}: non-numeric values in pair_ids {s.pair_id[bad].tolist()[:10]}")
        out = s[c].notna() & ~s[c].between(0, 100)
        if out.any():
            raise ValueError(f"{c}: scores outside 0-100 in pair_ids {s.pair_id[out].tolist()[:10]}")
    frac = s[TARGET].notna() & (s[TARGET] % 1 != 0)
    if frac.any():
        raise ValueError(f"{TARGET}: non-integer values in pair_ids {s.pair_id[frac].tolist()[:10]}")
    pairs = pd.read_csv(PAIRS, encoding="utf-8")
    df = s[["pair_id", "requirement_id", "vendor_id"] + SCORE_COLS].merge(
        pairs, on="pair_id", how="left", suffixes=("", "_pairs"), validate="one_to_one")
    mismatch = (df.requirement_id != df.requirement_id_pairs) | (df.vendor_id != df.vendor_id_pairs)
    if mismatch.any():
        raise ValueError(f"sheet rows do not match pairs.csv for {int(mismatch.sum())} pair_ids "
                         "(rows edited or reordered?)")
    return df.drop(columns=["requirement_id_pairs", "vendor_id_pairs"])


def rescore_sample_ids(pair_ids, n=RESCORE_N, seed=SEED):
    """Fixed random sample of pair_ids for the AI judge's test-retest check."""
    ids = sorted(pair_ids)
    rng = np.random.default_rng(seed)
    return sorted(ids[i] for i in rng.choice(len(ids), n, replace=False))


def readiness(df):
    complete = df[TARGET].notna()
    return {"target": TARGET_LABEL, "total_rows": len(df), "complete_rows": int(complete.sum()),
            "rater_1_filled": int(df.rater_1_score.notna().sum()),
            "rater_2_filled": int(df.rater_2_score.notna().sum()),
            "required_complete_rows": MIN_COMPLETE_ROWS,
            "ready": bool(complete.sum() >= MIN_COMPLETE_ROWS)}


def score_agreement(a, b):
    """Agreement between two score vectors on the same pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    r, pval = stats.pearsonr(a, b) if len(a) > 2 else (np.nan, np.nan)
    return {"n_pairs": len(a), "pearson_r": float(r), "pearson_p_value": float(pval),
            "mean_abs_difference": float(np.mean(np.abs(d))),
            f"n_abs_difference_gt_{DISAGREE_THRESHOLD}": int(np.sum(np.abs(d) > DISAGREE_THRESHOLD)),
            "max_abs_difference": float(np.max(np.abs(d))),
            "mean_signed_difference_first_minus_retest": float(np.mean(d)),
            "n_identical": int(np.sum(d == 0))}


def retest_consistency(df, path, col, blind, note):
    """expected_score vs a re-score of the fixed seed-42 sample stored in `path`."""
    re = pd.read_csv(path, encoding="utf-8", dtype={"pair_id": str})
    if sorted(re.pair_id) != rescore_sample_ids(df.pair_id):
        raise ValueError(f"{path} does not hold the fixed seed-{SEED} sample of {RESCORE_N}")
    m = re.merge(df[["pair_id", TARGET]], on="pair_id", validate="one_to_one")
    return {**score_agreement(m[TARGET], m[col]), "blind": blind, "file": str(path),
            "note": note}


def test_retest_consistency(df):
    """Non-blind (same-session) re-score and, if present, the blind fresh-agent re-score."""
    out = {"non_blind": retest_consistency(
        df, RESCORE, "rescore", False,
        "Same AI judge re-scored without re-reading ai_scores.csv, but its first scores were in "
        "its working context; upper bound on consistency.")}
    if RESCORE_BLIND.exists():
        out["blind"] = retest_consistency(
            df, RESCORE_BLIND, "blind_score", True,
            "Fresh AI agent given only RUBRIC.md and the 45 pairs' sheet details; it never saw "
            "the first scores.")
    return out


def human_agreement(df):
    """Reported automatically if any human rater scores are ever filled in."""
    out = {"rater_1_filled": int(df.rater_1_score.notna().sum()),
           "rater_2_filled": int(df.rater_2_score.notna().sum())}
    both = df[RATERS].notna().all(axis=1)
    if both.sum() > 0:
        out["rater_1_vs_rater_2"] = score_agreement(df.rater_1_score[both], df.rater_2_score[both])
    anyr = df[RATERS].notna().any(axis=1) & df[TARGET].notna()
    if anyr.sum() > 0:
        human = df.loc[anyr, RATERS].mean(axis=1)
        out["ai_expected_score_vs_human_mean"] = score_agreement(df.loc[anyr, TARGET], human)
    if len(out) == 2:
        out["status"] = "no human scores filled; human validation not done"
    return out


# ---------------------------------------------------------------- estimation
def fit_ols(X, y):
    """Least squares = Gaussian MLE for beta. Solved with lstsq (QR/SVD based, stable);
    the explicit inverse (X^T X)^{-1} X^T y is computed only as a cross-check."""
    beta, _, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    out = {"beta": beta, "rank": int(rank), "n": len(y), "p": X.shape[1],
           "cond_X": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")}
    try:
        beta_inv = np.linalg.inv(X.T @ X) @ X.T @ y
        out["max_abs_diff_inverse_vs_lstsq"] = float(np.max(np.abs(beta_inv - beta)))
    except np.linalg.LinAlgError:
        out["max_abs_diff_inverse_vs_lstsq"] = None  # X^T X singular
    sk = LinearRegression(fit_intercept=False).fit(X, y)
    out["max_abs_diff_sklearn_vs_lstsq"] = float(np.max(np.abs(sk.coef_ - beta)))
    return out


def fit_metrics(X, y, beta):
    resid = y - X @ beta
    rss = float(resid @ resid)
    n = len(y)
    return {"n": n, "RSS": rss, "MSE": rss / n, "RMSE": float(np.sqrt(rss / n)),
            "MAE": float(np.mean(np.abs(resid))),
            "R2": float(1 - rss / np.sum((y - y.mean()) ** 2))}


def gaussian_mle(X, y, beta):
    """sigma^2 MLE, unbiased s^2, and the maximised log-likelihood computed two ways."""
    n, p = X.shape
    resid = y - X @ beta
    rss = float(resid @ resid)
    sigma2 = rss / n
    ll_direct = float(np.sum(stats.norm.logpdf(resid, loc=0.0, scale=np.sqrt(sigma2))))
    ll_formula = float(-n / 2 * (np.log(2 * np.pi * sigma2) + 1))
    return {"sigma2_mle": sigma2, "sigma_mle": float(np.sqrt(sigma2)),
            "s2_unbiased": rss / (n - p), "s_unbiased": float(np.sqrt(rss / (n - p))),
            "loglik_direct_sum": ll_direct, "loglik_formula": ll_formula,
            "loglik_match": bool(np.isclose(ll_direct, ll_formula, rtol=1e-10, atol=1e-8))}


def split_by_requirement(df):
    """Hold out whole requirements: at deployment the requirements are new."""
    gss = GroupShuffleSplit(n_splits=1, test_size=N_TEST_REQUIREMENTS, random_state=SEED)
    tr, te = next(gss.split(df, groups=df.requirement_id))
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    assert not set(train.requirement_id) & set(test.requirement_id)
    return train, test


def run(df):
    """Full analysis on complete rows. Returns a results dict (used by script, plots, notebook)."""
    complete = df[df[TARGET].notna()].copy()
    complete["y"] = complete[TARGET].astype(float)
    consistency = test_retest_consistency(df)
    train, test = split_by_requirement(complete)
    X_tr, X_te = design_matrix(train), design_matrix(test)
    y_tr, y_te = train.y.to_numpy(), test.y.to_numpy()
    ols = fit_ols(X_tr, y_tr)
    beta = ols["beta"]
    return {"consistency": consistency, "train": train, "test": test, "X_tr": X_tr, "X_te": X_te,
            "y_tr": y_tr, "y_te": y_te, "ols": ols, "beta": beta,
            "train_metrics": fit_metrics(X_tr, y_tr, beta),
            "test_metrics": fit_metrics(X_te, y_te, beta),
            "mle": gaussian_mle(X_tr, y_tr, beta)}


def main():
    df = load_annotations()
    status = readiness(df)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    if not status["ready"]:
        OUT_METRICS.write_text(json.dumps({"status": "blocked_pending_annotation", **status},
                                          indent=2), encoding="utf-8")
        print(f"BLOCKED: {status['complete_rows']} of {status['total_rows']} rows have an "
              f"{TARGET_LABEL}; need >= {MIN_COMPLETE_ROWS}. Nothing fitted.")
        print(f"wrote {OUT_METRICS}")
        return

    r = run(df)
    names = ["intercept"] + FEATURES
    coef = pd.DataFrame({"feature": names, "beta": r["beta"]})
    coef.to_csv(OUT_COEF, index=False, encoding="utf-8", lineterminator="\n")
    preds = pd.concat([
        r["train"].assign(split="train", fitted=r["X_tr"] @ r["beta"]),
        r["test"].assign(split="test", fitted=r["X_te"] @ r["beta"])])
    preds["residual"] = preds.y - preds.fitted
    preds[["pair_id", "requirement_id", "split", "y", "fitted", "residual"]].to_csv(
        OUT_PRED, index=False, encoding="utf-8", lineterminator="\n")
    ols = {k: v for k, v in r["ols"].items() if k != "beta"}
    metrics = {"status": "fitted", **status,
               "target_note": ("AI-judged expected_score: judged by Claude Code against "
                               "RUBRIC.md; not human-annotated, not ground truth."),
               "test_retest_consistency": r["consistency"],
               "human_agreement": human_agreement(df),
               "split": {"method": "GroupShuffleSplit by requirement_id", "seed": SEED,
                         "train_requirements": int(r["train"].requirement_id.nunique()),
                         "test_requirements": int(r["test"].requirement_id.nunique()),
                         "n_train": len(r["y_tr"]), "n_test": len(r["y_te"])},
               "coefficients": dict(zip(names, map(float, r["beta"]))), "ols": ols,
               "train": r["train_metrics"], "test": r["test_metrics"], "gaussian_mle": r["mle"],
               "mse_equals_sigma2_mle_train": bool(np.isclose(r["train_metrics"]["MSE"],
                                                              r["mle"]["sigma2_mle"]))}
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {OUT_METRICS}, {OUT_COEF}, {OUT_PRED}")


if __name__ == "__main__":
    main()
