"""Load the saved Phase 1 models from artifacts/ and validate them against the shared
feature definition (features.FEATURES) and artifacts/model_card.json.

Artifacts (written by src/build_artifacts.py):
    artifacts/linear_model.joblib   {"kind", "feature_names", "columns", "beta"}
    artifacts/pharmacy_l2.joblib    {"kind", "feature_builder", "model", "C", "feature_names"}
    artifacts/model_card.json
"""
import json
from pathlib import Path

import joblib
import numpy as np

from matching.features import FEATURES

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
LINEAR_PATH = ARTIFACTS / "linear_model.joblib"
CLASSIFIER_PATH = ARTIFACTS / "pharmacy_l2.joblib"
CARD_PATH = ARTIFACTS / "model_card.json"
LINEAR_KIND = "linear_regression_gaussian_mle"
CLASSIFIER_KIND = "logistic_regression_l2_map"


class ArtifactError(ValueError):
    pass


def load_model_card(path=CARD_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_linear_model(path=LINEAR_PATH, card_path=CARD_PATH):
    """The saved linear match-score model. Raises ArtifactError unless its feature list and
    order equal features.FEATURES and the model card, and beta has one entry per column."""
    m = joblib.load(path)
    card = load_model_card(card_path)["linear_model"]
    if m.get("kind") != LINEAR_KIND:
        raise ArtifactError(f"{path}: kind {m.get('kind')!r}, expected {LINEAR_KIND!r}")
    if list(m["feature_names"]) != FEATURES:
        raise ArtifactError(f"{path}: features {m['feature_names']} != shared FEATURES {FEATURES}")
    if card["features"] != FEATURES:
        raise ArtifactError(f"model card features {card['features']} != shared FEATURES {FEATURES}")
    if list(m["columns"]) != ["intercept"] + FEATURES or len(m["beta"]) != len(FEATURES) + 1:
        raise ArtifactError(f"{path}: columns/beta do not match intercept + FEATURES")
    card_beta = np.array([card["coefficients"][c] for c in m["columns"]])
    if not np.array_equal(card_beta, np.asarray(m["beta"])):
        raise ArtifactError(f"{path}: coefficients differ from the model card")
    return m


def load_classifier(path=CLASSIFIER_PATH, card_path=CARD_PATH):
    """The saved L2 is_pharmacy classifier with its fitted FeatureBuilder."""
    m = joblib.load(path)
    card = load_model_card(card_path)["pharmacy_classifier"]
    if m.get("kind") != CLASSIFIER_KIND:
        raise ArtifactError(f"{path}: kind {m.get('kind')!r}, expected {CLASSIFIER_KIND!r}")
    names = list(m["feature_builder"].names)
    if names != list(m["feature_names"]) or names != card["feature_names"]:
        raise ArtifactError(f"{path}: classifier feature list differs from the model card")
    if names[0] != "intercept" or m["model"].coef_.shape[1] != len(names) - 1:
        raise ArtifactError(f"{path}: model has {m['model'].coef_.shape[1]} weights for "
                            f"{len(names) - 1} non-intercept features")
    return m


def pharmacy_probability(classifier, vendors):
    """P(is_pharmacy) for each vendor row (needs title and pincode)."""
    X = classifier["feature_builder"].transform(vendors)
    return classifier["model"].predict_proba(X[:, 1:])[:, 1]
