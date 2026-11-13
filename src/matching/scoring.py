"""Predicted match score from the saved Phase 1 linear model.

raw     = beta_0 + sum_j beta_j x_j   (stored as is; can be < 0 or > 100)
display = raw clipped to [0, 100], labelled SCORE_LABEL
"""
import numpy as np
import pandas as pd

from matching.features import FEATURES, design_matrix

SCORE_LABEL = "Predicted match score (model trained on AI-judged scores)"
SCORE_MIN, SCORE_MAX = 0.0, 100.0


def predict_raw(model, feats):
    return design_matrix(feats) @ np.asarray(model["beta"], float)


def display_score(raw):
    return np.clip(raw, SCORE_MIN, SCORE_MAX)


def contributions(model, feats):
    """One column per model term: intercept, then beta_j * x_j for each feature. Each row
    sums to the raw prediction."""
    X = design_matrix(feats)
    terms = X * np.asarray(model["beta"], float)[None, :]
    return pd.DataFrame(terms, index=feats.index, columns=["intercept"] + FEATURES)
