"""Checks of the PBL-3 implementations (Unit V) against scikit-learn, brute force and theory."""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ensemble import bma, gam, mixture
from ensemble.boosting import AdaBoost, GradientBoosting, fit_stump, stump_predict
from ensemble.committee import Forest
from ensemble.tree import CART


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[:, 3] = rng.integers(0, 2, 300)
    y = np.sin(X[:, 0]) * 3 + X[:, 1] * X[:, 3] + rng.normal(0, 0.3, 300)
    return X, y


def test_cart_regression_matches_sklearn(data):
    X, y = data
    ours = CART("regression", min_samples_leaf=3).fit(X, y)
    sk = DecisionTreeRegressor(min_samples_leaf=3, random_state=0).fit(X, y)
    assert ours.n_nodes == sk.tree_.node_count
    assert np.allclose(ours.predict(X), sk.predict(X), atol=1e-10)


def test_cart_classification_and_binary_path_match_sklearn(data):
    X, y = data
    yc = (y > 0).astype(int)
    # depth 5: deeper nodes are tiny and two thresholds can tie exactly on Gini, where the two
    # libraries break the tie differently (same node count, 2 of 300 rows differ)
    ours = CART("classification", max_depth=5, min_samples_leaf=2).fit(X, yc)
    sk = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=0).fit(X, yc)
    assert np.allclose(ours.predict_proba(X), sk.predict_proba(X))
    rng = np.random.default_rng(1)                       # all-binary design: the matmul path
    Xb = (rng.random((800, 60)) < 0.1).astype(float)
    yb = ((Xb[:, :3].sum(1) > 0) ^ (rng.random(800) < 0.05)).astype(int)
    ours = CART("classification", max_depth=6).fit(Xb, yb)
    sk = DecisionTreeClassifier(max_depth=6, random_state=0).fit(Xb, yb)
    assert ours.n_nodes == sk.tree_.node_count
    assert np.allclose(ours.predict_proba(Xb), sk.predict_proba(Xb))


def test_cost_complexity_path_matches_sklearn(data):
    X, y = data
    ours = CART("regression", min_samples_leaf=3).fit(X, y)
    alphas, impurities, _ = ours.pruning_path()
    sk = DecisionTreeRegressor(min_samples_leaf=3, random_state=0)
    path = sk.cost_complexity_pruning_path(X, y)
    assert np.allclose(alphas, path.ccp_alphas, atol=1e-10)
    assert np.allclose(impurities, path.impurities, atol=1e-10)
    a = path.ccp_alphas[len(path.ccp_alphas) // 2]
    pruned = DecisionTreeRegressor(min_samples_leaf=3, random_state=0, ccp_alpha=a).fit(X, y)
    assert np.allclose(ours.predict(X, alpha=a), pruned.predict(X))


def test_stump_is_the_brute_force_minimum(data):
    X, y = data
    t = np.where(y > 0, 1.0, -1.0)
    w = np.random.default_rng(2).random(len(t))
    j, c, s, err = fit_stump(X, t, w)
    best = min(min(np.sum(w * (t != ss)) for ss in (-1, 1)),          # constant classifiers
               min(np.sum(w * (stump_predict(X, jj, cc, ss) != t))
                   for jj in range(X.shape[1]) for cc in np.unique(X[:, jj]) for ss in (-1, 1)))
    assert err == pytest.approx(best, rel=1e-12)
    assert np.sum(w * (stump_predict(X, j, c, s) != t)) == pytest.approx(err, rel=1e-12)


def test_adaboost_exponential_loss_equals_product_bound(data):
    X, y = data
    ab = AdaBoost(n_rounds=60).fit(X, (y > 0).astype(int))
    assert np.allclose(ab.exp_loss, ab.bound, rtol=1e-10)        # E_M / N = prod 2 sqrt(e(1-e))
    assert all(np.diff(ab.exp_loss) < 0)                          # stagewise descent
    assert all(e <= b + 1e-12 for e, b in zip(ab.train_error, ab.exp_loss))


def test_gradient_boosting_matches_sklearn(data):
    X, y = data
    ours = GradientBoosting("squared", n_rounds=50, learning_rate=0.1, max_depth=3).fit(X, y)
    sk = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3,
                                   random_state=0).fit(X, y)
    assert np.allclose(ours.predict(X), sk.predict(X), atol=1e-8)
    yc = (y > 0).astype(int)
    ours = GradientBoosting("logistic", n_rounds=40, learning_rate=0.2, max_depth=2).fit(X, yc)
    sk = GradientBoostingClassifier(n_estimators=40, learning_rate=0.2, max_depth=2,
                                    random_state=0).fit(X, yc)
    assert np.allclose(ours.predict(X), sk.predict_proba(X)[:, 1], atol=1e-8)


def test_bagging_committee_error_below_average_member_error(data):
    X, y = data
    f = Forest("regression", n_trees=30, seed=0).fit(X[:200], y[:200])
    members = f.member_predictions(X[200:])
    e_av = np.mean((members - y[200:]) ** 2)
    e_com = np.mean((members.mean(0) - y[200:]) ** 2)
    assert e_com <= e_av
    assert 0.9 < f.oob_coverage_ <= 1.0


def test_backfitting_converges_to_direct_gam_solution(data):
    X, y = data
    terms = [gam.Term(0), gam.Term(1), gam.Term(3, "linear"), gam.Term(1, by=3)]
    direct = gam.GAM(terms, lam=3.0).fit(X, y)
    yhat_direct = direct.predict(X)
    bf_terms = [gam.Term(0), gam.Term(1), gam.Term(3, "linear"), gam.Term(1, by=3)]
    bf = gam.GAM(bf_terms, lam=3.0)
    hist = bf.backfit(X, y, tol=1e-12, max_iter=5000)
    assert hist[-1] < 1e-8
    assert np.allclose(bf.predict(X), yhat_direct, atol=1e-6)


def test_bma_full_model_is_ols_and_weights_are_a_distribution(data):
    X, y = data
    m = bma.LinearBMA(g=1e12).fit(X, y)          # g -> infinity: no shrinkage
    assert m.post.sum() == pytest.approx(1.0)
    full = [mm for mm in m.models if mm["k"] == X.shape[1]][0]
    ols = LinearRegression().fit(X, y)
    assert np.allclose(full["beta_post"], ols.coef_, atol=1e-6)
    assert np.all((m.inclusion >= -1e-12) & (m.inclusion <= 1 + 1e-12))
    # the g-prior Bayes factor needs only R^2: check it against the definition by brute force
    n, g = len(y), 50.0
    mg = bma.LinearBMA(g=g).fit(X, y)
    for mm in mg.models:
        k, r2 = mm["k"], mm["r2"]
        assert mm["log_bf"] == pytest.approx(0.5 * (n - 1 - k) * np.log(1 + g)
                                             - 0.5 * (n - 1) * np.log(1 + g * (1 - r2)))


def test_mixture_of_regressions_em_monotone_and_recovers_lines():
    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 3, 600)
    z = rng.random(600) < 0.4
    t = np.where(z, 2 + 3 * x, -1 - 2 * x) + rng.normal(0, 0.5, 600)
    Phi = np.column_stack([np.ones(600), x])
    m = mixture.mix_linreg(Phi, t, 2, n_init=5, var_floor=1e-6)
    assert all(np.diff(m["history"]) > -1e-6)
    slopes = sorted(m["W"][:, 1])
    assert slopes == pytest.approx([-2, 3], abs=0.1)
    assert sorted(m["pi"]) == pytest.approx([0.4, 0.6], abs=0.05)
