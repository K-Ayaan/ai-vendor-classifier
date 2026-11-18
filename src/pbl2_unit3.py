"""PBL-2 Unit III: mixture models and EM, PCA / factor analysis, clustering.

Usage: python src/pbl2_unit3.py
Reads data/processed/ and data/annotation/ (read-only). Writes reports/pbl2_unit3_metrics.json,
pbl-2/03_unit_topics_implementation/figures/U3_*.png and .../unit3_numbers.md.

1. Measuring dissimilarity: haversine vs projected km vs raw degrees; Jaccard; correlation
2. K-means (own Lloyd + k-means++) on vendor locations, K = 2..15
3. Gaussian mixture by EM (own), K = 1..15: log-likelihood, BIC, AIC, held-out likelihood
4. The alternative view of EM: ln p(X) = L(q, theta) + KL(q || p(Z|X)) tracked per iteration
5. K-means as the small-variance limit of EM; hard vs soft assignments
6. Evaluating clusterings: silhouette, Davies-Bouldin, Calinski-Harabasz (internal);
   ARI / NMI against locality and pincode (external); stability across seeds
7. PCA (own eigendecomposition) of the retail mix of 2 km grid cells; choosing the number of
   latent dimensions with scree, cumulative variance, PPCA and FA cross-validated likelihood
8. Hierarchical clustering of the 23 categories (where they locate together), 4 linkages
9. Does a GMM zone feature help the Phase 1 match-score regression? (train/test by requirement)
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linear_regression_mle as lr  # noqa: E402
import logistic_regression_classification as L  # noqa: E402
from latent import clustering as cl  # noqa: E402
from latent.pca import pca_eig, ppca_loglik, ppca_params  # noqa: E402
from matching.features import category_set, design_matrix  # noqa: E402
from preprocessing import haversine_km  # noqa: E402
from pbl2_style import (AQUA, BLUE, INK2, LIGHT, MUTED, ORANGE, PALETTE, PBL2,  # noqa: E402
                        PURPLE, SEED, md_table, plt, save, to_km, write_json)

OUT = PBL2 / "03_unit_topics_implementation"
FIG = OUT / "figures"
METRICS = Path("reports/pbl2_unit3_metrics.json")
K_RANGE = range(1, 26)
EM_RESTARTS = 3  # EM finds local optima: keep the best of 3 k-means-initialised runs
CELL_KM = 2.0
MIN_CELL_VENDORS = 15
N_FOLDS = 5
TOP_LOCALITIES = 12


def f4(x):
    return f"{x:.4f}"


def load():
    v = L.load_vendors()
    v["cats"] = v.categories.map(category_set)
    lat0, lon0 = float(v.lat.mean()), float(v.lon.mean())
    X = to_km(v.lat, v.lon, lat0, lon0)
    return v, X, lat0, lon0


# ---------------------------------------------------------------- 1 dissimilarity
def dissimilarity(v, X, md, m):
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(len(v), size=(2, 20000))
    keep = i != j
    i, j = i[keep], j[keep]
    hav = haversine_km(v.lat.values[i], v.lon.values[i], v.lat.values[j], v.lon.values[j])
    proj = np.linalg.norm(X[i] - X[j], axis=1)
    deg = np.hypot(v.lat.values[i] - v.lat.values[j], v.lon.values[i] - v.lon.values[j]) * 111.32
    rel_proj = np.abs(proj - hav) / hav
    rel_deg = np.abs(deg - hav) / hav
    Xdeg = np.column_stack([v.lon, v.lat])
    km_deg = KMeans(8, n_init=10, random_state=SEED).fit(Xdeg).labels_
    km_km = KMeans(8, n_init=10, random_state=SEED).fit(X).labels_
    m["dissimilarity"] = {"pairs": int(keep.sum()),
                          "projected_km_max_rel_error": float(rel_proj.max()),
                          "projected_km_mean_rel_error": float(rel_proj.mean()),
                          "raw_degrees_max_rel_error": float(rel_deg.max()),
                          "raw_degrees_mean_rel_error": float(rel_deg.mean()),
                          "kmeans8_degrees_vs_km_ARI": float(adjusted_rand_score(km_deg, km_km)),
                          "kmeans8_vendors_changing_cluster_approx":
                              int(len(v) - np.max(pd.crosstab(km_deg, km_km).values, 1).sum())}
    d = m["dissimilarity"]
    md += ["## 1 Measuring dissimilarity", "",
           f"Over {d['pairs']} random vendor pairs, against the haversine (great-circle) distance:",
           "", md_table(pd.DataFrame({
               "distance used": ["Euclidean on projected km (x = Δlon·111.32·cos φ0, y = Δlat·111.32)",
                                 "Euclidean on raw degrees (×111.32)"],
               "mean rel. error": [f"{d['projected_km_mean_rel_error']:.5f}",
                                   f"{d['raw_degrees_mean_rel_error']:.5f}"],
               "max rel. error": [f"{d['projected_km_max_rel_error']:.5f}",
                                  f"{d['raw_degrees_max_rel_error']:.5f}"]})), "",
           f"- K-means (K = 8) on raw degrees vs on km: ARI {d['kmeans8_degrees_vs_km_ARI']:.4f}; "
           f"about {d['kmeans8_vendors_changing_cluster_approx']} vendors change cluster. "
           "A degree of longitude is shorter than a degree of latitude here, so raw degrees "
           "stretch east-west distances.", ""]


# ---------------------------------------------------------------- 2-5 k-means and GMM
def best_gmm(X, K, restarts=EM_RESTARTS, **kw):
    return max((cl.gmm_em(X, K, seed=SEED + r, **kw) for r in range(restarts)),
               key=lambda g: g["loglik"])


def knee(xs, ys):
    """Kneedle rule: the point of a decreasing curve farthest below the straight line joining
    its first and last points (both axes scaled to [0, 1])."""
    x = (np.asarray(xs, float) - xs[0]) / (xs[-1] - xs[0])
    y = (np.asarray(ys, float) - ys[-1]) / (ys[0] - ys[-1])
    return int(np.asarray(xs)[np.argmax((1 - x) - y)])


def kmeans_gmm(v, X, md, m):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    Xtr, Xte = X[perm[:n_tr]], X[perm[n_tr:]]
    rows, km_runs, gmm_runs = [], {}, {}
    for K in K_RANGE:
        g = best_gmm(X, K)
        gtr = best_gmm(Xtr, K, restarts=1)
        held = cl.gmm_loglik(Xte, gtr["pi"], gtr["mus"], gtr["covs"]) / len(Xte)
        row = {"K": K, "gmm_loglik": g["loglik"], "gmm_params": g["n_params"],
               "BIC": cl.bic(g["loglik"], g["n_params"], len(X)),
               "AIC": cl.aic(g["loglik"], g["n_params"]), "heldout_ll_per_vendor": held,
               "em_iterations": g["n_iter"]}
        gmm_runs[K] = g
        if K >= 2:
            k = cl.kmeans(X, K, n_init=10, seed=SEED)
            km_runs[K] = k
            row.update({"kmeans_inertia": k["inertia"],
                        "silhouette": float(silhouette_score(X, k["labels"])),
                        "davies_bouldin": float(davies_bouldin_score(X, k["labels"])),
                        "calinski_harabasz": float(calinski_harabasz_score(X, k["labels"]))})
        rows.append(row)
    tab = pd.DataFrame(rows)
    K_bic = int(tab.loc[tab.BIC.idxmin(), "K"])
    K_aic = int(tab.loc[tab.AIC.idxmin(), "K"])
    K_held = int(tab.loc[tab.heldout_ll_per_vendor.idxmax(), "K"])
    K_sil = int(tab.loc[tab.silhouette.idxmax(), "K"])
    K_knee = knee(tab.K.values, tab.BIC.values)
    sk_km = KMeans(K_knee, n_init=10, random_state=SEED).fit(X)
    sk_g = GaussianMixture(K_knee, covariance_type="full", reg_covar=cl.COV_REG, n_init=3,
                           random_state=SEED, max_iter=1000, tol=1e-8).fit(X)
    m["choose_K"] = {"table": rows, "K_by_BIC": K_bic, "K_by_AIC": K_aic,
                     "K_by_heldout_ll": K_held, "K_by_silhouette": K_sil,
                     "K_by_BIC_knee_USED": K_knee,
                     "bic_still_falling_at_max_K": bool(K_bic == max(K_RANGE)),
                     "own_vs_sklearn": {
                         "K": K_knee, "kmeans_inertia_own": km_runs[K_knee]["inertia"],
                         "kmeans_inertia_sklearn": float(sk_km.inertia_),
                         "gmm_loglik_own": gmm_runs[K_knee]["loglik"],
                         "gmm_loglik_sklearn": float(sk_g.score(X) * len(X))}}
    K = K_knee
    g = best_gmm(X, K, track_bound=True)
    km = km_runs[K]

    # alternative view of EM
    hist = np.array(g["history"])
    kl = np.array(g["kl_after_m"])
    m["em_view"] = {"K": K, "iterations": g["n_iter"], "loglik_first": hist[0],
                    "loglik_last": hist[-1],
                    "loglik_non_decreasing": bool(np.all(np.diff(hist) >= -1e-8)),
                    "min_kl_after_m": float(kl.min()), "max_kl_after_m": float(kl.max()),
                    "kl_after_m_first5": kl[:5].tolist()}

    # k-means as the small-variance limit of EM (equal weights, shared covariance eps*I)
    lim = {}
    for eps in [10.0, 1.0, 0.1, 0.01]:
        mus = cl.kmeans(X, K, n_init=1, seed=SEED + 1)["centres"]
        for _ in range(200):
            R, _ = cl.e_step(X, np.full(K, 1 / K), mus, np.array([eps * np.eye(2)] * K))
            new = (R.T @ X) / R.sum(0)[:, None]
            if np.max(np.abs(new - mus)) < 1e-9:
                break
            mus = new
        lim[str(eps)] = {"ARI_vs_kmeans": float(adjusted_rand_score(R.argmax(1), km["labels"])),
                         "mean_max_responsibility": float(R.max(1).mean())}
    m["kmeans_limit"] = lim

    # hard vs soft
    rmax = g["R"].max(1)
    ari_km_gmm = float(adjusted_rand_score(km["labels"], g["labels"]))
    m["hard_vs_soft"] = {"ARI_kmeans_vs_gmm": ari_km_gmm,
                         "vendors_max_resp_below_0.9": int((rmax < 0.9).sum()),
                         "vendors_max_resp_below_0.6": int((rmax < 0.6).sum()),
                         "mean_max_resp": float(rmax.mean()),
                         "cluster_sizes_kmeans": np.bincount(km["labels"]).tolist(),
                         "gmm_weights": g["pi"].tolist()}

    # external evaluation + stability
    loc = v.locality.fillna("(none)")
    top = loc.value_counts().index[:TOP_LOCALITIES]
    loc_lab = loc.where(loc.isin(top), "other")
    pin = v.pincode.astype(str)
    ext = {}
    for name, lab in [("kmeans", km["labels"]), ("gmm", g["labels"])]:
        ext[name] = {"ARI_locality": float(adjusted_rand_score(loc_lab, lab)),
                     "NMI_locality": float(normalized_mutual_info_score(loc_lab, lab)),
                     "ARI_pincode": float(adjusted_rand_score(pin, lab)),
                     "NMI_pincode": float(normalized_mutual_info_score(pin, lab))}
    rnd = np.random.default_rng(SEED).integers(K, size=len(X))
    ext["random_labels"] = {"ARI_locality": float(adjusted_rand_score(loc_lab, rnd)),
                            "NMI_locality": float(normalized_mutual_info_score(loc_lab, rnd)),
                            "ARI_pincode": float(adjusted_rand_score(pin, rnd)),
                            "NMI_pincode": float(normalized_mutual_info_score(pin, rnd))}
    m["external"] = ext
    stab = {}
    for name, fit in [("kmeans_single_start", lambda s: cl.kmeans(X, K, n_init=1, seed=s)),
                      ("gmm_random_init", lambda s: cl.gmm_em(X, K, seed=s, init="random"))]:
        labs = [fit(s)["labels"] for s in range(5)]
        aris = [adjusted_rand_score(labs[a], labs[b]) for a in range(5) for b in range(a + 1, 5)]
        stab[name] = {"mean_pairwise_ARI": float(np.mean(aris)), "min": float(np.min(aris))}
    m["stability"] = stab

    # ---- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ks = tab.K[1:]
    axes[0].plot(ks, tab.kmeans_inertia[1:], "o-", color=BLUE)
    axes[0].set(xlabel="K", ylabel="distortion J (km²)", title="K-means elbow")
    axes[1].plot(tab.K, tab.BIC, "o-", color=BLUE, label="BIC")
    axes[1].plot(tab.K, tab.AIC, "s-", color=ORANGE, label="AIC")
    axes[1].axvline(K_knee, color=MUTED, ls=":", label=f"knee K = {K_knee}")
    axes[1].set(xlabel="K", ylabel="criterion (lower is better)", title="GMM: BIC and AIC")
    axes[1].legend(fontsize=8)
    ax2 = axes[2]
    ax2.plot(tab.K, tab.heldout_ll_per_vendor, "o-", color=AQUA, label="GMM held-out ll/vendor")
    ax2.set(xlabel="K", ylabel="held-out log-lik per vendor", title="Held-out likelihood & silhouette")
    ax3 = ax2.twinx()
    ax3.plot(ks, tab.silhouette[1:], "s--", color=PURPLE, label="k-means silhouette")
    ax3.set_ylabel("silhouette", color=PURPLE)
    ax3.grid(False)
    ax2.legend(loc="lower right", fontsize=8)
    ax3.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"U3-01 Choosing K: BIC still falls at K = {max(K_RANGE)}; its knee is "
                 f"K = {K_knee}",
                 fontweight="bold")
    save(fig, FIG, "U3_01_choose_k")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    it = np.arange(1, len(hist) + 1)
    axes[0].plot(it, hist, color=BLUE, lw=2, label="ln p(X | θ) after each E-step")
    axes[0].plot(np.arange(1, len(kl) + 1) + 0.5, np.array(g["bound_after_m"]), ".",
                 color=ORANGE, ms=4, label="bound L(q_old, θ_new) after M-step")
    axes[0].set(xlabel="EM iteration", ylabel="log-likelihood",
                title="Likelihood rises every iteration")
    axes[0].legend(fontsize=8)
    axes[1].semilogy(np.arange(1, len(kl) + 1), np.maximum(kl, 1e-12), color=PURPLE)
    axes[1].set(xlabel="EM iteration", ylabel="KL(q_old ‖ p(Z | X, θ_new))  (log scale)",
                title="The gap the next E-step closes")
    fig.suptitle(f"U3-02 Alternative view of EM: ln p = L(q, θ) + KL  (GMM, K = {K})",
                 fontweight="bold")
    save(fig, FIG, "U3_02_em_elbo_kl")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, lab, title in [(axes[0], km["labels"], f"K-means (hard), K = {K}"),
                           (axes[1], g["labels"], f"GMM (soft), K = {K}")]:
        cols = np.array(PALETTE)[lab % len(PALETTE)]
        if ax is axes[1]:
            from matplotlib.colors import to_rgba
            cols = np.array([to_rgba(c, a) for c, a in zip(cols, 0.15 + 0.85 * (rmax - rmax.min())
                                                               / (1 - rmax.min() + 1e-12))])
        ax.scatter(X[:, 0], X[:, 1], s=4, c=cols)
        ax.set(xlabel="east (km)", ylabel="north (km)", aspect="equal", title=title)
    for k in range(K):
        w, V = np.linalg.eigh(g["covs"][k])
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        from matplotlib.patches import Ellipse
        axes[1].add_patch(Ellipse(g["mus"][k], 2 * 2 * np.sqrt(w[1]), 2 * 2 * np.sqrt(w[0]),
                                  angle=ang, fill=False, color=INK2, lw=0.8))
    axes[0].scatter(km["centres"][:, 0], km["centres"][:, 1], marker="x", s=50, color=INK2)
    fig.suptitle(f"U3-03 Hard vs soft clusters of the {len(X)} vendors (faded = uncertain, "
                 "ellipses = 2σ)", fontweight="bold")
    save(fig, FIG, "U3_03_kmeans_vs_gmm_map")

    md += ["## 2–3 K-means and Gaussian mixtures on vendor locations (km)", "",
           md_table(pd.DataFrame([{
               "K": r["K"], "k-means J": f"{r.get('kmeans_inertia', float('nan')):.0f}",
               "silhouette": f"{r.get('silhouette', float('nan')):.4f}",
               "DB": f"{r.get('davies_bouldin', float('nan')):.3f}",
               "GMM ln p": f"{r['gmm_loglik']:.1f}", "params": r["gmm_params"],
               "BIC": f"{r['BIC']:.1f}", "AIC": f"{r['AIC']:.1f}",
               "held-out ll/vendor": f"{r['heldout_ll_per_vendor']:.4f}"} for r in rows])), "",
           f"- Minimum over K = 1..{max(K_RANGE)}: BIC → {K_bic}, AIC → {K_aic}, held-out "
           f"likelihood → {K_held}, silhouette (max) → {K_sil}. BIC still falling at the edge of "
           f"the range: {K_bic == max(K_RANGE)}.",
           f"- **K used = {K_knee}** (knee of the BIC curve: farthest point below the chord from "
           "K = 1 to the largest K).",
           f"- Own vs scikit-learn at K = {K_knee}: k-means J {km_runs[K_knee]['inertia']:.2f} vs "
           f"{sk_km.inertia_:.2f}; GMM ln p {gmm_runs[K_knee]['loglik']:.2f} vs "
           f"{sk_g.score(X) * len(X):.2f}.", "",
           f"## 4 Alternative view of EM (K = {K})", "",
           f"- {g['n_iter']} iterations; ln p(X) {hist[0]:.2f} → {hist[-1]:.2f}, never decreasing: "
           f"{m['em_view']['loglik_non_decreasing']}.",
           f"- KL gap after each M-step (closed by the next E-step): first five "
           + ", ".join(f"{x:.3g}" for x in kl[:5]) + f"; minimum {kl.min():.2e} (≥ 0 always).",
           "", "## 5 K-means as the limit of EM; hard vs soft", "",
           md_table(pd.DataFrame([{"shared variance ε (km²)": k,
                                   "ARI vs k-means": f4(d["ARI_vs_kmeans"]),
                                   "mean max responsibility": f4(d["mean_max_responsibility"])}
                                  for k, d in lim.items()])), "",
           f"- K-means vs full GMM partitions: ARI {ari_km_gmm:.4f}. "
           f"{m['hard_vs_soft']['vendors_max_resp_below_0.9']} vendors have max responsibility "
           f"< 0.9 and {m['hard_vs_soft']['vendors_max_resp_below_0.6']} have < 0.6 (border "
           "vendors k-means forces into one zone).", "",
           "## 6 Evaluating the clusterings", "",
           md_table(pd.DataFrame([{"labels": k, **{a: f4(b) for a, b in d.items()}}
                                  for k, d in ext.items()])), "",
           f"(locality = top {TOP_LOCALITIES} localities + 'other'; pincode = "
           f"{pin.nunique()} pincodes.)", "",
           md_table(pd.DataFrame([{"method (5 seeds)": k, "mean pairwise ARI": f4(d[
               "mean_pairwise_ARI"]), "min ARI": f4(d["min"])} for k, d in stab.items()])), ""]
    return g, K


# ---------------------------------------------------------------- 7 PCA / FA
def retail_mix(v, X, md, m):
    cats = sorted({c for s in v.cats for c in s})
    cell = np.floor(X / CELL_KM).astype(int)
    key = pd.Series([f"{a}_{b}" for a, b in cell])
    counts = key.value_counts()
    keep = counts[counts >= MIN_CELL_VENDORS].index
    rows, idx = [], []
    for k in keep:
        sel = v[key.values == k]
        c = Counter(cc for s in sel.cats for cc in s)
        rows.append([c.get(x, 0) / len(sel) for x in cats])
        idx.append(k)
    P = np.array(rows)
    Z = (P - P.mean(0)) / P.std(0)
    mu, vals, vecs = pca_eig(Z)
    skp = PCA().fit(Z)
    ratio = vals / vals.sum()
    cum = np.cumsum(ratio)
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    kaiser = int((vals > 1).sum())
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    qs = list(range(0, 16))
    ppca_cv, fa_cv = [], []
    for q in qs:
        lp, lf = [], []
        for tr, te in kf.split(Z):
            mu_t, va, ve = pca_eig(Z[tr])
            W, s2 = ppca_params(va, ve, q)
            lp.append(ppca_loglik(Z[te], mu_t, W, s2))
            if q >= 1:
                lf.append(FactorAnalysis(q, random_state=SEED).fit(Z[tr]).score(Z[te]))
        ppca_cv.append(float(np.mean(lp)))
        fa_cv.append(float(np.mean(lf)) if lf else None)
    q_ppca = qs[int(np.argmax(ppca_cv))]
    fa_valid = [(q, s) for q, s in zip(qs, fa_cv) if s is not None]
    q_fa = max(fa_valid, key=lambda t: t[1])[0]
    q_minka = int(PCA("mle").fit(Z).n_components_)
    load = pd.DataFrame(vecs[:, :3], index=cats, columns=["PC1", "PC2", "PC3"])
    tops = {pc: {"positive": load[pc].nlargest(4).round(3).to_dict(),
                 "negative": load[pc].nsmallest(4).round(3).to_dict()} for pc in load}
    m["pca"] = {"cells": len(P), "cell_km": CELL_KM, "min_vendors_per_cell": MIN_CELL_VENDORS,
                "vendors_covered": int(counts[keep].sum()), "features": len(cats),
                "eigenvalues": vals.tolist(), "explained_ratio": ratio.tolist(),
                "max_abs_diff_eig_vs_sklearn":
                    float(np.abs(vals - skp.explained_variance_ * (len(Z) - 1) / len(Z)).max()),
                "dims_for_90pct": d90, "kaiser_dims": kaiser, "ppca_cv_ll": ppca_cv,
                "fa_cv_ll": fa_cv, "q_by_ppca_cv": q_ppca, "q_by_fa_cv": q_fa,
                "q_by_minka_mle": q_minka, "loadings_top": tops}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(1, len(vals) + 1)
    axes[0].bar(x, ratio, color=BLUE)
    axes[0].plot(x, cum, "o-", color=ORANGE, ms=3, label="cumulative")
    axes[0].axhline(0.9, color=MUTED, ls=":")
    axes[0].set(xlabel="component", ylabel="share of variance",
                title=f"Scree: 90% needs {d90} of {len(cats)}")
    axes[0].legend(fontsize=8)
    axes[1].plot(qs, ppca_cv, "o-", color=BLUE, label="PPCA")
    axes[1].plot([q for q, _ in fa_valid], [s for _, s in fa_valid], "s-", color=ORANGE,
                 label="factor analysis")
    axes[1].axvline(q_ppca, color=BLUE, ls=":")
    axes[1].axvline(q_fa, color=ORANGE, ls=":")
    axes[1].set(xlabel="latent dimensions q", ylabel=f"{N_FOLDS}-fold held-out ll per cell",
                title="Choosing q by held-out likelihood")
    axes[1].legend(fontsize=8)
    S = Z @ vecs[:, :2]
    axes[2].scatter(S[:, 0], S[:, 1], s=14, color=BLUE, alpha=0.7)
    for c in load.PC1.abs().nlargest(3).index.tolist() + load.PC2.abs().nlargest(3).index.tolist():
        i = cats.index(c)
        tip = (vecs[i, 0] * 6, vecs[i, 1] * 6)
        axes[2].annotate("", xy=tip, xytext=(0, 0),
                         arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 0.8})
        axes[2].text(*tip, c, fontsize=7, color=ORANGE)
    axes[2].set(xlabel="PC1", ylabel="PC2", title="Cells on the first two components")
    fig.suptitle(f"U3-04 PCA of the retail mix of {len(P)} grid cells ({CELL_KM:.0f} km, "
                 f"≥ {MIN_CELL_VENDORS} vendors)", fontweight="bold")
    save(fig, FIG, "U3_04_pca_choose_q")

    md += ["## 7 PCA and factor analysis: retail mix of grid cells", "",
           f"- {len(P)} cells of {CELL_KM:.0f} × {CELL_KM:.0f} km with ≥ {MIN_CELL_VENDORS} "
           f"vendors ({int(counts[keep].sum())} vendors). Feature = share of the cell's vendors "
           f"in each of the {len(cats)} categories, standardised.",
           f"- Own eigendecomposition vs scikit-learn: max |Δ eigenvalue| "
           f"{m['pca']['max_abs_diff_eig_vs_sklearn']:.2e}.",
           f"- First 5 components explain " + ", ".join(f"{r:.4f}" for r in ratio[:5])
           + f" (cumulative {cum[4]:.4f}).", "",
           md_table(pd.DataFrame({"rule for q": ["90% cumulative variance",
                                                 "Kaiser (eigenvalue > 1)",
                                                 f"PPCA {N_FOLDS}-fold held-out likelihood",
                                                 f"FA {N_FOLDS}-fold held-out likelihood",
                                                 "Minka's Bayesian evidence (sklearn 'mle')"],
                                  "chosen q": [d90, kaiser, q_ppca, q_fa, q_minka]})), "",
           "Held-out log-likelihood per cell:", "",
           md_table(pd.DataFrame({"q": qs, "PPCA": [f"{x:.3f}" for x in ppca_cv],
                                  "FA": ["–" if x is None else f"{x:.3f}" for x in fa_cv]})), "",
           "Top loadings:", ""]
    for pc, d in tops.items():
        md.append(f"- **{pc}** + " + ", ".join(f"{k} {x}" for k, x in d["positive"].items())
                  + " | − " + ", ".join(f"{k} {x}" for k, x in d["negative"].items()))
    md.append("")
    return P, cats


# ---------------------------------------------------------------- 8 hierarchical
def hierarchical(v, P, cats, md, m):
    prof = P.T                                     # categories x cells
    D_corr = pdist(prof, metric="correlation")
    memb = np.array([[c in s for s in v.cats] for c in cats])
    D_jac = pdist(memb, metric="jaccard")
    res = {}
    for name, D in [("correlation_of_cell_shares", D_corr), ("jaccard_of_vendor_sets", D_jac)]:
        res[name] = {}
        for method in ["single", "complete", "average", "ward"]:
            Z = linkage(D, method=method)
            res[name][method] = float(cophenet(Z, D)[0])
    best = max(res["correlation_of_cell_shares"], key=res["correlation_of_cell_shares"].get)
    Zb = linkage(D_corr, method=best)
    groups = fcluster(Zb, 5, criterion="maxclust")
    g_list = {int(gid): [c for c, g in zip(cats, groups) if g == gid] for gid in sorted(set(groups))}
    m["hierarchical"] = {"cophenetic": res, "best_linkage_correlation": best,
                         "five_groups": g_list}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dendrogram(Zb, labels=cats, orientation="right", ax=axes[0], color_threshold=Zb[-4, 2],
               leaf_font_size=8, above_threshold_color=MUTED)
    axes[0].set(title=f"Spatial co-location (1 − correlation), {best} linkage",
                xlabel="merge height")
    axes[0].grid(False)
    Zj = linkage(D_jac, method="average")
    dendrogram(Zj, labels=cats, orientation="right", ax=axes[1], leaf_font_size=8,
               color_threshold=0.97, above_threshold_color=MUTED)
    axes[1].set(title="Shared vendors (Jaccard distance), average linkage", xlabel="merge height")
    axes[1].grid(False)
    fig.suptitle("U3-05 Hierarchical clustering of the 23 categories", fontweight="bold")
    save(fig, FIG, "U3_05_dendrograms")

    md += ["## 8 Hierarchical clustering of the 23 categories", "",
           "Cophenetic correlation (how well the tree preserves the original distances):", "",
           md_table(pd.DataFrame([{"dissimilarity": k, **{a: f4(b) for a, b in d.items()}}
                                  for k, d in res.items()])), "",
           f"Five groups from the {best}-linkage tree on spatial correlation:", ""]
    md += [f"- group {k}: {', '.join(c)}" for k, c in g_list.items()] + [""]


# ---------------------------------------------------------------- 9 zone feature for matching
def zone_feature(g, lat0, lon0, md, m):
    df = lr.load_annotations()
    df = df[df[lr.TARGET].notna()].copy()
    df["y"] = df[lr.TARGET].astype(float)
    reqs = pd.read_csv("data/processed/requirements.csv", keep_default_na=False)
    df = df.merge(reqs[["requirement_id", "buyer_lat", "buyer_lon"]], on="requirement_id",
                  how="left", validate="many_to_one")
    v = L.load_vendors().set_index("vendor_id")
    vb = to_km(df.buyer_lat.astype(float), df.buyer_lon.astype(float), lat0, lon0)
    vv = to_km(v.loc[df.vendor_id, "lat"], v.loc[df.vendor_id, "lon"], lat0, lon0)
    Rb, _ = cl.e_step(vb, g["pi"], g["mus"], g["covs"])
    Rv, _ = cl.e_step(vv, g["pi"], g["mus"], g["covs"])
    df["same_zone"] = (Rb.argmax(1) == Rv.argmax(1)).astype(float)
    df["zone_overlap"] = (Rb * Rv).sum(1)
    train, test = lr.split_by_requirement(df)
    out = {}
    for name, extra in [("phase1_features", []), ("+ same_zone (hard GMM)", ["same_zone"]),
                        ("+ zone_overlap (soft GMM)", ["zone_overlap"])]:
        Xtr = np.column_stack([design_matrix(train)] + [train[c].values for c in extra])
        Xte = np.column_stack([design_matrix(test)] + [test[c].values for c in extra])
        beta = lr.fit_ols(Xtr, train.y.values)["beta"]
        out[name] = {"train_R2": lr.fit_metrics(Xtr, train.y.values, beta)["R2"],
                     "test_R2": lr.fit_metrics(Xte, test.y.values, beta)["R2"],
                     "test_RMSE": lr.fit_metrics(Xte, test.y.values, beta)["RMSE"],
                     "extra_coef": float(beta[-1]) if extra else None}
    out["share_same_zone"] = float(df.same_zone.mean())
    out["corr_same_zone_within_radius"] = float(np.corrcoef(df.same_zone,
                                                            df.within_radius.astype(float))[0, 1])
    m["zone_feature_regression"] = out
    md += ["## 9 Does a GMM zone feature help the Phase 1 match-score model?", "",
           "Same split by requirement as Phase 1 (24 train / 6 test requirements).", "",
           md_table(pd.DataFrame([{"model": k, "train R²": f4(d["train_R2"]),
                                   "test R²": f4(d["test_R2"]), "test RMSE": f"{d['test_RMSE']:.3f}",
                                   "coef of new feature": "–" if d["extra_coef"] is None
                                   else f"{d['extra_coef']:.3f}"}
                                  for k, d in out.items() if isinstance(d, dict)])), "",
           f"- {out['share_same_zone']:.4f} of pairs have buyer and vendor in the same zone; "
           f"correlation with within_radius {out['corr_same_zone_within_radius']:.4f}.", ""]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    v, X, lat0, lon0 = load()
    md = ["# Unit III numbers (generated by `python src/pbl2_unit3.py`; do not edit by hand)", "",
          f"{len(v)} vendors projected to km around ({lat0:.4f}, {lon0:.4f}). Seed {SEED}.", ""]
    m = {"n_vendors": len(v), "projection_origin": [lat0, lon0]}
    dissimilarity(v, X, md, m)
    g, K = kmeans_gmm(v, X, md, m)
    P, cats = retail_mix(v, X, md, m)
    hierarchical(v, P, cats, md, m)
    zone_feature(g, lat0, lon0, md, m)
    write_json(METRICS, m)
    (OUT / "unit3_numbers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("K by BIC:", m["choose_K"]["K_by_BIC"], "| q by PPCA:", m["pca"]["q_by_ppca_cv"],
          "| zone R2:", {k: round(d["test_R2"], 4) for k, d in
                         m["zone_feature_regression"].items() if isinstance(d, dict)})
    print(f"wrote {METRICS}, {OUT / 'unit3_numbers.md'}, figures U3_*")


if __name__ == "__main__":
    main()
