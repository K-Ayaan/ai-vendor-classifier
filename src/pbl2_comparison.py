"""PBL-2 comparative analysis (section 4).

Usage: python src/pbl2_comparison.py   (after pbl2_unit3.py and pbl2_unit4.py)
Reads reports/pbl2_unit3_metrics.json, reports/pbl2_unit4_metrics.json and data/ (read-only).
Writes ONLY pbl-2/04_comparative_analysis/figures/C*.png and comparison_numbers.md.

C01 six clustering methods on the same vendor locations and the same K
C02 address-role decoders on the gold set
C03 sequence models: held-out log-likelihood per symbol
C04 GMM zone feature for the match-score regression, leave-one-requirement-out CV (30 folds)
C05 trackers on the simulated trajectory
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import linear_regression_mle as lr  # noqa: E402
import logistic_regression_classification as L  # noqa: E402
import pbl2_unit3 as U3  # noqa: E402
from latent import clustering as cl  # noqa: E402
from matching.features import design_matrix  # noqa: E402
from pbl2_style import (AQUA, BLUE, LIGHT, MUTED, ORANGE, PBL2, PURPLE, md_table, plt,  # noqa: E402
                        save, to_km)

OUT = PBL2 / "04_comparative_analysis"
FIG = OUT / "figures"


def f4(x):
    return f"{x:.4f}"


def c01_clustering(md, K):
    v, X, lat0, lon0 = U3.load()
    loc = v.locality.fillna("(none)")
    top = loc.value_counts().index[:U3.TOP_LOCALITIES]
    loc_lab = loc.where(loc.isin(top), "other")
    runs = {}
    t = time.time()
    runs["k-means (Lloyd, 10 starts)"] = (cl.kmeans(X, K, n_init=10, seed=U3.SEED)["labels"],
                                          time.time() - t)
    t = time.time()
    runs[f"GMM by EM ({U3.EM_RESTARTS} starts)"] = (U3.best_gmm(X, K)["labels"], time.time() - t)
    for method in ["single", "complete", "average", "ward"]:
        t = time.time()
        Z = linkage(X, method=method)
        runs[f"agglomerative, {method} linkage"] = (fcluster(Z, K, criterion="maxclust") - 1,
                                                    time.time() - t)
    rows = []
    for name, (lab, sec) in runs.items():
        sizes = np.bincount(lab)
        rows.append({"method": name, "silhouette": silhouette_score(X, lab),
                     "Davies-Bouldin": davies_bouldin_score(X, lab),
                     "Calinski-Harabasz": calinski_harabasz_score(X, lab),
                     "ARI locality": adjusted_rand_score(loc_lab, lab),
                     "NMI locality": normalized_mutual_info_score(loc_lab, lab),
                     "largest cluster share": sizes.max() / len(lab),
                     "smallest cluster": int(sizes.min()), "seconds": sec})
    tab = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = [n.replace("agglomerative, ", "agg. ").replace(" linkage", "") for n in tab.method]
    for ax, col, title, good in [(axes[0], "silhouette", "Silhouette (higher better)", "max"),
                                 (axes[1], "Davies-Bouldin", "Davies-Bouldin (lower better)",
                                  "min"),
                                 (axes[2], "ARI locality", "ARI vs locality (higher better)",
                                  "max")]:
        vals = tab[col].values
        best = vals.argmax() if good == "max" else vals.argmin()
        ax.barh(names, vals, color=[ORANGE if i == best else BLUE for i in range(len(vals))])
        for i, x in enumerate(vals):
            ax.text(x, i, f" {x:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"C01 Six clustering methods on the same {len(X)} vendor locations, K = {K}",
                 fontweight="bold")
    save(fig, FIG, "C01_clustering_methods")

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    from pbl2_style import PALETTE
    for ax, (name, (lab, _)) in zip(axes.ravel(), runs.items()):
        ax.scatter(X[:, 0], X[:, 1], s=2, c=np.array(PALETTE)[lab % len(PALETTE)])
        ax.set(aspect="equal", title=name, xticks=[], yticks=[])
    fig.suptitle("C01b The partitions on the map", fontweight="bold")
    save(fig, FIG, "C01b_clustering_maps")

    md += [f"## C01 Clustering methods (K = {K}, same data)", "",
           md_table(pd.DataFrame([{"method": r["method"], "silhouette": f4(r["silhouette"]),
                                   "DB": f"{r['Davies-Bouldin']:.3f}",
                                   "CH": f"{r['Calinski-Harabasz']:.0f}",
                                   "ARI loc.": f4(r["ARI locality"]),
                                   "NMI loc.": f4(r["NMI locality"]),
                                   "largest share": f4(r["largest cluster share"]),
                                   "smallest": r["smallest cluster"],
                                   "s": f"{r['seconds']:.2f}"} for r in rows])), ""]
    return tab


def c02_c03_c05(md, m4):
    d = m4["decoding_on_gold"]
    order = ["majority_state", "rules", "hmm_baum_welch_random_best",
             "hmm_baum_welch_from_supervised", "hmm_supervised_posterior",
             "hmm_supervised_viterbi"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(order))
    ax.barh(y - 0.2, [d[k]["segment_accuracy"] for k in order], 0.4, color=BLUE,
            label="part accuracy")
    ax.barh(y + 0.2, [d[k]["address_exact_match"] for k in order], 0.4, color=ORANGE,
            label="whole address exactly right")
    for i, k in enumerate(order):
        ax.text(d[k]["segment_accuracy"], i - 0.2, f" {d[k]['segment_accuracy']:.3f}",
                va="center", fontsize=8)
        ax.text(d[k]["address_exact_match"], i + 0.2, f" {d[k]['address_exact_match']:.3f}",
                va="center", fontsize=8)
    ax.set_yticks(y, [k.replace("_", " ") for k in order], fontsize=8)
    ax.set(xlim=(0, 1.1), xlabel=f"on the {m4['data']['gold_addresses']}-address gold set",
           title="C02 Address-role decoders")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, FIG, "C02_address_decoders")
    md += ["## C02 Address-role decoders (gold set)", "",
           md_table(pd.DataFrame([{"decoder": k, "part accuracy": f4(d[k]["segment_accuracy"]),
                                   "macro F1": f4(d[k]["macro_f1"]),
                                   "address exact": f4(d[k]["address_exact_match"])}
                                  for k in order])), ""]

    mk = m4["markov"]
    bw = m4["baum_welch"]
    seq = {"order 0 (i.i.d.)": mk["order0_heldout_ll_per_symbol"],
           "order 1 Markov chain": mk["order1_heldout_ll_per_symbol"],
           "HMM, supervised MLE": mk["hmm_supervised_heldout_ll_per_symbol"],
           "HMM, Baum-Welch from supervised": bw["from_supervised"]["heldout_ll_per_symbol"],
           "HMM, Baum-Welch best random": bw["best_random_heldout_ll_per_symbol"]}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(seq), list(seq.values()), color=[LIGHT, BLUE, AQUA, PURPLE, ORANGE])
    for i, x in enumerate(seq.values()):
        ax.text(x, i, f" {x:.4f}", va="center", fontsize=8, ha="right", color="white")
    ax.set(xlabel="held-out log-likelihood per symbol (higher = better)",
           title="C03 Sequence models on the 923 test addresses")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    save(fig, FIG, "C03_sequence_models")
    md += ["## C03 Sequence models (held-out log-likelihood per symbol)", "",
           md_table(pd.DataFrame({"model": list(seq), "test ll/symbol":
                                  [f4(x) for x in seq.values()],
                                  "perplexity": [f"{np.exp(-x):.3f}" for x in seq.values()]})),
           ""]

    r = m4["lds"]["rmse_m"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(list(r), list(r.values()), color=[LIGHT, MUTED, ORANGE, BLUE])
    for i, x in enumerate(r.values()):
        ax.text(x, i, f" {x:.2f} m", va="center", fontsize=8)
    ax.set(xlabel="position RMSE (m), lower is better",
           title="C05 Trackers on the SIMULATED trajectory")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    save(fig, FIG, "C05_trackers")
    md += ["## C05 Trackers (SIMULATED trajectory)", "",
           md_table(pd.DataFrame({"estimate": list(r), "RMSE (m)": [f"{x:.2f}" for x in
                                                                     r.values()]})), ""]


def c04_zone_cv(md, K):
    v, X, lat0, lon0 = U3.load()
    g = U3.best_gmm(X, K)
    df = lr.load_annotations()
    df = df[df[lr.TARGET].notna()].copy()
    df["y"] = df[lr.TARGET].astype(float)
    reqs = pd.read_csv("data/processed/requirements.csv", keep_default_na=False)
    df = df.merge(reqs[["requirement_id", "buyer_lat", "buyer_lon"]], on="requirement_id")
    vv = L.load_vendors().set_index("vendor_id")
    Rb, _ = cl.e_step(to_km(df.buyer_lat.astype(float), df.buyer_lon.astype(float), lat0, lon0),
                      g["pi"], g["mus"], g["covs"])
    Rv, _ = cl.e_step(to_km(vv.loc[df.vendor_id, "lat"], vv.loc[df.vendor_id, "lon"], lat0,
                            lon0), g["pi"], g["mus"], g["covs"])
    df["same_zone"] = (Rb.argmax(1) == Rv.argmax(1)).astype(float)
    df["zone_overlap"] = (Rb * Rv).sum(1)
    models = {"phase1_features": [], "+ same_zone": ["same_zone"],
              "+ zone_overlap": ["zone_overlap"]}
    preds = {k: np.empty(len(df)) for k in models}
    for rid in df.requirement_id.unique():
        te = (df.requirement_id == rid).values
        for k, extra in models.items():
            Xtr = np.column_stack([design_matrix(df[~te])] + [df.loc[~te, c] for c in extra])
            Xte = np.column_stack([design_matrix(df[te])] + [df.loc[te, c] for c in extra])
            beta = np.linalg.lstsq(Xtr, df.y[~te].values, rcond=None)[0]
            preds[k][te] = Xte @ beta
    y = df.y.values
    out = {}
    for k, p in preds.items():
        err = p - y
        out[k] = {"R2": 1 - (err @ err) / ((y - y.mean()) @ (y - y.mean())),
                  "RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err))}
    per_req = pd.DataFrame({k: [np.mean(np.abs(preds[k][(df.requirement_id == r).values]
                                               - y[(df.requirement_id == r).values]))
                                for r in df.requirement_id.unique()] for k in models})
    diff = per_req["phase1_features"] - per_req["+ same_zone"]
    from scipy.stats import wilcoxon
    w = wilcoxon(per_req["phase1_features"], per_req["+ same_zone"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].bar(list(out), [o["R2"] for o in out.values()], color=[LIGHT, ORANGE, BLUE])
    for i, o in enumerate(out.values()):
        axes[0].text(i, o["R2"], f"{o['R2']:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set(ylim=(0.6, 0.85), ylabel="pooled R² (30 held-out requirements)",
                title="Leave-one-requirement-out R²")
    axes[0].grid(axis="x", visible=False)
    axes[1].bar(range(len(diff)), np.sort(diff.values)[::-1],
                color=[AQUA if x > 0 else ORANGE for x in np.sort(diff.values)[::-1]])
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set(xlabel="requirement (sorted)", ylabel="MAE without − with same_zone",
                title=f"Per requirement: {int((diff > 0).sum())} improve, "
                      f"{int((diff < 0).sum())} worsen (Wilcoxon p = {w.pvalue:.3f})")
    fig.suptitle(f"C04 Does a GMM zone (K = {K}) help the match-score model?", fontweight="bold")
    save(fig, FIG, "C04_zone_feature_cv")
    md += ["## C04 Zone feature, leave-one-requirement-out CV (30 folds)", "",
           md_table(pd.DataFrame([{"model": k, "R²": f4(o["R2"]), "RMSE": f"{o['RMSE']:.3f}",
                                   "MAE": f"{o['MAE']:.3f}"} for k, o in out.items()])), "",
           f"- same_zone lowers the per-requirement MAE for {int((diff > 0).sum())} of 30 "
           f"requirements and raises it for {int((diff < 0).sum())}; mean change "
           f"{diff.mean():.3f} points; Wilcoxon signed-rank p = {w.pvalue:.4f}.", ""]
    return out, float(w.pvalue)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    m3 = json.loads(Path("reports/pbl2_unit3_metrics.json").read_text(encoding="utf-8"))
    m4 = json.loads(Path("reports/pbl2_unit4_metrics.json").read_text(encoding="utf-8"))
    K = m3["choose_K"]["K_by_BIC_knee_USED"]
    md = ["# Comparison numbers (generated by `python src/pbl2_comparison.py`; do not edit by "
          "hand)", ""]
    c01_clustering(md, K)
    c02_c03_c05(md, m4)
    out, p = c04_zone_cv(md, K)
    pca = m3["pca"]
    md += ["## C06 Rules for the number of latent dimensions (from `pbl2_unit3.py`)", "",
           md_table(pd.DataFrame({"rule": ["90% variance", "Kaiser", "PPCA held-out ll",
                                           "FA held-out ll", "Minka evidence"],
                                  "q": [pca["dims_for_90pct"], pca["kaiser_dims"],
                                        pca["q_by_ppca_cv"], pca["q_by_fa_cv"],
                                        pca["q_by_minka_mle"]]})), ""]
    (OUT / "comparison_numbers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("zone CV:", {k: round(o["R2"], 4) for k, o in out.items()}, "wilcoxon p", round(p, 4))
    print(f"wrote {OUT / 'comparison_numbers.md'} and figures C01-C05")


if __name__ == "__main__":
    main()
