"""PBL-2 Unit IV: Markov models, HMMs and a linear dynamical system.

Usage: python src/pbl2_unit4.py
Reads data/processed/vendors.csv and data/annotation/address_gold.csv (read-only). Writes
reports/pbl2_unit4_metrics.json and pbl-2/03_unit_topics_implementation/figures/U4_*.png and
pbl-2/03_unit_topics_implementation/unit4_numbers.md.

Part A (real data): each address is a sequence of comma-separated parts; the hidden state is the
part's role (PREMISE, STREET, AREA, CITY, STATE).
  1. Markov models of the symbol sequences: order 0 (i.i.d.) vs order 1, held-out log-likelihood
  2. HMM by supervised maximum likelihood (counts) from rule-based labels on the TRAIN split
  3. HMM by Baum-Welch (EM), from the supervised start and from random starts
  4. Forward-backward with scaling factors, checked against brute-force enumeration
  5. Viterbi (max-product) vs posterior (sum-product) decoding, scored on the 60-address gold set
Part B (SIMULATED data): a delivery agent driving between real vendor locations, tracked from
noisy GPS with a Kalman filter and RTS smoother; noise levels fitted by maximum likelihood.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logistic_regression_classification as L  # noqa: E402
import pbl2_address as AD  # noqa: E402
from latent import hmm, lds  # noqa: E402
from pbl2_style import (AQUA, BLUE, INK2, LIGHT, MUTED, ORANGE, PBL2, PURPLE, SEED,  # noqa: E402
                        md_table, plt, save, to_km, write_json)

OUT = PBL2 / "03_unit_topics_implementation"
FIG = OUT / "figures"
GOLD = Path("data/annotation/address_gold.csv")
METRICS = Path("reports/pbl2_unit4_metrics.json")
K, M = len(AD.STATES), len(AD.SYMBOLS)
PSEUDO = 1.0           # add-one (Dirichlet) smoothing for the supervised counts
N_RANDOM_STARTS = 5
GPS_SIGMA_KM = 0.020   # simulated GPS noise: 20 m
SPEED_KMH = 25.0
DT_S = 2.0


def symbol_markov_loglik(train, test):
    """Order-0 and order-1 Markov models of the symbol sequences (add-one smoothed MLE).
    Returns held-out log-likelihood per symbol for each."""
    uni = np.ones(M)
    start = np.ones(M)
    trans = np.ones((M, M))
    for x in train:
        np.add.at(uni, x, 1)
        start[x[0]] += 1
        np.add.at(trans, (x[:-1], x[1:]), 1)
    uni, start, trans = uni / uni.sum(), start / start.sum(), hmm.normalise(trans)
    n = sum(len(x) for x in test)
    ll0 = sum(np.log(uni[x]).sum() for x in test)
    ll1 = sum(np.log(start[x[0]]) + np.log(trans[x[:-1], x[1:]]).sum() for x in test)
    return ll0 / n, ll1 / n, trans


def decode_scores(gold, pred):
    g, p = np.concatenate(gold), np.concatenate(pred)
    return {"segment_accuracy": float((g == p).mean()),
            "macro_f1": float(f1_score(g, p, average="macro", labels=range(K))),
            "per_state_f1": dict(zip(AD.STATES, map(float, f1_score(
                g, p, average=None, labels=range(K))))),
            "address_exact_match": float(np.mean([np.array_equal(a, b)
                                                  for a, b in zip(gold, pred)]))}


def match_states(seqs, labels, params):
    """Unsupervised states have no names: map each to a role with the Hungarian algorithm,
    maximising agreement between Viterbi paths and the TRAIN rule labels."""
    C = np.zeros((K, K))
    for x, z in zip(seqs, labels):
        path, _ = hmm.viterbi(x, *params)
        np.add.at(C, (path, z), 1)
    rows, cols = linear_sum_assignment(-C)
    perm = np.empty(K, dtype=int)
    perm[rows] = cols
    return perm


def part_a(md, metrics):
    v = L.load_vendors()
    train, test = L.split(v)
    n_empty = int((train.address.map(len) == 0).sum() + (test.address.map(len) == 0).sum())
    train = train[train.address.map(lambda a: len(AD.segments(a)) > 0)].reset_index(drop=True)
    test = test[test.address.map(lambda a: len(AD.segments(a)) > 0)].reset_index(drop=True)
    towns = AD.town_gazetteer(train.address)
    tr_x = [AD.encode(a, towns) for a in train.address]
    tr_z = [AD.rule_labels(a, towns) for a in train.address]
    te_x = [AD.encode(a, towns) for a in test.address]
    gold = pd.read_csv(GOLD, keep_default_na=False)
    assert set(gold.vendor_id) <= set(test.vendor_id), "gold addresses must come from test"
    g_x = [AD.encode(a, towns) for a in gold.address]
    g_z = [AD.parse_label_string(s) for s in gold.gold_labels]
    metrics["data"] = {"train_addresses": len(tr_x), "train_segments": int(sum(map(len, tr_x))),
                       "test_addresses": len(te_x), "test_segments": int(sum(map(len, te_x))),
                       "empty_addresses_dropped": n_empty, "gold_addresses": len(g_x), "gold_segments": int(sum(map(len, g_x))),
                       "towns": sorted(towns), "symbols": AD.SYMBOLS, "states": AD.STATES}

    # ---------------------------------------------------------------- 1 Markov models
    ll0, ll1, sym_trans = symbol_markov_loglik(tr_x, te_x)
    # ---------------------------------------------------------------- 2 supervised MLE
    sup = hmm.supervised_mle(tr_x, tr_z, K, M, pseudo=PSEUDO)
    sup_tr_ll = sum(hmm.loglik(x, *sup) for x in tr_x)
    sup_te_ll = sum(hmm.loglik(x, *sup) for x in te_x)
    n_te = sum(map(len, te_x))
    metrics["markov"] = {"order0_heldout_ll_per_symbol": ll0,
                         "order1_heldout_ll_per_symbol": ll1,
                         "hmm_supervised_heldout_ll_per_symbol": sup_te_ll / n_te}
    # stationary distribution of the role chain (left eigenvector of A for eigenvalue 1)
    w, vecs = np.linalg.eig(sup[1].T)
    stat = np.real(vecs[:, np.argmin(np.abs(w - 1))])
    stat = stat / stat.sum()

    # ---------------------------------------------------------------- 3 Baum-Welch
    t0 = time.time()
    bw_sup = hmm.baum_welch(tr_x, *sup, max_iter=200)
    bw_sup_time = time.time() - t0
    rng = np.random.default_rng(SEED)
    starts = []
    for _ in range(N_RANDOM_STARTS):
        starts.append(hmm.baum_welch(tr_x, *hmm.random_params(K, M, rng), max_iter=300))
    best = max(starts, key=lambda r: r[3][-1])
    perm = match_states(tr_x, tr_z, best[:3])
    metrics["baum_welch"] = {
        "from_supervised": {"iterations": len(bw_sup[3]), "train_ll_start": bw_sup[3][0],
                            "train_ll_end": bw_sup[3][-1], "seconds": bw_sup_time,
                            "heldout_ll_per_symbol":
                                sum(hmm.loglik(x, *bw_sup[:3]) for x in te_x) / n_te},
        "random_starts": [{"iterations": len(r[3]), "train_ll_end": r[3][-1]} for r in starts],
        "best_random_heldout_ll_per_symbol":
            sum(hmm.loglik(x, *best[:3]) for x in te_x) / n_te,
        "best_random_state_to_role": {f"state_{i}": AD.STATES[perm[i]] for i in range(K)},
        "supervised_train_ll": sup_tr_ll,
        "monotone_from_supervised": bool(np.all(np.diff(bw_sup[3]) >= -1e-6 * abs(bw_sup[3][0]))),
    }

    # ---------------------------------------------------------------- 4 scaling + sum-product
    long_seq = np.concatenate(te_x)
    unscaled = hmm.forward_unscaled(long_seq, *sup)
    _, c = hmm.forward(long_seq, *sup)
    log10_partial = np.cumsum(np.log10(c))
    underflow_at = int(np.argmax(log10_partial < np.log10(np.finfo(float).tiny))) + 1
    short = next(x for x in g_x if len(x) == 7)
    px, marg, best_path = hmm.brute_force(short, *sup)
    gamma, _, ll_short = hmm.posteriors(short, *sup)
    vpath, _ = hmm.viterbi(short, *sup)
    metrics["scaling"] = {"long_sequence_length": int(len(long_seq)),
                          "unscaled_forward_result": unscaled,
                          "scaled_log10_likelihood": float(log10_partial[-1]),
                          "float64_underflow_after_symbols": underflow_at,
                          "c_t_min": float(c.min()), "c_t_max": float(c.max())}
    metrics["sum_product_check"] = {
        "sequence_length": 7, "paths_enumerated": K ** 7,
        "abs_diff_loglik": float(abs(ll_short - np.log(px))),
        "max_abs_diff_marginals": float(np.abs(gamma - marg).max()),
        "viterbi_equals_brute_force": bool(np.array_equal(vpath, best_path))}

    # ---------------------------------------------------------------- 5 decoding on gold
    majority = int(np.bincount(np.concatenate(tr_z), minlength=K).argmax())
    decoders = {
        "majority_state": [np.full(len(x), majority) for x in g_x],
        "rules": [AD.rule_labels(a, towns) for a in gold.address],
        "hmm_supervised_viterbi": [hmm.viterbi(x, *sup)[0] for x in g_x],
        "hmm_supervised_posterior": [hmm.posteriors(x, *sup)[0].argmax(1) for x in g_x],
        "hmm_baum_welch_from_supervised": [hmm.viterbi(x, *bw_sup[:3])[0] for x in g_x],
        "hmm_baum_welch_random_best": [perm[hmm.viterbi(x, *best[:3])[0]] for x in g_x],
    }
    metrics["decoding_on_gold"] = {k: decode_scores(g_z, p) for k, p in decoders.items()}
    rules_vs_vit = np.concatenate(decoders["rules"]) != np.concatenate(
        decoders["hmm_supervised_viterbi"])
    gz = np.concatenate(g_z)
    metrics["rules_vs_viterbi_disagreements"] = {
        "segments": int(rules_vs_vit.sum()),
        "viterbi_right": int((np.concatenate(decoders["hmm_supervised_viterbi"])
                              [rules_vs_vit] == gz[rules_vs_vit]).sum()),
        "rules_right": int((np.concatenate(decoders["rules"])[rules_vs_vit]
                            == gz[rules_vs_vit]).sum())}
    # path-level disagreement examples for the README
    ex = []
    for (_, r), z, pr, pv in zip(gold.iterrows(), g_z, decoders["rules"],
                                 decoders["hmm_supervised_viterbi"]):
        if not np.array_equal(pr, pv):
            segs = AD.segments(r.address)
            for i in np.flatnonzero(pr != pv):
                ex.append((r.vendor_id, segs[i], AD.STATES[z[i]], AD.STATES[pr[i]],
                           AD.STATES[pv[i]]))

    # ---------------------------------------------------------------- anomaly: per-part ll
    te_scores = np.array([hmm.loglik(x, *sup) / len(x) for x in te_x])
    worst = np.argsort(te_scores)[:5]
    metrics["anomaly"] = {"heldout_ll_per_part_median": float(np.median(te_scores)),
                          "lowest": [{"vendor_id": test.vendor_id.iloc[i],
                                      "ll_per_part": float(te_scores[i]),
                                      "address": test.address.iloc[i]} for i in worst]}
    area_found = np.mean([np.any(hmm.viterbi(x, *sup)[0] == AD.S["AREA"]) for x in te_x])
    metrics["area_extraction_coverage_test"] = float(area_found)

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.7]})
    for ax, Mx, xt, yt, title in [
            (axes[0], sup[1], AD.STATES, AD.STATES, "Transitions A = p(next role | role)"),
            (axes[1], sup[2], AD.SYMBOLS, AD.STATES, "Emissions B = p(symbol | role)")]:
        ax.imshow(Mx, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(xt)), xt, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(yt)), yt, fontsize=8)
        ax.grid(False)
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Mx[i, j] > 0.55 else INK2)
        ax.set_title(title)
    fig.suptitle("U4-01 HMM fitted by supervised maximum likelihood (train split)",
                 fontweight="bold")
    save(fig, FIG, "U4_01_hmm_parameters")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_tr = sum(map(len, tr_x))
    ax.plot(np.array(bw_sup[3]) / n_tr, color=BLUE, lw=2, label="start = supervised MLE")
    for i, r in enumerate(starts):
        ax.plot(np.array(r[3]) / n_tr, color=ORANGE if r is best else LIGHT, lw=1.3,
                label="random starts (best in orange)" if i == 0 else None)
    ax.axhline(sup_tr_ll / n_tr, color=AQUA, ls="--", lw=1,
               label="supervised MLE (before EM)")
    ax.set(xlabel="Baum-Welch iteration", ylabel="train log-likelihood per part",
           title="U4-02 Baum-Welch (EM) never decreases the likelihood")
    ax.legend(fontsize=8)
    save(fig, FIG, "U4_02_baum_welch")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(1, len(long_seq) + 1)
    ax.plot(t, log10_partial, color=BLUE, lw=2, label="log10 p(x1..xt) via scaling: sum log10 c_t")
    ax.axhline(np.log10(np.finfo(float).tiny), color=ORANGE, ls="--",
               label="smallest float64 (1e-308): unscaled alpha becomes 0 below this")
    ax.axvline(underflow_at, color=MUTED, ls=":", lw=1)
    ax.set(xlabel="symbols processed (all test addresses joined)", ylabel="log10 probability",
           title="U4-03 Why scaling factors are needed")
    ax.legend(fontsize=8)
    save(fig, FIG, "U4_03_scaling")

    idx = int(np.argmax([len(x) for x in g_x[:30]]))
    x, z, adr = g_x[idx], g_z[idx], gold.address.iloc[idx]
    gam = hmm.posteriors(x, *sup)[0]
    vit = hmm.viterbi(x, *sup)[0]
    segs = AD.segments(adr)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(gam.T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.scatter(range(len(x)), vit, marker="o", s=60, facecolor="none", edgecolor=ORANGE, lw=2,
               label="Viterbi path")
    ax.scatter(range(len(x)), z, marker="x", s=40, color=INK2, label="gold label")
    ax.set_yticks(range(K), AD.STATES, fontsize=8)
    ax.set_xticks(range(len(x)), [s[:18] for s in segs], rotation=35, ha="right", fontsize=7)
    ax.grid(False)
    ax.set_title("U4-04 Posterior p(role | whole address) from forward-backward")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, FIG, "U4_04_posterior_example")

    md += ["## Part A: HMM over address parts (real data)", "",
           f"- Train: {len(tr_x)} addresses / {n_tr} parts; test: {len(te_x)} / {n_te}; gold: "
           f"{len(g_x)} test addresses / {sum(map(len, g_x))} parts (AI-judged, "
           "`data/annotation/address_gold.csv`).",
           f"- Town list learned from train ({len(towns)} names, each the part before the state "
           f"in ≥ {AD.TOWN_MIN_COUNT} train addresses): {', '.join(sorted(towns))}.", "",
           "### Markov models (held-out log-likelihood per symbol, higher is better)", "",
           md_table(pd.DataFrame({
               "model": ["order 0 (i.i.d. symbols)", "order 1 Markov chain on symbols",
                         "HMM, 5 hidden roles (supervised MLE)",
                         "HMM after Baum-Welch (from supervised)"],
               "test ll / symbol": [f"{ll0:.4f}", f"{ll1:.4f}", f"{sup_te_ll / n_te:.4f}",
                                    f"{metrics['baum_welch']['from_supervised']['heldout_ll_per_symbol']:.4f}"]})),
           "", "Stationary distribution of the role chain: " + ", ".join(
               f"{s} {p:.3f}" for s, p in zip(AD.STATES, stat)), "",
           "### Baum-Welch", "",
           f"- From the supervised start: {len(bw_sup[3])} iterations, train ll "
           f"{bw_sup[3][0]:.1f} → {bw_sup[3][-1]:.1f} ({bw_sup_time:.1f} s).",
           f"- {N_RANDOM_STARTS} random starts end at train ll "
           + ", ".join(f"{r[3][-1]:.1f}" for r in starts) + " (local optima differ).",
           "- Best random start, states mapped to roles by the Hungarian algorithm: "
           + ", ".join(f"{k}→{v}" for k, v in
                       metrics["baum_welch"]["best_random_state_to_role"].items()), "",
           "### Scaling and sum-product checks", "",
           f"- All {len(te_x)} test addresses joined = {len(long_seq)} symbols. Unscaled forward "
           f"pass returns **{unscaled}**; scaled: log10 p(X) = {log10_partial[-1]:.1f}. The "
           f"unscaled value drops below the smallest float64 after {underflow_at} symbols.",
           f"- Brute force over {K}^7 = {K ** 7} paths for a 7-part address: |Δ log-lik| = "
           f"{metrics['sum_product_check']['abs_diff_loglik']:.2e}, max |Δ marginal| = "
           f"{metrics['sum_product_check']['max_abs_diff_marginals']:.2e}, Viterbi = best path: "
           f"{metrics['sum_product_check']['viterbi_equals_brute_force']}.", "",
           "### Decoding the gold set", "",
           md_table(pd.DataFrame([{"decoder": k, "segment acc": f"{s['segment_accuracy']:.4f}",
                                   "macro F1": f"{s['macro_f1']:.4f}",
                                   "whole address exact": f"{s['address_exact_match']:.4f}",
                                   **{f"F1 {st}": f"{f:.3f}" for st, f in
                                      s["per_state_f1"].items()}}
                                  for k, s in metrics["decoding_on_gold"].items()])), "",
           f"- Rules vs supervised Viterbi disagree on "
           f"{metrics['rules_vs_viterbi_disagreements']['segments']} gold parts: Viterbi right "
           f"{metrics['rules_vs_viterbi_disagreements']['viterbi_right']}, rules right "
           f"{metrics['rules_vs_viterbi_disagreements']['rules_right']}.", "",
           md_table(pd.DataFrame(ex[:15], columns=["vendor", "part", "gold", "rules",
                                                   "Viterbi"])), "",
           "### Least likely test addresses (per-part log-likelihood under the HMM)", "",
           md_table(pd.DataFrame([{"vendor": d["vendor_id"],
                                   "ll/part": f"{d['ll_per_part']:.3f}",
                                   "address": d["address"].replace("|", "/")}
                                  for d in metrics["anomaly"]["lowest"]])),
           f"\nMedian test ll/part: {metrics['anomaly']['heldout_ll_per_part_median']:.3f}. "
           f"Viterbi finds at least one AREA in {area_found:.4f} of test addresses.", ""]
    return sym_trans


def simulate_route(v):
    """Pick 8 real pharmacies in a nearest-neighbour chain and drive between them at constant
    speed, with a 20 s stop at each. Returns true positions (km) sampled every DT_S seconds."""
    lat0, lon0 = v.lat.mean(), v.lon.mean()
    ph = v[v.y == 1]
    P = to_km(ph.lat, ph.lon, lat0, lon0)
    rng = np.random.default_rng(SEED)
    cur = int(rng.integers(len(P)))
    order, left = [cur], set(range(len(P))) - {cur}
    while len(order) < 8:
        nxt = min(left, key=lambda j: np.sum((P[j] - P[cur]) ** 2) if
                  np.sum((P[j] - P[cur]) ** 2) > 0.25 else np.inf)
        order.append(nxt)
        left.discard(nxt)
        cur = nxt
    stops = P[order]
    step = SPEED_KMH / 3600 * DT_S
    pts = []
    for a, b in zip(stops[:-1], stops[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        pts += [a + (b - a) * i / n for i in range(n)]
        pts += [b] * int(20 / DT_S)
    return np.array(pts), stops, ph.title.iloc[order].tolist()


def part_b(md, metrics):
    v = L.load_vendors()
    truth, stops, names = simulate_route(v)
    rng = np.random.default_rng(SEED)
    obs = truth + rng.normal(0, GPS_SIGMA_KM, truth.shape)
    m0 = np.r_[obs[0], 0, 0]
    P0 = np.diag([GPS_SIGMA_KM ** 2] * 2 + [0.01] * 2)
    q_grid = np.logspace(-8, -4, 17)
    r_grid = np.logspace(-4.5, -2.5, 17)
    LL = np.empty((len(q_grid), len(r_grid)))
    for i, q in enumerate(q_grid):
        F, Q, H = lds.constant_velocity(DT_S, q)
        for j, r in enumerate(r_grid):
            LL[i, j] = lds.kalman_filter(obs, F, H, Q, r * np.eye(2), m0, P0)["loglik"]
    i, j = np.unravel_index(LL.argmax(), LL.shape)
    q_hat, r_hat = q_grid[i], r_grid[j]
    F, Q, H = lds.constant_velocity(DT_S, q_hat)
    kf = lds.kalman_filter(obs, F, H, Q, r_hat * np.eye(2), m0, P0)
    ms, _ = lds.rts_smoother(F, kf)
    k = 5
    ma = np.column_stack([np.convolve(obs[:, d], np.ones(k) / k, mode="same") for d in (0, 1)])
    ma[:k // 2], ma[-(k // 2):] = obs[:k // 2], obs[-(k // 2):]

    def rmse_m(est):
        return float(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))) * 1000)

    res = {"raw_gps": rmse_m(obs), "moving_average_5": rmse_m(ma),
           "kalman_filter": rmse_m(kf["m"][:, :2]), "rts_smoother": rmse_m(ms[:, :2])}
    metrics["lds"] = {"SIMULATED": True, "n_steps": int(len(truth)), "dt_s": DT_S,
                      "true_gps_sigma_m": GPS_SIGMA_KM * 1000,
                      "ml_gps_sigma_m": float(np.sqrt(r_hat) * 1000), "ml_q": float(q_hat),
                      "route_km": float(np.sum(np.linalg.norm(np.diff(truth, axis=0), axis=1))),
                      "rmse_m": res, "stops": names}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(obs[:, 0], obs[:, 1], s=3, color=LIGHT, label="noisy GPS (simulated)")
    ax.plot(truth[:, 0], truth[:, 1], color=INK2, lw=1, label="true path")
    ax.plot(kf["m"][:, 0], kf["m"][:, 1], color=ORANGE, lw=1.2, label="Kalman filter")
    ax.plot(ms[:, 0], ms[:, 1], color=BLUE, lw=1.4, label="RTS smoother")
    ax.scatter(stops[:, 0], stops[:, 1], marker="*", s=120, color=AQUA, zorder=5,
               label="pharmacies visited")
    ax.set(xlabel="east (km)", ylabel="north (km)", aspect="equal",
           title="U4-05 Tracking a simulated agent between real vendors")
    ax.legend(fontsize=7)
    ax = axes[1]
    cs = ax.contourf(np.log10(r_grid), np.log10(q_grid), LL - LL.max(),
                     levels=np.linspace(-60, 0, 13), cmap="Blues", extend="min")
    ax.plot(np.log10(r_hat), np.log10(q_hat), "o", color=ORANGE, ms=8, label="ML estimate")
    ax.axvline(np.log10(GPS_SIGMA_KM ** 2), color=PURPLE, ls="--", lw=1,
               label="true GPS variance")
    fig.colorbar(cs, ax=ax, label="log-likelihood − max")
    ax.set(xlabel="log10 R (GPS variance, km²)", ylabel="log10 q (process noise)",
           title="Maximum-likelihood noise levels (innovations)")
    ax.legend(fontsize=8)
    ax.grid(False)
    save(fig, FIG, "U4_05_kalman")

    md += ["## Part B: linear dynamical system (SIMULATED trajectory)", "",
           "**The trajectory is simulated**: the dataset has no movement data. Stops are real "
           f"pharmacy locations; the agent drives at {SPEED_KMH:.0f} km/h, sampled every "
           f"{DT_S:.0f} s, with {GPS_SIGMA_KM * 1000:.0f} m GPS noise.", "",
           f"- {len(truth)} time steps, route length "
           f"{metrics['lds']['route_km']:.2f} km.",
           f"- ML (grid over the innovations log-likelihood): GPS σ = "
           f"{metrics['lds']['ml_gps_sigma_m']:.1f} m (true 20 m), q = {q_hat:.2e}.", "",
           md_table(pd.DataFrame({"estimate": list(res), "position RMSE (m)":
                                  [f"{x:.2f}" for x in res.values()]})), ""]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    md = ["# Unit IV numbers (generated by `python src/pbl2_unit4.py`; do not edit by hand)", ""]
    metrics = {}
    part_a(md, metrics)
    part_b(md, metrics)
    write_json(METRICS, metrics)
    (OUT / "unit4_numbers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    d = metrics["decoding_on_gold"]
    print("gold segment accuracy:", {k: round(s["segment_accuracy"], 4) for k, s in d.items()})
    print("lds rmse (m):", metrics["lds"]["rmse_m"])
    print(f"wrote {METRICS}, {OUT / 'unit4_numbers.md'}, figures U4_*")


if __name__ == "__main__":
    main()
