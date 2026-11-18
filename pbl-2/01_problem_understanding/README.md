# 1. Problem Understanding & Dataset Familiarization (4 marks)

> What the examiner checks: can you state what Units III and IV are used **for** in this
> project, which parts of the data each method needs, and what the data cannot support?

---

## 1.1 The problem in one paragraph

PBL-1 built a vendor-matching agent: given a buyer's need, location and distance limit, score
and rank local vendors. That used **supervised** learning: every target (match score,
is-pharmacy) was given. PBL-2 asks what **hidden (latent) structure** the same data contains
when nobody gives us labels:

1. **Where are the market zones?** Vendors are not spread evenly. Group them into zones
   (Unit III: k-means, Gaussian mixtures by EM, hierarchical clustering).
2. **What kinds of retail area exist?** Each area has a mix of shop types. Find the few hidden
   factors behind that mix (Unit III: PCA, factor analysis, choosing the number of dimensions).
3. **What role does each part of an address play?** An address is a *sequence* of parts
   whose hidden role (door number, street, area, city, state) is never written down
   (Unit IV: Markov models, HMMs, forward-backward, Viterbi).
4. **Where is a moving agent right now?** A delivery agent moving between vendors reports noisy
   GPS. Recover the true path (Unit IV: linear dynamical systems, Kalman filter).

## 1.2 Why these are latent-variable problems

| Question | Observed | Hidden (latent) | Model |
|---|---|---|---|
| market zones | vendor lat/lon | which zone each vendor belongs to (z_n) | mixture of Gaussians, k-means |
| retail mix | share of each category per 2 km cell (23 numbers) | a few continuous factors | PCA / PPCA / factor analysis |
| category groups | which categories occur where | a tree of groups | hierarchical clustering |
| address roles | sequence of comma-separated parts | role of each part, z_1 … z_T | hidden Markov model |
| agent position | noisy GPS fixes x_t | true position and velocity z_t | linear dynamical system |

In every case **maximum likelihood** with hidden variables leads to the same idea: if we knew
z, estimation would be easy (counts, means, covariances), so we alternate between inferring z
(E-step / forward-backward / Kalman smoother) and re-estimating the parameters (M-step). This
links PBL-2 to the MLE theme of PBL-1.

## 1.3 The dataset again, through the lens of Units III and IV

**Source:** the same 23 JSON files (Google Maps snapshot, 2026-09-27, South Chennai to
Chengalpattu), merged into **4,612 unique vendors** in `data/processed/vendors.csv`. See
[PBL-1 §1.4](../../pbl-1/01_problem_understanding/README.md) for the full field audit.

| Field | Used in PBL-2 for | Notes |
|---|---|---|
| `lat`, `lon` | clustering (k-means, GMM, hierarchical), grid cells, the simulated route | projected to km: x = Δlon·111.32·cos φ₀, y = Δlat·111.32. Max error vs haversine 0.19% (§3.4) |
| `categories` | retail mix per cell (PCA/FA), category dendrograms | multi-label set, 23 categories |
| `address` | **the sequential data** for the HMM: split on commas | 1 empty address; mean 6.12 parts, max 17 |
| `locality`, `pincode` | **external** labels to evaluate clusters (ARI, NMI) | never used to fit a clustering |
| `title` | names of pharmacies on the simulated route | – |
| `expected_score` (pairs) | testing whether a zone feature helps the PBL-1 regression | same split by requirement as PBL-1 |

## 1.4 What the data does NOT contain (and what we did about it)

| Missing | Consequence | Our answer |
|---|---|---|
| Any time series (one snapshot) | no natural sequential data for Unit IV | the **address** is the sequence: parts in reading order. This is a real, classic HMM use (address parsing) |
| Labels for address roles | can't train or score an HMM directly | (a) a **rule-based labeller** (`src/pbl2_address.py`) gives "silver" labels on the **train** split for supervised MLE; (b) a **gold set**: 60 addresses (371 parts) from the **test** split, labelled by the AI judge following a written guide (`data/annotation/ADDRESS_GUIDE.md`) |
| Movement / GPS traces | an LDS needs a moving object | **simulated** agent: drives between 8 real pharmacy locations at 25 km/h, GPS every 2 s with 20 m noise. Clearly marked SIMULATED everywhere |
| True zones | can't measure clustering "accuracy" | internal measures (silhouette, Davies-Bouldin, Calinski-Harabasz) plus agreement with locality and pincode (ARI, NMI) as external proxies |

> **Say this confidently in the viva:** "The data is one snapshot, so it has no time axis. But
> every address *is* a sequence, and its parts follow a strong order (door, street, area, city,
> state). That is exactly what an HMM models. The only simulated part is the Kalman tracking
> demo, and I say so on every figure."

## 1.5 How the address becomes HMM data

```
"No.7, GST Rd, Chengalpattu, Thirutheri R.F., Tamil Nadu 603001"
   │      │          │              │                  │
 NUMBER  ROAD       TOWN           WORD             STATEPIN      ← observed symbol x_t (10 kinds)
PREMISE STREET      CITY           CITY              STATE        ← hidden role z_t (5 kinds)
```

**Observations (M = 10 symbols):** each part is reduced to its surface form by fixed rules,
checked in order: STATEPIN (`tamil nadu`), PLUSCODE (e.g. `X57X+2C4`), NUMBER (starts with a
door number), TOWN (in a town list learned from the **train** split: names that come right
before the state in ≥ 30 train addresses, 13 names), ROAD (st, rd, salai, …), NAGAR (nagar,
colony, …), LANDMARK (near, opp, temple, …), BUILDING (floor, complex, shop, …), SHORT, WORD.

**Hidden states (K = 5 roles):** PREMISE, STREET, AREA, CITY, STATE.

**Why it needs a sequence model:** WORD is the most common symbol (19.8%) and can be a shop
name (PREMISE), a neighbourhood (AREA) or a town (CITY). Only its **position and neighbours**
tell which.

## 1.6 Splits and leakage control

| Item | Split | Why |
|---|---|---|
| Vendors for the HMM | the PBL-1 stratified 80/20 split: 3,688 train / 923 test addresses (1 empty dropped) | same split as the classifier, so nothing new is peeked at |
| Town list, rule labels, HMM parameters | **train only** | a gazetteer is a fitted vocabulary |
| Gold set | 60 random **test** addresses (seed 42) | asserted in code: gold ⊂ test |
| GMM held-out likelihood | random 80/20 of vendors | to choose K without rewarding overfitting |
| PCA/FA held-out likelihood | 5-fold CV over the 78 cells | to choose q |
| Zone feature regression | PBL-1 split by requirement (24/6), then leave-one-requirement-out (30 folds) | a test buyer is never seen in training |

## 1.7 Assumptions (know these)

- **Gaussian zones:** a GMM assumes each zone is an ellipse. Real zones follow roads (long thin
  shapes), so it needs more components or one wide "background" component (§5).
- **First-order Markov roles:** the next part's role depends only on the current role. Long
  addresses with repeated sections ("…, Chennai, New, Trunk Rd, …") break this.
- **One symbol per part:** the HMM sees the symbol, not the words. Two different neighbourhoods
  both look like WORD.
- **Linear-Gaussian motion:** the Kalman model assumes nearly constant velocity. The simulated
  agent stops at each pharmacy, which violates it (that is why the smoother matters, §5).
- **Gold labels are AI-judged**, like the PBL-1 scores: consistent with the written guide, but
  not human ground truth.

## 1.8 Glossary

| Term | One-line meaning |
|---|---|
| Latent variable | a quantity in the model that is never observed (zone, role, true position) |
| Responsibility r_nk | posterior probability that vendor n belongs to component k |
| ELBO / lower bound L(q, θ) | Σ q ln p(x, z) − Σ q ln q; ln p(X) = L + KL(q ‖ posterior) |
| BIC | −2 ln L + p ln n: likelihood penalised by parameter count |
| PPCA | PCA as a Gaussian latent model x = Wz + μ + ε, ε ~ N(0, σ²I) |
| Factor analysis | like PPCA but ε ~ N(0, Ψ) with a separate noise variance per feature |
| Silhouette | (b − a)/max(a, b): own-cluster vs nearest-other-cluster distance, −1 to 1 |
| ARI / NMI | agreement between two partitions, corrected for chance / information-based |
| Cophenetic correlation | correlation between original distances and the tree's merge heights |
| Markov chain | p(x_t \| x_1..x_{t−1}) = p(x_t \| x_{t−1}) |
| HMM | Markov chain on hidden states; each state emits an observation |
| Forward α, backward β | α_t(k) = p(x_1..x_t, z_t = k); β_t(k) = p(x_{t+1}..x_T \| z_t = k) |
| Scaling factor c_t | p(x_t \| x_1..x_{t−1}); keeps α in range; ln p(X) = Σ ln c_t |
| Viterbi | most probable whole path (max-product) |
| Posterior decoding | most probable role for each part separately (sum-product) |
| Baum-Welch | EM for HMMs |
| Kalman filter / RTS smoother | forward / forward-backward inference for a linear-Gaussian state-space model |
| Gold set | held-out labelled examples used only for scoring |

---

## Viva questions on this section

**Q1. Why is an address "sequential data"? There is no time.**
Sequential data only needs an order in which neighbours depend on each other. Address parts
follow a near-fixed order (door → street → area → city → state). E07 shows that knowing the
previous part cuts the uncertainty of the next part's symbol from 2.943 to 1.902 bits (35%).

**Q2. Where do the HMM's training labels come from?**
From a rule-based labeller applied to the 3,688 **train** addresses. The HMM is fitted by
maximum likelihood (normalised counts with add-one smoothing) on those labels, then scored on 60
**test** addresses labelled separately by the AI judge. Baum-Welch needs no labels at all.

**Q3. Isn't the Kalman part fake?**
The trajectory is simulated, and I say so. The stops are real pharmacy coordinates and the noise
level is realistic (20 m). It demonstrates the LDS algorithm and maximum-likelihood noise
estimation. ML recovers σ = 20.5 m against a true 20 m. It is not a result about the dataset.

**Q4. Why project lat/lon to km before clustering?**
K-means and GMMs use Euclidean distance. At 12.9° N a degree of longitude is 108.5 km but a
degree of latitude is 111.3 km, so raw degrees distort east-west distance by up to 2.8%. The km
projection is within 0.19% of the true great-circle distance over the whole area.

**Q5. How do you evaluate clusters when there is no ground truth?**
Internally (silhouette, Davies-Bouldin, Calinski-Harabasz: compactness vs separation) and
externally against labels the clustering never saw (locality, pincode) with ARI and NMI. I also
check stability across random starts.
