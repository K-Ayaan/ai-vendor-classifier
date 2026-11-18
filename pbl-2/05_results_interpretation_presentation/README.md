# 5. Result Interpretation & Presentation (4 marks)

> What the examiner checks: can you say what the numbers **mean** in plain words, explain
> **where and why** the methods fail, state limitations honestly, and present it all in a clear
> order?

---

## 5.1 Interpreting the market zones (Unit III)

**K = 5 zones** (figure U3_03; the place names are the most common `locality` values inside each
k-means zone):

| zone | vendors | main localities | what it means for the agent |
|---|---|---|---|
| Chennai city (north-east) | 1,249 | Chennai 1,197 | dense, many options per category |
| south-east suburbs | 763 | Chennai 441, Medavakkam 90, Perumbakkam 46, Thoraipakkam 27 | spread along several roads: "nearest" depends on the road |
| Tambaram | 458 | Tambaram 270, Pallavaram 16 | the smallest zone, a compact town |
| GST Road corridor | 919 | Maraimalai Nagar 177, Nandivaram-Guduvancheri 169, Kattankulathur 156 | linear: a radius search wastes area off the road |
| Chengalpattu (south-west) | 1,223 | Chengalpattu 620, Gokulapuram 174, Ammanambakkam 79 | compact town centre plus villages around it |

**Plain-words reading of the numbers:**
- **Hopkins 0.925:** vendors are far from random. Clustering finds real market areas.
- **K-means vs locality ARI 0.390:** the zones agree with Google's locality names well above
  chance (0.000) but far from perfectly, because a "locality" (e.g. *Chennai*, 1,776 vendors) is
  an administrative label, not a market area.
- **1,138 vendors with max responsibility < 0.9:** about a quarter of vendors sit between zones.
  For them a zone label is a guess, and the GMM says so.
- **BIC keeps falling to K = 25:** there is no single "true" number of zones; structure exists at
  every scale. K = 5 is a *choice* of scale, made by the BIC knee, not a discovery.

**For the matching agent:** a zone feature does **not** improve the score model significantly
(30-fold CV R² 0.7784 → 0.7829, Wilcoxon p = 0.33), because the existing distance and radius
features already carry that information. Zones are better used for **explanation and
coverage** ("3 pharmacies in your zone, 7 in the neighbouring one") than as a scoring feature.

## 5.2 Interpreting the retail-mix factors

- **PC1** (12.4% of variance): daily-food high streets (grocery, bakery, food stalls,
  vegetables) vs durable-goods strips (home appliances, electronics, pet supplies).
- **PC2** (9.6%): basic-needs residential streets (meat/fish, ration shops, fruits) vs
  consumer retail (mobile accessories, homeware, beauty).
- **But held-out likelihood keeps only q = 1.** PC2 and beyond do not generalise to new cells:
  they describe *this* set of 78 cells, not a stable pattern. **Report PC1 as a finding; treat
  PC2 and later as descriptive.**
- The category dendrogram (U3_05) tells the same story at the category level: food/daily items
  group together, consumer retail groups together, and ration_pds + meat_fish form their own
  pair.

## 5.3 Interpreting the HMM (Unit IV)

**Transition matrix in plain words** (U4_01):
- After a STREET comes an AREA 67% of the time; after an AREA comes a CITY 52% of the time;
  after a CITY comes the STATE 86% of the time. Addresses run from small to large.
- 87.8% of addresses start with a PREMISE part (door number, shop or building).

**Accuracy in plain words:** on 60 unseen addresses the HMM gives **94 of every 100 parts** the
right role, and gets **42 of 60 whole addresses** completely right (70%). The rules it was
trained on manage 91% and 38 of 60.

**Why it beats its teacher:** the rules look at one part at a time; the HMM weighs both
neighbours. "Porur" in "Devi Nagar, **Porur**, Chennai, Tamil Nadu" has no road before it, so
the rule calls it a PREMISE (shop name). The HMM knows a WORD between a NAGAR part and the city
is almost always an AREA.

## 5.4 Error analysis

**HMM (Viterbi) errors on the gold set:** 21 parts wrong out of 371, grouped by cause:

| part (vendor) | gold | Viterbi | real reason |
|---|---|---|---|
| numbered streets: `2nd St`, `4th Street`, `100 Feet Rd`, `6th cross st`, `27 Mettu Street` (11 of the 21 errors) | STREET | PREMISE | the symbol rules check NUMBER before ROAD, so a street name starting with a digit becomes NUMBER and the HMM never sees the road word. **The emission alphabet loses information; the HMM cannot recover what the symbols threw away** |
| `Express Avenue Mall SHOP 272…`, `near Express Avenue Mall` (V2109, V1086) | PREMISE | AREA | "avenue" is on the NAGAR word list, so a mall name looks like a neighbourhood |
| `New` (V2887) | STREET | **STATE** | "New, Trunk Rd" is one street split by a comma. The STATE row of A was learned from almost no transitions out of STATE, so add-one smoothing made STATE → STREET look plausible (0.231). **Smoothing a near-empty row can create impossible paths** |
| `Chengalpattu` as the first part (V3995) | CITY | PREMISE | a town name *first* in the address; the HMM has learned that addresses start with PREMISE (π = 0.878) |
| `SH 49`, `SH58` (state highway codes) | STREET | PREMISE | not on the ROAD word list |

**Least likely test addresses** (lowest per-part log-likelihood under the HMM):

| address | why it looks strange to the model |
|---|---|
| `Pareri, Tamil Nadu 603204` | only 2 parts; a single WORD straight before the state |
| `GST Rd, Singaperumal Koil, Tamil Nadu 603204` | **"Koil"** (temple) is on the LANDMARK word list, so the *town* Singaperumal Koil becomes a LANDMARK right before the state, which almost never happens in training |
| `CAR STREET, Theradi St, Singaperumal Koil, …` | same town, same cause |

All five lowest addresses come from **one pincode (603204)**, and four of them contain Singaperumal
Koil. The HMM's likelihood has found **a bug in my own symbol rules**: an anomaly score is
also a debugging tool.

**Clustering failure modes:**
- Single linkage chains the whole corridor into one cluster (99.6%).
- Average linkage scores best but leaves clusters of 3 vendors.
- EM from random starts lands in different optima (pairwise ARI as low as 0.55), and one HMM
  random start was stuck 1,450 nats below the others.

## 5.5 Key takeaways (the five sentences to remember)

1. **Clustering is justified and multi-scale.** Hopkins 0.925 vs 0.500 random; BIC never
   settles, so K = 5 is a choice of scale (BIC knee), and the GMM shows 1,138 uncertain border
   vendors.
2. **EM provably climbs.** Measured: ln p rises every iteration (−29,459 → −27,215), and the KL
   gap after each M-step is always ≥ 0. K-means is the ε → 0 limit (ARI 0.9994).
3. **Variance is not structure.** 90% variance needs 17 components; held-out likelihood (PPCA,
   FA, Minka) says 1.
4. **Context wins.** An HMM fitted by maximum likelihood labels 94.3% of address parts correctly
   and beats the rules it learned from (91.4%), because transitions use neighbours.
5. **Likelihood ≠ usefulness.** Baum-Welch improves held-out likelihood but drops accuracy to
   88.7%; a zone feature looked useful on 6 test buyers but is not significant over 30.

## 5.6 Limitations (say them before you're asked)

| Limitation | Effect | Next step |
|---|---|---|
| Gold labels are AI-judged (60 addresses) | accuracy is agreement with a written guide, not human truth | a human-labelled set; inter-annotator agreement |
| Single snapshot, no real sequences in time | Unit IV on time series could not be done on real data | repeated scrapes; real agent GPS logs |
| Kalman trajectory is **simulated** | proves the algorithm, not a field result | real delivery traces |
| Symbol alphabet is hand-written | loses words: numbered streets cause 11 of 21 HMM errors; "Koil" makes a town look like a landmark | emissions over words or characters; a learned symbol set |
| Add-one smoothing on a near-empty STATE row | allows impossible STATE → STREET paths | an explicit END state; smaller pseudo-count for rare rows |
| Gaussian zones vs road-shaped markets | needs a wide background component | mixtures along road distance, or density-based clustering |
| 78 cells for PCA/FA | little power to find more than one factor | smaller cells over a larger area; more snapshots |
| K chosen by a heuristic (BIC knee) | another rule gives another K | report sensitivity; let the application fix the scale |

## 5.7 How to present (about 10 minutes, in this order)

| Min | Slide | Show | Say |
|---|---|---|---|
| 0–1 | Problem | §1.1, §1.2 table | "Same vendors as PBL-1; now find hidden structure: zones, retail factors, address roles, agent position." |
| 1–2 | EDA | E02, E03, E07 | "Hopkins 0.925 → cluster; clustered at every scale → BIC won't settle; previous part cuts entropy 35% → Markov." |
| 2–4 | K-means & GMM by EM | U3_01, U3_03 | derive E and M steps; K by BIC knee = 5; own = scikit-learn (−27,215.33); hard vs soft, 1,138 border vendors |
| 4–5 | Alternative view of EM | U3_02 | ln p = L + KL; KL after M-step ≥ 0; k-means as ε → 0 |
| 5–6 | PCA / FA / choosing q | U3_04 | "Variance rules say 17, likelihood says 1." PPCA vs FA noise models |
| 6–6.5 | Hierarchical | U3_05, C01 | 4 linkages; cophenetic; single linkage chaining |
| 6.5–8.5 | HMM | U4_01, U4_03, U4_04, C02 | counts = MLE; forward-backward = sum-product (checked on 78,125 paths); scaling (underflow at 424); Viterbi 94.3% > rules 91.4%; Baum-Welch paradox |
| 8.5–9 | LDS | U4_05 | "Simulated: Kalman = continuous HMM; ML σ 20.5 m vs 20; smoother 27.8 → 8.4 m." |
| 9–10 | Interpretation & limits | §5.4, §5.6 | Koil bug found by likelihood; zone feature p = 0.33; honest limitations |

**Presentation tips**
- For every algorithm, show **one check against a reference** (scikit-learn, brute force, true
  noise level). It proves the code is right without reading it line by line.
- Say "**simulated**" every time the Kalman figure is on screen.
- Lead with the surprising results (HMM beats its teacher; higher likelihood, lower accuracy;
  17 vs 1 dimensions). They show you understood, not just ran, the methods.
- Keep `src/latent/hmm.py` open: `forward` is a dozen lines and maps line-by-line to the recursion.

---

## Viva questions on this section

**Q1. What did clustering add to the vendor-matching agent?**
Zones for explanation and coverage, plus an "uncertain zone" flag for 1,138 border vendors. As a
scoring feature it adds nothing significant (p = 0.33), because distance and radius already say
it.

**Q2. Your HMM is trained on rule labels. How can it be better than the rules?**
The rules make *systematic* errors (any WORD before the first road becomes PREMISE), but most of
their labels are right. The HMM fits the *pattern* across 3,688 addresses: AREA is usually
followed by CITY, PREMISE rarely is. On ambiguous parts the context outweighs the occasional
wrong training label. On the 14 parts where they disagree, the HMM is right 12 times.

**Q3. Why report a result where Baum-Welch made things worse?**
Because it is the most important lesson about latent-variable models: maximum likelihood finds
states that explain the data, not states that match our labels. Hiding it would overstate what
EM does.

**Q4. What would you do with one more week?**
Fix the symbol rules (numbered streets = 11 of the 21 HMM errors; Koil) or replace them with word-level emissions; add an
END state instead of smoothing the STATE row; get 200 human-labelled addresses; try road-network
distance for the zones.

**Q5. What is the single biggest weakness?**
No real sequential *time* data, so the LDS part is a simulation. The address HMM is real, and
it is where the Unit IV results come from.

**Q6. Is a Hopkins statistic of 0.925 enough to say there are 5 zones?**
No. Hopkins says the data are clustered, not how many clusters there are. The number came from
BIC's knee, and the BIC curve itself says structure continues at finer scales.
