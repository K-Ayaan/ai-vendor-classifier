# 6. Practice questions (short answers)

**Q1. Why is minimising RSS the same as maximising the likelihood?**
In ℓ = −(n/2)ln(2πσ²) − RSS/(2σ²), β appears only in −RSS/(2σ²). Since 1/(2σ²) > 0,
maximising ℓ over β means minimising RSS. This holds only for Gaussian errors.

**Q2. Why divide by n for σ̂²_MLE but n − p for s²?**
The MLE maximises the likelihood and gives RSS/n = 224.2774, but it is biased low by (n−p)/n
because 7 parameters were fitted. s² = RSS/(n−p) = 228.7249 is unbiased. The ratio is
353/360 = 0.980556.

**Q3. MSE equals σ̂²_MLE on train, so why give them different names?**
Same number, different meaning. MSE is a loss on any dataset (test MSE = 304.7785).
σ̂²_MLE estimates the noise-variance parameter of the Gaussian model.

**Q4. What is the maximised log-likelihood, and why is the second term −180?**
ℓ = −1305.1370 − 180 = −1485.1370. At σ̂² = RSS/n, the term RSS/(2σ̂²) = n/2 = 180.

**Q5. Interpret β₀ = 0.2962.**
The prediction when all features are 0. No vendor has 0 categories, so it is an anchor, not
a real vendor. A right-category vendor 1 km away inside the radius with no keyword is
predicted 74.48, close to the rubric's 75.

**Q6. Is R² = 0.80 good?**
It means 80% of the variation in the AI judge's scores is explained (75% on unseen
requirements). But the features are the judge's own criteria, so a high R² is expected and
does not prove the matcher is good in the real world.

**Q7. Why not use linear regression for is_pharmacy?**
It produced 326 negative and 18 above-1 "probabilities" on the test set, assumes Gaussian
constant-variance noise for a 0/1 variable, and its straight line cannot flatten near 0 and 1.

**Q8. Why does logistic regression need IRLS?**
The score equations Xᵀ(y − σ(Xβ)) = 0 are non-linear in β, so there is no algebraic
solution. Newton's method on the concave log-likelihood, each step a weighted least-squares
fit, solves it.

**Q9. What does "the MLE does not exist" mean here?**
The training data are separable (e.g. *medicals* appears only in pharmacies), so ℓ keeps
increasing as coefficients go to ±∞ and never reaches a maximum. Evidence: 276 one-class
tokens, an LP separating direction, ‖β‖ running away, and two solvers giving the same ℓ with
coefficients 2705 apart.

**Q10. How did you fix it, and is the L2 model an MLE?**
(1) A small model that avoids separating features, which has a true MLE (IRLS = sklearn to
1.78e-15). (2) L2 penalty: a unique solution always exists, but it is a penalised likelihood
or MAP estimate, **not** an MLE.

**Q11. Accuracy is 0.987. Why also report F1 and PR-AUC?**
Only 10.5% are pharmacies; always predicting "no" already gets 0.8949 accuracy with recall 0.
Precision, recall, F1 and PR-AUC show performance on the rare class.

**Q12. Read the L2 confusion matrix.**
TN 825, FP 1, FN 11, TP 86. One false alarm; 11 missed "pharmacies", several of which are
actually mislabelled (a chemistry academy, supermarkets).

**Q13. Why split regression by requirement, not randomly?**
The 15 pairs of a requirement share a buyer. A random split would put the same requirement in
train and test and inflate scores. At deployment the requirements are new.

**Q14. Where did the labels come from? Are they ground truth?**
Match scores and the 200 audit labels are AI-judged (Claude Code, against a written rubric).
They are not ground truth, and human validation was not done. AI consistency was checked
blind on 45 pairs: r = 0.9762, mean absolute difference 4.133.

**Q15. What does the Q-Q plot tell you?**
The tails bend away from the line, so the residuals are heavier-tailed than Gaussian
(Shapiro-Wilk p = 3.59e-08). The Gaussian-MLE interpretation and the SEs are approximate.

**Q16. How is ridge related to MLE and to Bayesian regression?**
Ridge minimises RSS + λ‖β‖², which is the MAP estimate under a Gaussian prior on β. With
λ = 0 it is the MLE. Here CV chose λ = 0.001, so ridge = OLS: 6 well-conditioned features
have nothing to shrink.

**Q17. What does Bayesian linear regression give that OLS does not?**
A predictive distribution: each prediction has a mean and a variance (1/α + xᵀS_N x). On test,
the 95% intervals covered 93.33% of the scores; the noise variance 1/α = 228.06 ≈ s² = 228.72.

**Q18. Why use a robust (Huber) loss?**
Squared loss lets large errors dominate. Huber is linear beyond ε·scale, so mislabelled pairs
pull less. 121 of 360 pairs were in the linear zone; test R² rose from 0.7510 to 0.7630.

**Q19. What is the kernel trick, and why did kernel ridge help?**
A kernel computes inner products in a higher-dimensional feature space without building it.
A degree-2 polynomial kernel adds every pairwise interaction. The rubric is multiplicative
(category AND radius AND keyword), so test R² rose from 0.751 to 0.808 (poly) and 0.813 (RBF).

**Q20. Generative vs discriminative: which won, and why?**
Logistic regression (discriminative, models p(C|x)) scored F1 0.9348. Naive Bayes (generative,
models p(x|C) with an independence assumption) scored 0.9101, and LDA (Gaussian classes) 0.5185.
With 3,689 training vendors the discriminative model wins; LDA's Gaussian assumption suits 0/1
words poorly.

**Q21. What is the Laplace approximation?**
A Gaussian approximation to a posterior with no closed form: centre it at the MAP, with
covariance equal to the inverse Hessian of the negative log-posterior, (XᵀWX + λP)⁻¹. It
reuses the IRLS Hessian.

**Q22. Did Bayesian logistic regression help?**
It gives the same decisions as L2 (F1 0.9348) and always has a solution, which fixes separation
with a prior. But averaging over a wide posterior pulled probabilities towards 0.5, so log loss
worsened from 0.0583 to 0.1065. Bayesian is not automatically better; the prior matters.

**Q23. Why can an SVM work when the logistic MLE does not exist?**
On separable data the logistic likelihood has no maximum, but the maximum-margin problem has a
unique solution. The linear SVM matched L2 logistic (F1 0.9348) using 342 support vectors.

**Q24. Why notebooks if you have scripts?**
Scripts are the reproducible source of truth. Notebooks import the same functions and put the
derivation, code, number and plot together for explanation, re-executed from scratch.
