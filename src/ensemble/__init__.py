"""Own implementations for PBL-3 (Unit V, combining models), in numpy/scipy only.

tree      CART for regression (squared error) and classification (Gini), sample weights,
          random feature subsets, cost-complexity (weakest-link) pruning, impurity importance
boosting  AdaBoost with decision stumps (Bishop 14.3), gradient boosting for squared error
          (L2-boosting) and logistic loss (Newton leaf values)
committee bagging and random forests with out-of-bag predictions
gam       generalised additive model: P-spline smoothers, backfitting, direct solve, GCV
bma       Bayesian model averaging over linear models (Zellner g-prior and BIC) and over
          logistic models (BIC)
mixture   conditional mixture models: mixture of linear regressions (EM) and a two-expert
          mixture of experts with a logistic gate (EM + IRLS)
"""
