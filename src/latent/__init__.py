"""PBL-2 latent-variable models, written from the equations (numpy/scipy only).

clustering.py  k-means (Lloyd + k-means++), Gaussian mixture fitted by EM (with the
               ELBO / KL decomposition of the "alternative view of EM")
pca.py         PCA by eigendecomposition, probabilistic-PCA log-likelihood
hmm.py         discrete HMM: scaled forward-backward, Viterbi, Baum-Welch, supervised MLE,
               brute-force enumeration (for checking)
lds.py         linear dynamical system: Kalman filter, RTS smoother, innovations log-likelihood
"""
