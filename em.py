"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    # unpack tuplu
    mu, var, pi = mixture
    K = mu.shape[0]

    # create a delta matrix to indicate where X is non-zero
    delta = X.astype(bool).astype(int)
    # Exponent term
    # norm matrix/(2*variance)
    f = (np.sum(X ** 2, axis=1)[:, None] + (delta @ mu.T ** 2) - 2 * (X @ mu.T)) / (2 * var)
    # pre-exponent term : a matrix of shape (n, K)
    pre_exp = (-np.sum(delta, axis=1).reshape(-1, 1) / 2.0) @ (np.log((2 * np.pi * var)).reshape(-1, 1)).T
    # together
    f = pre_exp - f

    f = f + np.log(pi + 1e-16)  # f(i, j) matrix
    # log of normalizing term in [(j|u)
    logsum = logsumexp(f, axis=1).reshape(-1, 1)  # store this to calculate log_lh
    log_posts = f - logsum  # this is the log of posteriod prob. matrix log(p(j|u)
    log_lh = np.sum(logsum, axis=0).item()  # this is the log liklihood

    return np.exp(log_posts), log_lh


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset
    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian
    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu_rev, _, _ = mixture
    K = mu_rev.shape[0]

    # calculate revised pi(j): same expression as in the naive case
    pi_rev = np.sum(post, axis=0) / n

    # create delta matrix indic where X is non-zero
    delta = X.astype(bool).astype(int)

    # update means only when sum_u(p(j\u) * delta (l, Cu)) >= 1
    denom = post.T @ delta  # denomonator (K, d). only include dims that have information
    numer = post.T @ X  # numerator (K, d)
    update_indices = np.where(denom >= 1)  # indices for update

    mu_rev[update_indices] = numer[update_indices] / denom[update_indices]  # only update where necessary (denom >=1)

    # update variances
    denom_var = np.sum(post * np.sum(delta, axis=1).reshape(-1, 1), axis=0)

    norms = np.sum(((X[:, None, :] - mu_rev) * delta[:, None, :]) ** 2, axis=2)
    # norms = np.sum(X**2, axis = 1)[:,None] + (delta @ mu_rev.T**2) - 2 * (X @ mu_rev.T)

    var_rev = np.maximum(np.sum(post * norms, axis=0) / denom_var, min_variance)
    return GaussianMixture(mu_rev, var_rev, pi_rev)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    old_log_lh = None
    new_log_lh = None

    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6 * np.abs(new_log_lh)):
        old_log_lh = new_log_lh

        # e-step
        post, new_log_lh = estep(X, mixture)

        # m-step
        mixture = mstep(X, post, mixture)
    return mixture, post, new_log_lh


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model
    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians
    Returns
        np.ndarray: a (n, d) array with completed data
    """

    [n, d] = np.shape(X)
    k = len(mixture.p)
    NK = np.zeros([n, k])

    ################################# e-step ###########################
    # for i in range(n):
    #     Cu = np.where(X[i] != 0)[0] # return tuple so need [0]
    #     Hu = np.where(X[i] == 0)[0]
    #     d = len(Cu) # dimension decided by non-zero features

    #     for j in range(k):

    #         A = -d/2*np.log(2*np.pi*mixture.var[j]) # (1,1) -> ()
    #         B = np.linalg.norm(X[i,Cu] - mixture.mu[j,Cu]) # (1,1) -> ()
    #         C = -1/2/mixture.var[j] * B**2 # (1,1) -> ()

    #         # K-class Gaussian before mixture
    #         NK[i,j] = A+C # (n,k)

    # # apply weighting to perform Gaussian mixture
    # N_post = NK + np.log(mixture.p) # (n,k)
    # N_post_mix = logsumexp(N_post, axis=1) # (n,1) -> (n,)

    # # log-likelihood
    # # normalized posterior
    # L = np.sum(N_post_mix)
    # N_post_norm = N_post - N_post_mix[:,None]
    #
    # post = np.exp(N_post_norm)

    ##############################################
    [post, L] = estep(X, mixture)
    ##############################################

    # make a copy
    X_pred = np.copy(X)

    # expectation value
    update = post @ mixture.mu  # (n,d)

    # selection Hu
    for i in range(n):
        Cu = np.where(X[i] != 0)[0]  # return tuple so need [0]
        Hu = np.where(X[i] == 0)[0]

        X_pred[i, Hu] = update[i, Hu]

    return X_pred