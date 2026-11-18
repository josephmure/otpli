#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2026

otpli.perturbation: weights, KL and ESS of Gaussian tilts (Q1).

MEAN mode: `U_i -> N(delta, 1)`, `w = exp(delta * u)`, `KL = delta**2 / 2`.
VAR mode: `U_i -> N(0, tau**2)` with `tau**2 = r`, `KL = log(tau) + 1/(2 tau**2) - 1/2`.
"""
import numpy as np

from .utils import ess_ratio_from_logweights


def logweights_mean(u_col, delta):
    """
    Log-weights of the mean tilt.

    Parameters
    ----------
    u_col : numpy.ndarray, size `(n,)`
        Standard coordinate of the perturbed variable.
    delta : float
        Signed shift in standard space.

    Returns
    -------
    numpy.ndarray
        `log w = delta * u` (constant `-delta**2/2` omitted: normalized
        in the estimator, see report §5.5).
    """
    return np.asarray(u_col, dtype=float) * float(delta)


def logweights_var(u_col, ratio):
    """
    Log-weights of the Gaussian scale tilt (VAR mode reinstated on 11/16).

    `w ~ (1/tau) * exp(a * u**2)` with `a = (1 - 1/tau**2)/2`, `tau = sqrt(r)`.

    Parameters
    ----------
    u_col : numpy.ndarray, size `(n,)`
    ratio : float
        `r = V / sigma0**2 > 0` strict, `r = 1` is nominal.

    Returns
    -------
    numpy.ndarray
        Unnormalized log-weights.
    """
    r = float(ratio)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("Variance ratio r must be > 0 strict.")
    tau2 = r
    a = 0.5 * (1.0 - 1.0 / tau2)
    u = np.asarray(u_col, dtype=float)
    return a * u * u - 0.5 * np.log(tau2)


def kl_mean(delta):
    """
    `KL(nominal || perturbed)` divergence of the mean tilt.

    Parameters
    ----------
    delta : float

    Returns
    -------
    float
        `delta**2 / 2`.
    """
    return 0.5 * float(delta) ** 2


def kl_var(ratio):
    """
    `KL(nominal || perturbed)` divergence of the scale tilt.

    `N(0,1) -> N(0, tau**2)`: `log(tau) + 1/(2 tau**2) - 1/2`.

    Parameters
    ----------
    ratio : float
        `r = tau**2 > 0`.

    Returns
    -------
    float
    """
    r = float(ratio)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("Variance ratio r must be > 0 strict.")
    tau = float(np.sqrt(r))
    return float(np.log(tau) + 1.0 / (2.0 * r) - 0.5)


def normalized_weights(logw):
    """
    Normalize log-weights via `logsumexp` (stable, vectorizable).

    Parameters
    ----------
    logw : numpy.ndarray, size `(n,)`

    Returns
    -------
    numpy.ndarray
        Weights summing to 1.
    """
    logw = np.asarray(logw, dtype=float)
    m = np.max(logw)
    w = np.exp(logw - m)
    s = np.sum(w)
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError("Weight normalization failed.")
    return w / s


def ess_ratio(logw):
    """
    `ESS / N` from log-weights (§5.5 diagnostic).

    Parameters
    ----------
    logw : numpy.ndarray

    Returns
    -------
    float
    """
    return ess_ratio_from_logweights(np.asarray(logw, dtype=float))
