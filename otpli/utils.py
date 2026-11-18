#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2026

otpli.utils: guards and conversions.

Covered decisions: Q2 (independence), Q1 (KL/ESS), Q6 (joint bootstrap).
"""
import warnings

import numpy as np
import openturns as ot


def check_independent(distribution):
    """
    Require independent inputs (Q2-A).

    Parameters
    ----------
    distribution : :py:class:`openturns.Distribution`
        Joint input distribution.

    Raises
    ------
    ValueError
        If the copula is not independent.
    """
    if not distribution.hasIndependentCopula():
        raise ValueError(
            "otpli v1 is restricted to independent inputs "
            "(Q2-A): non-independent copula rejected."
        )


def check_standard_normal(distribution):
    """
    Check that the standard distribution is standardized Gaussian.

    The proxy returned by `getStandardDistribution` is not reliable alone:
    the actual implementation class name is checked (see report §5.1).

    Parameters
    ----------
    distribution : :py:class:`openturns.Distribution`

    Raises
    ------
    ValueError
        If the standard distribution is not `Normal`.
    """
    std = distribution.getStandardDistribution()
    try:
        classname = std.getImplementation().getClassName()
    except Exception:
        classname = std.getClassName() if hasattr(std, "getClassName") else ""
    if classname != "Normal":
        raise ValueError(
            "otpli requires a Gaussian standard distribution "
            f"(Q1, Gaussian KL frame): standard={classname!r} rejected."
        )
    if std.getDimension() != distribution.getDimension():
        raise ValueError("Inconsistent standard distribution dimension.")


def to_numpy(sample):
    """
    Convert an `ot.Sample` to a 2D `numpy.ndarray`.

    Parameters
    ----------
    sample : :py:class:`openturns.Sample`

    Returns
    -------
    numpy.ndarray
        Array of size `(n, d)`.
    """
    return np.asarray(sample)


def to_sample(array, descriptions=None):
    """
    Convert a 2D numpy array to an `ot.Sample`.

    Parameters
    ----------
    array : numpy.ndarray
    descriptions : list of str, optional

    Returns
    -------
    :py:class:`openturns.Sample`
    """
    sample = ot.Sample(np.ascontiguousarray(array).tolist())
    if descriptions is not None:
        sample.setDescription(list(descriptions))
    return sample


def physical_to_standard(distribution, X):
    """
    Apply the isoprobabilistic transformation `U = T(X)`.

    Parameters
    ----------
    distribution : :py:class:`openturns.Distribution`
    X : :py:class:`openturns.Sample`

    Returns
    -------
    numpy.ndarray
        Standard coordinates of size `(n, d)`.
    """
    check_independent(distribution)
    check_standard_normal(distribution)
    T = distribution.getIsoProbabilisticTransformation()
    return to_numpy(T(X))


def ess_ratio_from_logweights(logw):
    """
    Normalized effective sample size.

    Parameters
    ----------
    logw : numpy.ndarray
        Unnormalized log-weights of size `(n,)`.

    Returns
    -------
    float
        `ESS / N` in `(0, 1]`.
    """
    m = np.max(logw)
    w = np.exp(logw - m)
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    if s1 == 0.0 or s2 == 0.0 or not np.isfinite(s1 + s2):
        return 0.0
    return float(s1 * s1 / (len(w) * s2))


def warn_if_low_ess(grid, ess_ratio, threshold=0.01):
    """
    Warn when `ESS / N` falls below the threshold (Q1, §5.5).

    No hard thresholding on the grid: only validity is flagged.

    Parameters
    ----------
    grid : numpy.ndarray
        Grid (`delta` or `r`).
    ess_ratio : numpy.ndarray
        `ESS / N` of the same size.
    threshold : float
        Indicative threshold (default `0.01`, see report §5.5).
    """
    bad = np.asarray(ess_ratio) < threshold
    if np.any(bad):
        vals = np.asarray(grid)[np.atleast_1d(bad)]
        warnings.warn(
            f"ESS/N < {threshold} for {int(np.sum(bad))} grid point(s) "
            f"(e.g. {np.asarray(vals).ravel()[:4]}): "
            "PLI curve potentially noisy.",
            UserWarning,
            stacklevel=3,
        )
