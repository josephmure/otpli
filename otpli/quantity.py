#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2026

otpli.quantity: weighted quantity-of-interest contract (Q3-A, Q5-A).

The nominal value is computed by `Q(y, 1)`: `PLI(0) = 0` by construction,
continuous curve, single estimator (report rules D1/D2).
"""
import numpy as np


class QuantityOfInterest:
    """
    Weighted contract: `Q(y, w)` on the tilted empirical measure.

    Parameters
    ----------
    y : numpy.ndarray, size `(n,)`
        Model outputs.
    w : numpy.ndarray, size `(n,)`
        Nonnegative weights (not necessarily normalized).

    Returns
    -------
    float
        Estimated quantity.
    """

    def __call__(self, y, w):
        raise NotImplementedError


def _as_1d(y, w):
    y = np.ravel(np.asarray(y, dtype=float))
    w = np.ravel(np.asarray(w, dtype=float))
    if y.shape != w.shape:
        raise ValueError("y and w must have the same size.")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("Negative or non-finite weights.")
    s = np.sum(w)
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError("Null or non-finite weight sum.")
    return y, w


class Quantile(QuantityOfInterest):
    """
    Weighted stepwise `alpha`-quantile (Q5-A).

    `inf{t : F_w(t) >= alpha}` with `F_w` the weighted empirical CDF.
    Equivalent to `numpy` `method="inverted_cdf"`.

    Parameters
    ----------
    alpha : float
        Order in `(0, 1)`.
    """

    def __init__(self, alpha):
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = float(alpha)

    def __call__(self, y, w):
        y, w = _as_1d(y, w)
        order = np.argsort(y, kind="mergesort")
        ys, ws = y[order], w[order]
        cdf = np.cumsum(ws) / np.sum(ws)
        k = int(np.searchsorted(cdf, self.alpha, side="left"))
        k = min(max(k, 0), len(ys) - 1)
        return float(ys[k])


class SuperQuantile(QuantityOfInterest):
    """
    Coherent weighted `E[Y | Y > q_alpha]` superquantile (fixes D1).

    The threshold `q` and the mean are computed with the *same* weights.

    Parameters
    ----------
    alpha : float
        Order in `(0, 1)`.
    """

    def __init__(self, alpha):
        self._q = Quantile(alpha)
        self.alpha = self._q.alpha

    def __call__(self, y, w):
        y, w = _as_1d(y, w)
        q = self._q(y, w)
        mask = y >= q
        if not np.any(mask):
            return float(np.max(y))
        return float(np.sum(y[mask] * w[mask]) / np.sum(w[mask]))


class Mean(QuantityOfInterest):
    """Weighted mean (generic-QoI example, meeting notes)."""

    def __call__(self, y, w):
        y, w = _as_1d(y, w)
        return float(np.sum(y * w) / np.sum(w))


class Variance(QuantityOfInterest):
    """Weighted variance (generic-QoI example, meeting notes)."""

    def __call__(self, y, w):
        y, w = _as_1d(y, w)
        m = np.sum(y * w) / np.sum(w)
        return float(np.sum(w * (y - m) ** 2) / np.sum(w))
