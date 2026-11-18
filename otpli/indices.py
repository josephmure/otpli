#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2026

otpli.indices: Gaussian PLI engine (Q1-Q7).

- `compute_mean(X, Y, shifts)`: mean tilt, signed `delta` parameter.
- `compute_var(X, Y, ratios)`: scale tilt, `r > 0` parameter.
- Joint `(U, Y)` bootstrap by per-replicate ratio (Q6-A),
  `n_bootstrap=1000` by default, `0` = disabled (Q7-B).
- `variation="relative"` (`Qd/Q0 - 1`, default) or `"absolute"` (`Qd - Q0`, Q4-B).
- Single stepwise quantile for nominal and perturbed (Q5-A): `PLI = 0`
  exact at the nominal point by construction.
"""
from dataclasses import dataclass

import numpy as np
import openturns as ot

from .perturbation import (
    ess_ratio,
    kl_mean,
    kl_var,
    logweights_mean,
    logweights_var,
    normalized_weights,
)
from .utils import physical_to_standard, to_numpy, warn_if_low_ess


@dataclass
class PerturbedLawIndicesResult:
    """PLI result: four blocks + diagnostics (Q1, §5.5)."""

    mode: str  # "mean" or "var"
    grid: np.ndarray  # (K,): delta shifts or r ratios
    pli: np.ndarray  # (K, d)
    pli_low: np.ndarray | None  # (K, d) or None if n_bootstrap=0
    pli_high: np.ndarray | None
    quantity: np.ndarray  # (K, d): perturbed Q
    quantity_low: np.ndarray | None
    quantity_high: np.ndarray | None
    quantity_nominal: np.ndarray  # (d,)
    ess_ratio: np.ndarray  # (K, d)
    kl: np.ndarray  # (K,): diagnostic, see Q1


class PerturbedLawIndices:
    """
    Gaussian PLI engine on an existing sample (no model re-evaluation).

    Parameters
    ----------
    distribution : :py:class:`openturns.Distribution`
        Joint input distribution (v1: independent, Gaussian standard).
    quantity : callable
        Quantity `Q(y, w)` of the weighted contract (Q3-A).
    n_bootstrap : int
        Number of replicates (Q7-B: default 1000, 0 = no CI).
    confidence : float
        Confidence level in `(0, 1)` (default 0.95).
    variation : str
        `"relative"` (`Qd/Q0 - 1`, default) or `"absolute"` (`Qd - Q0`, Q4-B).
    ess_threshold : float
        Indicative `ESS/N` warning threshold (default 0.01, §5.5).
    seed : int or None
        Joint bootstrap seed.
    """

    def __init__(
        self,
        distribution,
        quantity,
        n_bootstrap=1000,
        confidence=0.95,
        variation="relative",
        ess_threshold=0.01,
        seed=0,
    ):
        if variation not in ("relative", "absolute"):
            raise ValueError("variation must be 'relative' or 'absolute'.")
        if not 0.0 < float(confidence) < 1.0:
            raise ValueError("confidence must be in (0, 1).")
        if int(n_bootstrap) < 0:
            raise ValueError("n_bootstrap must be >= 0.")
        self.distribution = distribution
        self.quantity = quantity
        self.n_bootstrap = int(n_bootstrap)
        self.confidence = float(confidence)
        self.variation = variation
        self.ess_threshold = float(ess_threshold)
        self.seed = seed

    # -- internals ----------------------------------------------------
    def _prepare(self, X, Y):
        Xn = to_numpy(X)
        Yn = to_numpy(Y)
        if Xn.shape[0] != Yn.shape[0]:
            raise ValueError("X and Y must share the same N.")
        if Yn.shape[1] != 1:
            raise ValueError("1D Y only in this sketch.")
        U = physical_to_standard(self.distribution, ot.Sample(Xn.tolist()))
        y = Yn[:, 0]
        return U, y

    def _pli_from_quantities(self, qd, q0):
        if self.variation == "relative":
            if abs(float(q0)) == 0.0:
                return float("nan")
            return float(qd) / float(q0) - 1.0
        return float(qd) - float(q0)

    def _run_grid(self, U, y, grid, logw_fn, kl_fn, mode):
        d = U.shape[1]
        K = len(grid)
        N = len(y)
        w0 = normalized_weights(np.zeros(N))
        q0 = np.array([self.quantity(y, w0) for _ in range(d)])
        # 1D QoI: same nominal for each input before perturbation
        q0 = np.repeat(q0[:1], d)
        pli = np.zeros((K, d))
        quant = np.zeros((K, d))
        ess = np.zeros((K, d))
        kl = np.array([kl_fn(g) for g in grid], dtype=float)
        boot_qd = None
        boot_t = None
        if self.n_bootstrap > 0:
            boot_qd = np.zeros((self.n_bootstrap, K, d))
            boot_t = np.zeros((self.n_bootstrap, K, d))
        rng = np.random.default_rng(self.seed)
        for j, g in enumerate(grid):
            for i in range(d):
                logw = logw_fn(U[:, i], g)
                w = normalized_weights(logw)
                ess[j, i] = ess_ratio(logw)
                qd = self.quantity(y, w)
                quant[j, i] = qd
                pli[j, i] = self._pli_from_quantities(qd, q0[i])
        warn_if_low_ess(np.asarray(grid), np.mean(ess, axis=1), self.ess_threshold)
        if self.n_bootstrap > 0:
            idx = rng.integers(0, N, size=(self.n_bootstrap, N))
            alpha_low = (1.0 - self.confidence) / 2.0
            alpha_high = (1.0 + self.confidence) / 2.0
            for b in range(self.n_bootstrap):
                yb, Ub = y[idx[b]], U[idx[b]]
                q0b = self.quantity(yb, normalized_weights(np.zeros(N)))
                for j, g in enumerate(grid):
                    for i in range(d):
                        wb = normalized_weights(logw_fn(Ub[:, i], g))
                        qdb = self.quantity(yb, wb)
                        boot_qd[b, j, i] = qdb
                        boot_t[b, j, i] = self._pli_from_quantities(qdb, q0b)
            pli_low = np.quantile(boot_t, alpha_low, axis=0)
            pli_high = np.quantile(boot_t, alpha_high, axis=0)
            q_low = np.quantile(boot_qd, alpha_low, axis=0)
            q_high = np.quantile(boot_qd, alpha_high, axis=0)
        else:
            pli_low = pli_high = q_low = q_high = None
        return PerturbedLawIndicesResult(
            mode=mode,
            grid=np.asarray(grid, dtype=float),
            pli=pli,
            pli_low=pli_low,
            pli_high=pli_high,
            quantity=quant,
            quantity_low=q_low,
            quantity_high=q_high,
            quantity_nominal=q0,
            ess_ratio=ess,
            kl=kl,
        )

    # -- API ----------------------------------------------------------
    def compute_mean(self, X, Y, shifts):
        """
        PLI under mean tilt: signed `delta` in standard space (Q1-A).

        Parameters
        ----------
        X : :py:class:`openturns.Sample`, size `(N, d)`
        Y : :py:class:`openturns.Sample`, size `(N, 1)`
        shifts : array_like, size `(K,)`
            Grid covering `delta < 0` and `delta > 0`.

        Returns
        -------
        PerturbedLawIndicesResult
        """
        U, y = self._prepare(X, Y)
        return self._run_grid(U, y, np.asarray(shifts, dtype=float),
                             logweights_mean, kl_mean, "mean")

    def compute_var(self, X, Y, ratios):
        """
        PLI under Gaussian scale tilt: `r = V / sigma0**2` (Q1-A1).

        Parameters
        ----------
        X : :py:class:`openturns.Sample`, size `(N, d)`
        Y : :py:class:`openturns.Sample`, size `(N, 1)`
        ratios : array_like, size `(K,)`
            Grid with `r > 0` covering `r < 1` and `r > 1`; `r = 1` is nominal.

        Returns
        -------
        PerturbedLawIndicesResult
        """
        ratios = np.asarray(ratios, dtype=float)
        if np.any(ratios <= 0.0):
            raise ValueError("Ratios r must be > 0 strict.")
        U, y = self._prepare(X, Y)
        return self._run_grid(U, y, ratios, logweights_var, kl_var, "var")
