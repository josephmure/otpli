#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) EDF 2026

otpli.indices: Gaussian PLI engine (Q1-Q7).

- `compute_mean(X, Y, shifts)`: mean tilt, signed `delta` parameter.
- `compute_var(X, Y, ratios)`: scale tilt, `r > 0` parameter.
- Joint `(U, Y)` bootstrap by per-replicate ratio (Q6-A),
  `n_bootstrap=1000` by default, `0` = disabled (Q7-B).
- `absolute_variation=False` by default: dimensionless relative
  variation (`Qd/Q0 - 1`, the norm, comparable across quantities);
  `True` selects the absolute fallback (`Qd - Q0`) for null references
  (Q4-B).
- Single stepwise quantile for nominal and perturbed (Q5-A): `PLI = 0`
  exact at the nominal point by construction.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import openturns as ot
from numpy.typing import ArrayLike

from .perturbation import (
    ess_ratio,
    kl_mean,
    kl_var,
    logweights_mean,
    logweights_var,
    normalized_weights,
)
from .quantity import QuantityOfInterest
from .utils import physical_to_standard, warn_if_low_ess


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
        Number of replicates (Q7-B: default 1000, 0 = no confidence interval).
    confidence : float
        Confidence level in `(0, 1)` (default 0.95).
    absolute_variation : bool
        `False` (default): relative variation `Qd/Q0 - 1`, dimensionless
        and comparable across quantities of different natures (the norm).
        `True`: absolute fallback `Qd - Q0`, for null references (Q4-B).
    ess_threshold : float
        Indicative `ESS/N` warning threshold (default 0.01, §5.5).
    seed : int or None
        Joint bootstrap seed.
    """

    def __init__(
        self,
        distribution: ot.Distribution,
        quantity: QuantityOfInterest,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        absolute_variation: bool = False,
        ess_threshold: float = 0.01,
        seed: int | None = 0,
    ) -> None:
        if not 0.0 < float(confidence) < 1.0:
            raise ValueError(f"confidence must be in (0, 1): got {confidence!r}.")
        if int(n_bootstrap) < 0:
            raise ValueError(f"n_bootstrap must be >= 0: got {n_bootstrap!r}.")
        self.distribution = distribution
        self.quantity = quantity
        self.n_bootstrap = int(n_bootstrap)
        self.confidence = float(confidence)
        self.absolute_variation = bool(absolute_variation)
        self.ess_threshold = float(ess_threshold)
        self.seed = seed

    # -- internals ----------------------------------------------------
    def _prepare(self, X: ot.Sample, Y: ot.Sample) -> tuple[np.ndarray, np.ndarray]:
        Xn = np.asarray(X)
        Yn = np.asarray(Y)
        if Xn.shape[0] != Yn.shape[0]:
            raise ValueError(
                "X and Y must have the same size: "
                f"got size(X)={Xn.shape[0]}, size(Y)={Yn.shape[0]}."
            )
        if Yn.shape[1] != 1:
            raise ValueError(
                f"Y must have dimension 1: got dim={Yn.shape[1]} "
                f"(shape={Yn.shape})."
            )
        U = physical_to_standard(self.distribution, ot.Sample(Xn.tolist()))
        y = Yn[:, 0]
        return U, y

    def _pli_from_quantities(self, qd: float, q0: float) -> float:
        if not self.absolute_variation:
            if abs(float(q0)) == 0.0:
                return float("nan")
            return float(qd) / float(q0) - 1.0
        return float(qd) - float(q0)

    def _run_grid(
        self,
        U: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray,
        logw_fn: Callable[[np.ndarray, float], np.ndarray],
        kl_fn: Callable[[float], float],
        mode: str,
    ) -> PerturbedLawIndicesResult:
        d = U.shape[1]
        K = len(grid)
        N = len(y)
        w0 = normalized_weights(np.zeros(N))
        # 1D QoI: single nominal call, repeated for each input (shape (d,))
        q0 = np.full(d, self.quantity(y, w0))
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
    def compute_mean(
        self, X: ot.Sample, Y: ot.Sample, shifts: ArrayLike
    ) -> PerturbedLawIndicesResult:
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
        return self._run_grid(
            U, y, np.asarray(shifts, dtype=float), logweights_mean, kl_mean, "mean"
        )

    def compute_var(
        self, X: ot.Sample, Y: ot.Sample, ratios: ArrayLike
    ) -> PerturbedLawIndicesResult:
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
            raise ValueError(
                "Ratios r must be > 0 strict: "
                f"got min={float(np.min(ratios))!r} in {np.asarray(ratios).ravel()[:8]}."
            )
        U, y = self._prepare(X, Y)
        return self._run_grid(U, y, ratios, logweights_var, kl_var, "var")
