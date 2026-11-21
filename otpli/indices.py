r"""
Copyright (C) EDF 2026

otpli.indices: Gaussian PLI engine.

- `compute_mean(X, Y, shifts)`: mean tilt, signed :math:`\delta` parameter.
- `compute_var(X, Y, ratios)`: scale tilt, :math:`r > 0` parameter.
- Joint `(U, Y)` bootstrap by per-replicate ratio,
  `n_bootstrap=1000` by default, `0` = disabled.
- `absolute_variation=False` by default: dimensionless relative
  variation (:math:`Q_d/Q_0 - 1`, the norm, comparable across quantities);
  `True` selects the absolute fallback (:math:`Q_d - Q_0`) for null references.
- Single stepwise quantile for nominal and perturbed: :math:`\mathrm{PLI} = 0`
  exact at the nominal point by construction.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import openturns as ot
from numpy.typing import ArrayLike

from .perturbation import (
    kl_mean,
    kl_var,
    logweights_mean,
    logweights_var,
    normalized_weights,
)
from .quantity import DomainProbability
from .utils import (
    check_independent,
    check_standard_normal,
    ess_ratio_from_logweights,
    physical_to_standard,
    warn_if_low_ess,
)


@dataclass
class PerturbedLawIndicesResult:
    """PLI result: four blocks + diagnostics."""

    mode: Literal["mean", "var"]
    grid: np.ndarray  # (K,): delta shifts or r ratios
    pli: np.ndarray  # (K, d)
    pli_low: np.ndarray | None  # (K, d); bootstrap, or analytic CLT for DomainProbability
    pli_high: np.ndarray | None
    quantity: np.ndarray  # (K, d): perturbed Q
    quantity_low: np.ndarray | None
    quantity_high: np.ndarray | None
    quantity_nominal: np.ndarray  # (d,) repeated scalar per input
    ess_ratio: np.ndarray  # (K, d)
    kl: np.ndarray  # (K,): KL(P0 || Pdelta) diagnostic


class PerturbedLawIndices:
    r"""
    Gaussian PLI engine on an existing sample (no model re-evaluation).

    Parameters
    ----------
    distribution : :py:class:`openturns.Distribution`
        Joint input distribution (v1: independent, Gaussian standard).
    quantity : callable
        Quantity :math:`Q(y, w)` with `y, w` 1D arrays size `(N,)` and
        float return. Any callable with this signature is accepted.
    n_bootstrap : int
        Number of replicates (default 1000, 0 = no bootstrap confidence
        interval). Must be 0 with a `DomainProbability` quantity:
        bootstrap is disabled for this QoI, whose intervals always come
        from the closed-form CLT.
    confidence : float
        Confidence level in `(0, 1)` (default 0.95). Ignored if
        `n_bootstrap=0`, except for the analytic `DomainProbability`
        intervals which always use it.
    absolute_variation : bool
        `False` (default): relative variation :math:`Q_d/Q_0 - 1`, dimensionless
        and comparable across quantities of different natures (recommended).
        `True`: absolute fallback :math:`Q_d - Q_0`, for null references.
    ess_threshold : float
        Indicative :math:`\mathrm{ESS}/N` warning threshold (default 0.01).
    seed : int or None
        Joint bootstrap seed (default 0: reproducible; None: non-deterministic).
    n_jobs : int
        Bootstrap workers via stdlib threads (default -1: all available cores;
        1: sequential; >1: `ThreadPoolExecutor`, same results,
        seed-independent order).
    """

    def __init__(
        self,
        distribution: ot.Distribution,
        quantity: Callable[[np.ndarray, np.ndarray], float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        absolute_variation: bool = False,
        ess_threshold: float = 0.01,
        seed: int | None = 0,
        n_jobs: int = -1,
    ) -> None:
        if not 0.0 < float(confidence) < 1.0:
            raise ValueError(f"confidence must be in (0, 1): got {confidence!r}.")
        if int(n_bootstrap) < 0:
            raise ValueError(f"n_bootstrap must be >= 0: got {n_bootstrap!r}.")
        try:
            thr = float(ess_threshold)
        except (TypeError, ValueError):
            raise ValueError(
                f"ess_threshold must be in [0, 1]: got {ess_threshold!r}."
            ) from None
        if not 0.0 <= thr <= 1.0:
            raise ValueError(
                f"ess_threshold must be in [0, 1]: got {ess_threshold!r}."
            )
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be int or None: got {seed!r}.")
        if isinstance(quantity, DomainProbability) and int(n_bootstrap) != 0:
            raise ValueError(
                "Bootstrap is disabled for DomainProbability quantities: "
                "confidence intervals come from the closed-form CLT, "
                f"set n_bootstrap=0 (got n_bootstrap={n_bootstrap!r})."
            )
        try:
            nj = int(n_jobs)
            if float(n_jobs) != nj:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(
                f"n_jobs must be an integer >= 1 or -1: got {n_jobs!r}."
            ) from None
        if nj == -1:
            import os

            nj = os.cpu_count() or 1
        if nj < 1:
            raise ValueError(
                f"n_jobs must be an integer >= 1 or -1: got {n_jobs!r}."
            )
        # Fail-fast: reject unsupported distributions at construction (P7).
        check_independent(distribution)
        check_standard_normal(distribution)
        self.distribution = distribution
        self.quantity = quantity
        self.n_bootstrap = int(n_bootstrap)
        self.confidence = float(confidence)
        self.absolute_variation = bool(absolute_variation)
        self.ess_threshold = thr
        self.seed = None if seed is None else int(seed)
        self.n_jobs = nj

    # -- internals ----------------------------------------------------
    def _prepare(self, X: ot.Sample, Y: ot.Sample) -> tuple[np.ndarray, np.ndarray]:
        """Validate `(X, Y)` and map `X` to standard space.

        Parameters
        ----------
        X : array-like, size `(N, d)`
            Physical inputs; `d` must match the distribution dimension.
        Y : array-like, size `(N,)` or `(N, 1)`
            Scalar model outputs, must be finite.

        Returns
        -------
        U : numpy.ndarray, size `(N, d)`
            Standard-space coordinates :math:`U = T(X)`.
        y : numpy.ndarray, size `(N,)`
            Flattened outputs.
        """
        try:
            Xn = np.asarray(X, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"X must be convertible to a 2D float array: got {type(X).__name__}."
            ) from None
        try:
            Yn = np.asarray(Y, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"Y must be convertible to a float array: got {type(Y).__name__}."
            ) from None
        if Xn.ndim != 2:
            raise ValueError(
                f"X must be 2D (N, d): got shape={Xn.shape}."
            )
        if Xn.shape[0] == 0:
            raise ValueError("X must be non-empty: got N=0.")
        if Xn.shape[1] != self.distribution.getDimension():
            raise ValueError(
                "X dimension must match distribution dimension: "
                f"got dim(X)={Xn.shape[1]}, "
                f"dim(distribution)={self.distribution.getDimension()}."
            )
        if Yn.ndim == 1:
            Yn = Yn[:, None]
        if Yn.ndim != 2 or Yn.shape[1] != 1:
            raise ValueError(
                f"Y must have dimension 1: got shape={Yn.shape} "
                "(accepted: (N,) or (N, 1))."
            )
        if Xn.shape[0] != Yn.shape[0]:
            raise ValueError(
                "X and Y must have the same size: "
                f"got size(X)={Xn.shape[0]}, size(Y)={Yn.shape[0]}."
            )
        # T accepts Sample / ndarray / list: pass the validated array directly.
        U = physical_to_standard(self.distribution, Xn)
        y = Yn[:, 0]
        return U, y

    @staticmethod
    def _as_grid(values: ArrayLike, name: str, *, positive: bool = False) -> np.ndarray:
        """Validate a perturbation grid as a finite 1D float array.

        Parameters
        ----------
        values : array-like
            Candidate grid values.
        name : str
            Parameter name used in error messages.
        positive : bool
            If True, require strictly positive values (var ratios).

        Returns
        -------
        numpy.ndarray
            Flattened finite grid.
        """
        try:
            g = np.asarray(values, dtype=float).ravel()
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be a 1D finite float array: got {values!r}."
            ) from None
        if g.size == 0:
            raise ValueError(f"{name} must be non-empty.")
        if not np.all(np.isfinite(g)):
            n_bad = int(np.sum(~np.isfinite(g)))
            raise ValueError(
                f"{name} must be finite: got n_nonfinite={n_bad} in {g[:8]!r}."
            )
        if positive and np.any(g <= 0.0):
            raise ValueError(
                f"{name} must be > 0: "
                f"got min={float(np.min(g))!r} in {g[:8]!r}."
            )
        return g

    def _pli_from_quantities(self, qd: float, q0: float) -> float:
        """Compute one PLI value from perturbed and nominal quantities.

        Parameters
        ----------
        qd : float
            Perturbed quantity :math:`Q_d`.
        q0 : float
            Nominal quantity :math:`Q_0`.

        Returns
        -------
        float
            :math:`Q_d/Q_0 - 1` in relative mode, :math:`Q_d - Q_0` in absolute mode.
            In relative mode a null :math:`Q_0` naturally raises
            `ZeroDivisionError`; use `absolute_variation=True` for null
            references.
        """
        if not self.absolute_variation:
            return float(qd) / float(q0) - 1.0
        return float(qd) - float(q0)

    def _bootstrap_intervals(
        self,
        U: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray,
        logw_fn: Callable[[np.ndarray, float], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Joint `(U, Y)` bootstrap confidence intervals.

        Replicates are seeded streaming via `SeedSequence.spawn`
        (no `(B, N)` index array); results are identical whatever
        `n_jobs` is.

        Parameters
        ----------
        U : numpy.ndarray, size `(N, d)`
            Standard-space inputs.
        y : numpy.ndarray, size `(N,)`
            Model outputs.
        grid : numpy.ndarray, size `(K,)`
            Perturbation grid (shifts or ratios).
        logw_fn : callable
            `logw_fn(u_col, g)` returning unnormalized log-weights.

        Returns
        -------
        pli_low, pli_high : numpy.ndarray, size `(K, d)`
            PLI confidence bounds.
        q_low, q_high : numpy.ndarray, size `(K, d)`
            Perturbed-quantity confidence bounds.
        """
        N = len(y)
        d = U.shape[1]
        K = len(grid)
        boot_qd = np.zeros((self.n_bootstrap, K, d))
        boot_t = np.zeros((self.n_bootstrap, K, d))
        alpha_low = (1.0 - self.confidence) / 2.0
        alpha_high = (1.0 + self.confidence) / 2.0
        if self.seed is None:
            child_seqs: list[np.random.SeedSequence | None] = [None] * self.n_bootstrap
        else:
            child_seqs = list(
                np.random.SeedSequence(int(self.seed)).spawn(self.n_bootstrap)
            )
        w0b = normalized_weights(np.zeros(N))

        def _one(b: int) -> tuple[np.ndarray, np.ndarray]:
            """Compute one joint bootstrap replicate.

            Parameters
            ----------
            b : int
                Replicate index (selects the child seed).

            Returns
            -------
            qd_row : numpy.ndarray, size `(K, d)`
                Perturbed quantities for this replicate.
            t_row : numpy.ndarray, size `(K, d)`
                PLI values for this replicate.
            """
            if child_seqs[b] is None:
                rb = np.random.default_rng()
            else:
                rb = np.random.default_rng(child_seqs[b])
            idx_b = rb.integers(0, N, size=N)
            yb, Ub = y[idx_b], U[idx_b]
            q0b = self.quantity(yb, w0b)
            qd_row = np.empty((K, d))
            t_row = np.empty((K, d))
            for j, g in enumerate(grid):
                for i in range(d):
                    wb = normalized_weights(logw_fn(Ub[:, i], g))
                    qdb = self.quantity(yb, wb)
                    qd_row[j, i] = qdb
                    t_row[j, i] = self._pli_from_quantities(qdb, q0b)
            return qd_row, t_row

        if self.n_jobs == 1:
            for b in range(self.n_bootstrap):
                boot_qd[b], boot_t[b] = _one(b)
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                for b, (qd_b, t_b) in enumerate(ex.map(_one, range(self.n_bootstrap))):
                    boot_qd[b], boot_t[b] = qd_b, t_b
        return (
            np.quantile(boot_t, alpha_low, axis=0),
            np.quantile(boot_t, alpha_high, axis=0),
            np.quantile(boot_qd, alpha_low, axis=0),
            np.quantile(boot_qd, alpha_high, axis=0),
        )

    def _analytic_intervals_domain(
        self,
        U: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray,
        logw_fn: Callable[[np.ndarray, float], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""CLT-based confidence intervals for `DomainProbability` (no bootstrap).

        Closed form from Lemaitre et al. (2015): the perturbed failure
        probability estimator and the nominal one are jointly asymptotically
        normal with covariance :math:`\Sigma`, propagated to the PLI by the
        delta method. With :math:`I_i = \mathbb{1}_D(y_i)`,
        :math:`r_i` the density ratios rescaled to :math:`\mathbb{E}[r] = 1`,
        :math:`P_0 = \bar I` and :math:`P_\delta = \overline{rI}`:

        :math:`\Sigma = \begin{pmatrix} P_0(1-P_0) &
        P_\delta(1-P_0) \\ P_\delta(1-P_0) & s^2 \end{pmatrix}`,
        :math:`s^2 = \overline{(rI)^2} - P_\delta^2`,
        :math:`\nabla S = (-P_\delta/P_0^2,\ 1/P_0)` in relative mode,
        :math:`(-1,\ 1)` in absolute mode,
        :math:`\hat\sigma_S = \sqrt{\nabla S^\top \Sigma \nabla S / N}`.

        Bounds are :math:`\pm z_{(1+c)/2}\,\hat\sigma` with `z` the standard
        normal quantile at `self.confidence`, centered on the point
        estimates. Cost is :math:`O(K \cdot d \cdot N)` with no resampling.

        Parameters
        ----------
        U : numpy.ndarray, size `(N, d)`
            Standard-space inputs.
        y : numpy.ndarray, size `(N,)`
            Model outputs.
        grid : numpy.ndarray, size `(K,)`
            Perturbation grid (shifts or ratios).
        logw_fn : callable
            `logw_fn(u_col, g)` returning unnormalized log-weights
            (multiplicative constants cancel out).

        Returns
        -------
        pli_low, pli_high : numpy.ndarray, size `(K, d)`
            PLI confidence bounds.
        q_low, q_high : numpy.ndarray, size `(K, d)`
            Perturbed-probability confidence bounds.
        """
        if not isinstance(self.quantity, DomainProbability):
            raise TypeError(
                "Analytic CLT intervals require a DomainProbability quantity: "
                f"got {type(self.quantity).__name__}."
            )
        N = len(y)
        d = U.shape[1]
        K = len(grid)
        z = ot.DistFunc.qNormal((1.0 + self.confidence) / 2.0)
        mask = self.quantity.event_mask(y)
        I = mask.astype(float)
        P0 = float(np.mean(I))
        pli_low = np.zeros((K, d))
        pli_high = np.zeros((K, d))
        q_low = np.zeros((K, d))
        q_high = np.zeros((K, d))
        for j, g in enumerate(grid):
            for i in range(d):
                w = normalized_weights(logw_fn(U[:, i], g))
                r = N * np.asarray(w, dtype=float)
                rI = r * I
                Pd = float(np.mean(rI))
                s2 = max(float(np.mean(rI * rI) - Pd * Pd), 0.0)
                half_q = z * float(np.sqrt(s2 / N))
                qd = float(np.sum(w[mask]))
                q_low[j, i] = qd - half_q
                q_high[j, i] = qd + half_q
                if self.absolute_variation:
                    grad = np.array([-1.0, 1.0])
                else:
                    grad = np.array([-Pd / P0**2, 1.0 / P0])
                cov = np.array(
                    [
                        [P0 * (1.0 - P0), Pd * (1.0 - P0)],
                        [Pd * (1.0 - P0), s2],
                    ]
                )
                half_pli = z * float(np.sqrt(max(grad @ cov @ grad, 0.0) / N))
                qd_pli = self._pli_from_quantities(qd, P0)
                pli_low[j, i] = qd_pli - half_pli
                pli_high[j, i] = qd_pli + half_pli
        return pli_low, pli_high, q_low, q_high

    def _run_grid(
        self,
        U: np.ndarray,
        y: np.ndarray,
        grid: np.ndarray,
        logw_fn: Callable[[np.ndarray, float], np.ndarray],
        kl_fn: Callable[[float], float],
        mode: Literal["mean", "var"],
    ) -> PerturbedLawIndicesResult:
        r"""Run the full PLI grid in standard space, with joint bootstrap.

        Parameters
        ----------
        U : numpy.ndarray, size `(N, d)`
            Standard-space inputs.
        y : numpy.ndarray, size `(N,)`
            Model outputs.
        grid : numpy.ndarray, size `(K,)`
            Perturbation grid (shifts or ratios).
        logw_fn : callable
            `logw_fn(u_col, g)` returning unnormalized log-weights.
        kl_fn : callable
            `kl_fn(g)` returning :math:`\mathrm{KL}(P_0 \parallel P_\delta)`.
        mode : {"mean", "var"}
            Tilt mode, stored in the result.

        Returns
        -------
        PerturbedLawIndicesResult
            PLI, quantities, :math:`\mathrm{ESS}/N`,
            :math:`\mathrm{KL}` and confidence intervals (bootstrap, or
            closed-form CLT for `DomainProbability` when `n_bootstrap=0`).
        """
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
        for j, g in enumerate(grid):
            for i in range(d):
                logw = logw_fn(U[:, i], g)
                w = normalized_weights(logw)
                ess[j, i] = ess_ratio_from_logweights(logw)
                qd = self.quantity(y, w)
                quant[j, i] = qd
                pli[j, i] = self._pli_from_quantities(qd, q0[i])
        warn_if_low_ess(np.asarray(grid), ess, self.ess_threshold)
        if self.n_bootstrap > 0:
            pli_low, pli_high, q_low, q_high = self._bootstrap_intervals(
                U, y, grid, logw_fn
            )
        elif isinstance(self.quantity, DomainProbability):
            # Closed-form CLT intervals (Lemaitre et al. 2015) instead of
            # bootstrap: same output shapes, no resampling.
            pli_low, pli_high, q_low, q_high = self._analytic_intervals_domain(
                U, y, grid, logw_fn
            )
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
        r"""
        PLI under mean tilt: signed :math:`\delta` in standard space.

        Parameters
        ----------
        X : :py:class:`openturns.Sample`, size `(N, d)`
        Y : :py:class:`openturns.Sample`, size `(N, 1)`
        shifts : array_like, size `(K,)`
            Grid covering :math:`\delta < 0` and :math:`\delta > 0`.

        Returns
        -------
        PerturbedLawIndicesResult
        """
        shifts = self._as_grid(shifts, "shifts")
        U, y = self._prepare(X, Y)
        return self._run_grid(U, y, shifts, logweights_mean, kl_mean, "mean")

    def compute_var(
        self, X: ot.Sample, Y: ot.Sample, ratios: ArrayLike
    ) -> PerturbedLawIndicesResult:
        r"""
        PLI under Gaussian scale tilt: :math:`r = V/\sigma_0^2`.

        Parameters
        ----------
        X : :py:class:`openturns.Sample`, size `(N, d)`
        Y : :py:class:`openturns.Sample`, size `(N, 1)`
        ratios : array_like, size `(K,)`
            Grid with :math:`r > 0` covering :math:`r < 1` and
            :math:`r > 1`; :math:`r = 1` is nominal.

        Returns
        -------
        PerturbedLawIndicesResult
        """
        ratios = self._as_grid(ratios, "ratios", positive=True)
        U, y = self._prepare(X, Y)
        return self._run_grid(U, y, ratios, logweights_var, kl_var, "var")
