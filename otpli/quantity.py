r"""
Copyright (C) EDF 2026

otpli.quantity: weighted quantity-of-interest contract.

The nominal value is computed by :math:`Q(y, 1)`:
:math:`\mathrm{PLI}(0) = 0` by construction,
continuous curve, single estimator.
"""
import warnings

import numpy as np
import openturns as ot
from numpy.typing import ArrayLike


class QuantityOfInterest:
    """
    Weighted contract: :math:`Q(y, w)` on the tilted empirical measure.

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

    def __call__(self, y: np.ndarray, w: np.ndarray) -> float:
        """Evaluate the quantity on the tilted empirical measure."""
        raise NotImplementedError


def _as_1d(y: ArrayLike, w: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate and flatten a `(y, w)` pair for weighted estimators.

    Parameters
    ----------
    y : array-like
        Model outputs, must be finite.
    w : array-like
        Nonnegative finite weights with strictly positive sum.

    Returns
    -------
    y : numpy.ndarray, size `(n,)`
    w : numpy.ndarray, size `(n,)`
    """
    y = np.ravel(np.asarray(y, dtype=float))
    w = np.ravel(np.asarray(w, dtype=float))
    if y.shape != w.shape:
        raise ValueError(
            f"y and w must have the same size: got {y.shape} and {w.shape}."
        )
    if not np.all(np.isfinite(y)):
        n_bad = int(np.sum(~np.isfinite(y)))
        raise ValueError(
            "Non-finite model outputs: "
            f"n_nonfinite={n_bad} (y must be finite)."
        )
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        n_neg = int(np.sum(w < 0.0))
        n_bad = int(np.sum(~np.isfinite(w)))
        raise ValueError(
            "Negative or non-finite weights: "
            f"n_negative={n_neg}, n_nonfinite={n_bad}, min={float(np.min(w))!r}."
        )
    s = np.sum(w)
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError(f"Null or non-finite weight sum: got sum={float(s)!r}.")
    return y, w


class Quantile(QuantityOfInterest):
    r"""
    Weighted stepwise `alpha`-quantile.

    :math:`\inf\{t : F_w(t) \geq \alpha\}` with :math:`F_w` the weighted
    empirical CDF.
    Equivalent to `numpy` `method="inverted_cdf"`.

    Parameters
    ----------
    alpha : float
        Order in `(0, 1)`.
    """

    def __init__(self, alpha: float) -> None:
        """Store the quantile order after validating it lies in `(0, 1)`."""
        try:
            a = float(alpha)
        except (TypeError, ValueError):
            raise ValueError(f"alpha must be in (0, 1): got {alpha!r}.") from None
        if not 0.0 < a < 1.0:
            raise ValueError(f"alpha must be in (0, 1): got {alpha!r}.")
        self.alpha = a

    def __call__(self, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted stepwise quantile at level `alpha`."""
        y, w = _as_1d(y, w)
        order = np.argsort(y, kind="mergesort")
        ys, ws = y[order], w[order]
        cdf = np.cumsum(ws) / np.sum(ws)
        k = int(np.searchsorted(cdf, self.alpha, side="left"))
        k = min(max(k, 0), len(ys) - 1)
        return float(ys[k])


class SuperQuantile(QuantityOfInterest):
    r"""
    Coherent weighted :math:`\mathbb{E}[Y \mid Y \geq q_\alpha]` superquantile.

    The threshold `q` and the mean are computed with the *same* weights.

    A low tail mass warning is emitted when the effective tail count
    :math:`\mathrm{mass} \times N < 10` or the raw tail count :math:`< 10`:
    the conditional mean
    is then high-variance. This is complementary to the global
    :math:`\mathrm{ESS}/N`
    diagnostic (a good global ESS can hide an empty tail).

    Parameters
    ----------
    alpha : float
        Order in `(0, 1)`.
    """

    def __init__(self, alpha: float) -> None:
        """Store the order and validate it lies in `(0, 1)`."""
        self._q = Quantile(alpha)
        self.alpha = self._q.alpha

    def __call__(self, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted superquantile with tail-sparsity warning."""
        y, w = _as_1d(y, w)
        q = self._q(y, w)
        mask = y >= q
        if not np.any(mask):
            return float(np.max(y))
        w_tail = w[mask]
        n_tail = int(np.sum(mask))
        mass = float(np.sum(w_tail) / np.sum(w))
        if n_tail < 10 or mass * len(y) < 10.0:
            warnings.warn(
                f"SuperQuantile tail is sparse: n_tail={n_tail}, "
                f"mass={mass:.4g} (N={len(y)}). "
                "Conditional mean is high-variance.",
                UserWarning,
                stacklevel=4,
            )
        return float(np.sum(y[mask] * w_tail) / np.sum(w_tail))


class Mean(QuantityOfInterest):
    """Weighted mean (generic-QoI example, meeting notes)."""

    def __call__(self, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted arithmetic mean."""
        y, w = _as_1d(y, w)
        return float(np.sum(y * w) / np.sum(w))


class DomainProbability(QuantityOfInterest):
    r"""
    Weighted probability of belonging to a 1D OpenTURNS domain.

    :math:`P = \mathbb{E}[\mathbb{1}_D(Y)]`, estimated by
    :math:`\sum w_i \mathbb{1}_D(y_i) / \sum w_i`.

    The domain defines the event: :math:`D = [s, +\infty[` (exceedance),
    :math:`D = {]}-\infty, s]` (below threshold),
    :math:`D = [a, b]` (inside an interval), or any 1D OpenTURNS domain
    (`DomainUnion`, `DomainComplement`, `LevelSet`, ...).
    Bootstrap is disabled for this quantity: confidence intervals always
    come from the closed-form CLT (Lemaitre et al. 2015), so
    `n_bootstrap` must be 0.
    """

    def __init__(self, domain: ot.Domain) -> None:
        """Store the 1D event domain after checking its dimension."""
        if not (hasattr(domain, "contains") and hasattr(domain, "getDimension")):
            raise ValueError(
                "domain must be an OpenTURNS domain with `contains` "
                f"and `getDimension`: got {type(domain).__name__}."
            )
        try:
            dim = int(domain.getDimension())
        except (TypeError, ValueError):
            raise ValueError(
                "domain must have an integer dimension: "
                f"got {domain!r}."
            ) from None
        if dim != 1:
            raise ValueError(
                f"domain must be 1D (output space): got dim={dim}."
            )
        self.domain = domain

    def event_mask(self, y: ArrayLike) -> np.ndarray:
        r"""Boolean membership of outputs in the domain.

        Parameters
        ----------
        y : array-like, size `(n,)`
            Model outputs, must be finite.

        Returns
        -------
        numpy.ndarray, size `(n,)`, dtype bool
            :math:`\mathbb{1}_D(y_i)`.
        """
        y1 = np.ravel(np.asarray(y, dtype=float))
        if not np.all(np.isfinite(y1)):
            raise ValueError(
                "Non-finite model outputs: "
                f"n_nonfinite={int(np.sum(~np.isfinite(y1)))}."
            )
        ind = self.domain.contains(y1.reshape(-1, 1))
        return np.asarray(ind, dtype=bool)

    def __call__(self, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted fraction of outputs inside the domain."""
        y, w = _as_1d(y, w)
        return float(np.sum(w[self.event_mask(y)]) / np.sum(w))


class Variance(QuantityOfInterest):
    """Weighted variance (generic-QoI example, meeting notes)."""

    def __call__(self, y: np.ndarray, w: np.ndarray) -> float:
        """Weighted population variance around the weighted mean."""
        y, w = _as_1d(y, w)
        m = np.sum(y * w) / np.sum(w)
        return float(np.sum(w * (y - m) ** 2) / np.sum(w))
