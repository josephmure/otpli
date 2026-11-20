"""
Copyright (C) EDF 2026

otpli.utils: guards and conversions.
"""
import warnings

import numpy as np
import openturns as ot
from numpy.typing import ArrayLike


def check_independent(distribution: ot.Distribution) -> None:
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
        try:
            copula_name = (
                distribution.getCopula().getImplementation().getClassName()
            )
        except Exception:  # noqa: BLE001 - diagnostic name only, never masks logic
            copula_name = "unknown"
        raise ValueError(
            "otpli v1 is restricted to independent inputs: "
            f"non-independent copula rejected (copula={copula_name!r})."
        )


def check_standard_normal(distribution: ot.Distribution) -> None:
    """
    Check that the standard distribution is standardized Gaussian.

    The proxy returned by `getStandardDistribution` is not reliable alone:
    the actual implementation class name is checked.

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
    except Exception:  # noqa: BLE001 - fallback introspection only
        classname = std.getClassName() if hasattr(std, "getClassName") else ""
    if classname != "Normal":
        raise ValueError(
            "otpli requires a Gaussian standard distribution "
            f"(Gaussian KL frame): standard={classname!r} rejected."
        )
    if std.getDimension() != distribution.getDimension():
        raise ValueError(
            "Inconsistent standard distribution dimension: "
            f"standard dim={std.getDimension()}, "
            f"distribution dim={distribution.getDimension()}."
        )


def to_sample(
    array: ArrayLike, descriptions: list[str] | None = None
) -> ot.Sample:
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


def physical_to_standard(
    distribution: ot.Distribution, X: ArrayLike
) -> np.ndarray:
    """
    Apply the isoprobabilistic transformation :math:`U = T(X)`.

    `T` accepts `Sample`, `ndarray` or nested lists, so no
    `Sample` conversion is needed beforehand.

    Parameters
    ----------
    distribution : :py:class:`openturns.Distribution`
    X : array-like, size `(n, d)`

    Returns
    -------
    numpy.ndarray
        Standard coordinates of size `(n, d)`.
    """
    check_independent(distribution)
    check_standard_normal(distribution)
    T = distribution.getIsoProbabilisticTransformation()
    return np.asarray(T(np.ascontiguousarray(np.asarray(X, dtype=float))))


def ess_ratio_from_logweights(logw: ArrayLike) -> float:
    r"""
    Normalized effective sample size.

    Parameters
    ----------
    logw : numpy.ndarray
        Unnormalized log-weights of size `(n,)`.

    Returns
    -------
    float
        :math:`\mathrm{ESS}/N \in (0, 1]`.
    """
    logw = np.asarray(logw, dtype=float)
    m = np.max(logw)
    w = np.exp(logw - m)
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    if s1 == 0.0 or s2 == 0.0 or not np.isfinite(s1 + s2):
        return 0.0
    return float(s1 * s1 / (len(w) * s2))


def warn_if_low_ess(
    grid: ArrayLike, ess_ratio: ArrayLike, threshold: float = 0.01
) -> None:
    r"""
    Warn when :math:`\mathrm{ESS}/N` falls below the threshold.

    No hard thresholding on the grid: only validity is flagged.
    The check uses the minimum across input dimensions: if any
    dimension is degenerate at a grid point, the point is flagged.

    Parameters
    ----------
    grid : numpy.ndarray
        Grid (:math:`\delta` or :math:`r`), size `(K,)`.
    ess_ratio : numpy.ndarray
        :math:`\mathrm{ESS}/N`, size `(K,)` or `(K, d)` per (grid, dim).
    threshold : float
        Indicative threshold (default `0.01`).
    """
    ess = np.asarray(ess_ratio, dtype=float)
    g = np.asarray(grid)
    if ess.ndim == 2:
        # (K, d): flag grid points where min_d ESS/N < threshold.
        bad = np.min(ess, axis=1) < threshold
        if np.any(bad):
            vals = g[np.atleast_1d(bad)]
            # Worst (grid, dim) for the message.
            j_bad, i_bad = np.unravel_index(
                int(np.argmin(ess)), ess.shape
            )
            warnings.warn(
                f"ESS/N < {threshold} for {int(np.sum(bad))} grid point(s) "
                f"(e.g. grid={np.asarray(vals).ravel()[:4]}, "
                f"worst at grid={g[j_bad]!r} dim={i_bad} "
                f"ESS/N={ess[j_bad, i_bad]:.4g}): "
                "PLI curve potentially noisy.",
                UserWarning,
                stacklevel=3,
            )
        return
    bad = ess < threshold
    if np.any(bad):
        vals = np.asarray(grid)[np.atleast_1d(bad)]
        warnings.warn(
            f"ESS/N < {threshold} for {int(np.sum(bad))} grid point(s) "
            f"(e.g. {np.asarray(vals).ravel()[:4]}): "
            "PLI curve potentially noisy.",
            UserWarning,
            stacklevel=3,
        )
