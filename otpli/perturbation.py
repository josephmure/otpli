r"""
Copyright (C) EDF 2026

otpli.perturbation: weights, KL and ESS of Gaussian tilts.

MEAN mode: :math:`U_i \to \mathcal{N}(\delta, 1)`,
:math:`w = \exp(\delta u)`,
:math:`\mathrm{KL} = \delta^2/2`.
VAR mode: :math:`U_i \to \mathcal{N}(0, \tau^2)` with
:math:`\tau^2 = r`,
:math:`\mathrm{KL} = \log(\tau) + 1/(2\tau^2) - 1/2`.
"""
import numpy as np
from numpy.typing import ArrayLike


def logweights_mean(u_col: ArrayLike, delta: float) -> np.ndarray:
    r"""
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
        :math:`\log w = \delta u` (constant :math:`-\delta^2/2` omitted:
        normalized in the estimator).
    """
    return np.asarray(u_col, dtype=float) * float(delta)


def logweights_var(u_col: ArrayLike, ratio: float) -> np.ndarray:
    r"""
    Log-weights of the Gaussian scale tilt.

    :math:`w \sim \exp(a u^2)` with :math:`a = (1 - 1/\tau^2)/2`,
    :math:`\tau = \sqrt{r}`.
    The multiplicative constant is omitted (as in mean mode):
    only differences matter after `log-sum-exp` normalization.

    Parameters
    ----------
    u_col : numpy.ndarray, size `(n,)`
    ratio : float
        :math:`r = V/\sigma_0^2 > 0` strict, :math:`r = 1` is nominal.

    Returns
    -------
    numpy.ndarray
        Unnormalized log-weights.
    """
    r = float(ratio)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"Variance ratio r must be > 0 strict: got {ratio!r}.")
    tau2 = r
    a = 0.5 * (1.0 - 1.0 / tau2)
    u = np.asarray(u_col, dtype=float)
    return a * u * u


def kl_mean(delta: float) -> float:
    r"""
    :math:`\mathrm{KL}(P_0 \parallel P_\delta)` divergence of the mean tilt.

    :math:`P_0 = \mathcal{N}(0, 1)`,
    :math:`P_\delta = \mathcal{N}(\delta, 1)`: symmetric,
    :math:`\delta^2/2` in both directions.

    Parameters
    ----------
    delta : float

    Returns
    -------
    float
        :math:`\delta^2/2`.
    """
    return 0.5 * float(delta) ** 2


def kl_var(ratio: float) -> float:
    r"""
    :math:`\mathrm{KL}(P_0 \parallel P_\delta)` divergence of the scale tilt.

    :math:`P_0 = \mathcal{N}(0, 1)`, :math:`P_\delta = \mathcal{N}(0, r)`:
    :math:`\log(\sqrt{r}) + 1/(2r) - 1/2`.
    Asymmetric: the reverse :math:`\mathrm{KL}(P_\delta \parallel P_0)` is
    :math:`-\log(\sqrt{r}) + r/2 - 1/2`. The reported diagnostic keeps the
    reference-first convention :math:`\mathrm{KL}(P_0 \parallel P_\delta)`.

    Parameters
    ----------
    ratio : float
        :math:`r = \tau^2 > 0`.

    Returns
    -------
    float
    """
    r = float(ratio)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"Variance ratio r must be > 0 strict: got {ratio!r}.")
    tau = float(np.sqrt(r))
    return float(np.log(tau) + 1.0 / (2.0 * r) - 0.5)


def normalized_weights(logw: ArrayLike) -> np.ndarray:
    """
    Normalize log-weights with the log-sum-exp trick (stable, vectorizable).

    The maximum is subtracted before exponentiation, which is the
    log-sum-exp stabilization done by hand (no `scipy` call).
    Raises when no mathematically valid normalized distribution exists
    (all weights zero / non-finite); use `ess_ratio_from_logweights` for a soft
    degeneracy diagnostic (returns `0.0` instead of raising).

    Parameters
    ----------
    logw : numpy.ndarray, size `(n,)`

    Returns
    -------
    numpy.ndarray
        Weights summing to 1.
    """
    logw = np.asarray(logw, dtype=float)
    if logw.size == 0:
        raise ValueError("Weight normalization failed: empty log-weights.")
    m = np.max(logw)
    w = np.exp(logw - m)
    s = np.sum(w)
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError(
            "Weight normalization failed: degenerate weights "
            f"(size={logw.size}, max_logw={float(m)!r}, sum={float(s)!r})."
        )
    return w / s
