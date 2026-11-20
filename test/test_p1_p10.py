import warnings

import numpy as np
import openturns as ot

import otpli
from otpli.perturbation import logweights_var


def _toy(n=800, seed=0):
    ot.RandomGenerator.SetSeed(seed)
    dist = ot.JointDistribution([ot.Normal()] * 2)
    X = dist.getSample(n)
    Y = ot.Sample(np.asarray(X)[:, [0]].tolist())
    return dist, X, Y


def test_q0_null_raises_and_absolute():
    dist, _, _ = _toy()
    X0 = dist.getSample(200)
    Y0 = ot.Sample([[0.0]] * 200)
    # Relative mode with null Q0: natural ZeroDivisionError, user responsibility.
    # Use absolute_variation=True for null references.
    try:
        otpli.PerturbedLawIndices(dist, otpli.Mean(), n_bootstrap=0).compute_mean(
            X0, Y0, [0.0, 0.5]
        )
        raise AssertionError("expected ZeroDivisionError")
    except ZeroDivisionError:
        pass
    r2 = otpli.PerturbedLawIndices(
        dist, otpli.Mean(), n_bootstrap=0, absolute_variation=True
    ).compute_mean(X0, Y0, [0.0, 0.5])
    assert np.all(np.isfinite(r2.pli))


def test_y_finite_and_shapes():
    dist, X, Y = _toy()
    pli = otpli.PerturbedLawIndices(dist, otpli.Mean(), n_bootstrap=0)
    # Y 1D accepted
    r = pli.compute_mean(X, np.asarray(X)[:, 0], [0.0])
    assert r.pli.shape == (1, 2)
    # Y non-finite raises
    Yn = np.asarray(X)[:, [0]].copy()
    Yn[0, 0] = np.nan
    try:
        pli.compute_mean(X, ot.Sample(Yn.tolist()), [0.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # X dim mismatch
    try:
        pli.compute_mean(ot.Sample(10, 3), ot.Sample(10, 1), [0.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # empty / NaN grids
    for bad in ([], [0.0, float("nan")], [0.0, float("inf")]):
        try:
            pli.compute_mean(X, Y, bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    try:
        pli.compute_var(X, Y, [1.0, -2.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_ess_min_warn_and_logweights_const():
    dist, X, Y = _toy()
    u = np.array([0.5, -1.0, 1.5])
    assert np.allclose(logweights_var(u, 2.0), 0.25 * u * u)
    pli = otpli.PerturbedLawIndices(
        dist, otpli.Quantile(0.9), n_bootstrap=0, ess_threshold=0.99
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pli.compute_mean(X, Y, [0.0, 2.0])
        assert any("ESS/N" in str(x.message) for x in w)


def test_superquantile_tail_warn():
    y = np.arange(20, dtype=float)
    w = np.ones(20) / 20.0
    sq = otpli.SuperQuantile(0.95)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        v = sq(y, w)
        assert np.isfinite(v)
        assert any("tail" in str(x.message).lower() for x in wlist)


def test_njobs_deterministic():
    dist, X, Y = _toy(n=600)
    g = np.array([-0.5, 0.0, 0.5])
    kw = {"n_bootstrap": 30, "seed": 0}
    r1 = otpli.PerturbedLawIndices(
        dist, otpli.Quantile(0.9), n_jobs=1, **kw
    ).compute_mean(X, Y, g)
    r2 = otpli.PerturbedLawIndices(
        dist, otpli.Quantile(0.9), n_jobs=2, **kw
    ).compute_mean(X, Y, g)
    assert np.allclose(r1.pli, r2.pli)
    assert np.allclose(r1.pli_low, r2.pli_low)
