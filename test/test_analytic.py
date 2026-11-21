import numpy as np
import openturns as ot

import otpli


def _toy(n=4000, seed=0):
    ot.RandomGenerator.SetSeed(seed)
    dist = ot.JointDistribution([ot.Normal()] * 2)
    X = dist.getSample(n)
    Y = np.asarray(X).sum(axis=1)
    return dist, X, Y


def _run(absolute=False, mode="mean", conf=0.95):
    dist, X, Y = _toy()
    Ys = ot.Sample(Y.reshape(-1, 1).tolist())
    qoi = otpli.DomainProbability(ot.Interval(2.0, float("inf")))
    pli = otpli.PerturbedLawIndices(
        dist, qoi, n_bootstrap=0, seed=0,
        confidence=conf, absolute_variation=absolute,
    )
    if mode == "mean":
        return pli.compute_mean(X, Ys, shifts=np.array([-1.0, 0.0, 1.0]))
    return pli.compute_var(X, Ys, ratios=np.array([0.5, 1.0, 2.0]))


def test_analytic_shapes_and_ordering():
    for absolute in (False, True):
        for mode in ("mean", "var"):
            res = _run(absolute=absolute, mode=mode)
            assert res.pli_low.shape == res.pli.shape
            assert res.pli_high.shape == res.pli.shape
            assert res.quantity_low.shape == res.quantity.shape
            assert res.quantity_high.shape == res.quantity.shape
            assert np.all(res.pli_low <= res.pli + 1e-12)
            assert np.all(res.pli + 1e-12 >= res.pli_low)
            assert np.all(res.pli_high + 1e-12 >= res.pli)
            assert np.all(res.quantity_low <= res.quantity + 1e-12)
            assert np.all(res.quantity_high + 1e-12 >= res.quantity)


def test_analytic_nominal_point():
    # At delta=0 / r=1 the PLI is 0 with symmetric bounds, and the
    # quantity interval is the plain binomial one.
    res = _run()
    j0 = 1  # shifts == 0.0
    assert np.allclose(res.pli_low[j0], -res.pli_high[j0])
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.975)
    p = float(np.mean(np.asarray(_toy()[2]) > 2.0))
    n = 4000
    half = z * np.sqrt(p * (1 - p) / n)
    assert np.allclose(res.quantity_low[j0], p - half, rtol=1e-9)
    assert np.allclose(res.quantity_high[j0], p + half, rtol=1e-9)


def test_analytic_confidence_scaling():
    r90 = _run(conf=0.90)
    r99 = _run(conf=0.99)
    from statistics import NormalDist

    ratio = NormalDist().inv_cdf(0.995) / NormalDist().inv_cdf(0.95)
    half90 = r90.pli_high - r90.pli_low
    half99 = r99.pli_high - r99.pli_low
    nz = half90 > 1e-15
    assert np.allclose(half99[nz] / half90[nz], ratio, rtol=1e-9)


def test_analytic_nominal_point_collapses():
    # At delta=0 / r=1 there is no tilt: the PLI is identically 0, so its
    # analytic interval collapses to the point, while the quantity
    # interval is the plain binomial one.
    res = _run()
    j0 = 1  # shifts == 0.0
    assert np.allclose(res.pli[j0], 0.0)
    assert np.all(res.pli_high[j0] - res.pli_low[j0] < 1e-9)
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.975)
    p = float(np.mean(np.asarray(_toy()[2]) > 2.0))
    n = 4000
    half = z * np.sqrt(p * (1 - p) / n)
    assert np.allclose(res.quantity_low[j0], p - half, rtol=1e-6)
    assert np.allclose(res.quantity_high[j0], p + half, rtol=1e-6)
