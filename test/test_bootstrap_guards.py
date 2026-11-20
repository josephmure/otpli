import numpy as np
import openturns as ot

import otpli


def test_bootstrap_disabled():
    ot.RandomGenerator.SetSeed(0)
    dist = ot.JointDistribution([ot.Normal(0.0, 1.0)] * 2)
    X = dist.getSample(500)
    Y = ot.Sample((np.asarray(X)[:, 0] ** 2).reshape(-1, 1).tolist())
    pli = otpli.PerturbedLawIndices(dist, otpli.Quantile(0.9),
                                    n_bootstrap=0)
    res = pli.compute_mean(X, Y, shifts=np.array([0.0, 0.5]))
    assert res.pli_low is None and res.quantity_low is None


def test_bootstrap_ratio_shape():
    ot.RandomGenerator.SetSeed(1)
    dist = ot.JointDistribution([ot.Normal(0.0, 1.0)] * 2)
    X = dist.getSample(500)
    Y = ot.Sample(np.asarray(X)[:, [0]].tolist())
    pli = otpli.PerturbedLawIndices(dist, otpli.Quantile(0.9),
                                    n_bootstrap=50, seed=1)
    res = pli.compute_mean(X, Y, shifts=np.array([0.0, 0.5]))
    assert res.pli_low.shape == (2, 2)
    assert np.all(res.pli_low <= res.pli_high)


def _assert_raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_guards():
    ot.RandomGenerator.SetSeed(0)
    # Q2-A: non-independent copula rejected (fail-fast at construction)
    R = ot.CorrelationMatrix(2)
    R[0, 1] = 0.8
    dep = ot.JointDistribution([ot.Normal()] * 2, ot.NormalCopula(R))
    X = dep.getSample(100)
    ot.Sample(np.asarray(X)[:, [0]].tolist())
    _assert_raises(lambda: otpli.PerturbedLawIndices(dep, otpli.Mean(), n_bootstrap=0))
    # Non-Gaussian standard law (1D Student) rejected (§5.1):
    # the standalone marginal has a Student standard (std ~ 1.29),
    # the MGF diverges, the tilt is not definable.
    stu = ot.Student(5.0, 0.0, 1.0)
    _assert_raises(lambda: otpli.PerturbedLawIndices(stu, otpli.Mean(), n_bootstrap=0))
    # r <= 0 rejected (Q1)
    ind = ot.JointDistribution([ot.Normal()] * 1)
    Xi = ind.getSample(100)
    Yi = ot.Sample(np.asarray(Xi)[:, [0]].tolist())
    plii = otpli.PerturbedLawIndices(ind, otpli.Mean(), n_bootstrap=0)
    _assert_raises(lambda: plii.compute_var(Xi, Yi, ratios=np.array([1.0, -2.0])))


def test_docstring():
    import doctest

    import otpli.indices
    import otpli.perturbation
    import otpli.quantity
    import otpli.utils
    for mod in (otpli.indices, otpli.perturbation, otpli.quantity, otpli.utils):
        n_fail, _ = doctest.testmod(mod, verbose=False)
        assert n_fail == 0
