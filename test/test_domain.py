import numpy as np
import openturns as ot

import otpli


def _toy(n=1500, seed=0):
    ot.RandomGenerator.SetSeed(seed)
    dist = ot.JointDistribution([ot.Normal()] * 2)
    X = dist.getSample(n)
    Y = np.asarray(X).sum(axis=1)
    return dist, X, Y


def _interval_domain(low, high):
    return ot.Interval(float(low), float(high))


def _outside_domain(low, high):
    return ot.DomainUnion(
        [
            ot.Interval(float("-inf"), float(low)),
            ot.Interval(float(high), float("inf")),
        ]
    )


def test_domain_matches_numpy_mask():
    _, _, Y = _toy()
    w = np.random.default_rng(0).random(len(Y))
    w = w / w.sum()
    for domain, mask in [
        (_interval_domain(1.0, float("inf")), Y >= 1.0),
        (_interval_domain(float("-inf"), 1.0), Y <= 1.0),
        (_interval_domain(-1.0, 1.0), (Y >= -1.0) & (Y <= 1.0)),
        (
            _outside_domain(-1.0, 1.0),
            (Y <= -1.0) | (Y >= 1.0),
        ),
    ]:
        qo = otpli.DomainProbability(domain)
        assert qo(Y, w) == float(np.sum(w[mask]) / np.sum(w))


def test_domain_matches_ot_contains():
    _, _, Y = _toy(n=500)
    w = np.full(len(Y), 1.0 / len(Y))
    for domain in [
        _interval_domain(0.5, float("inf")),
        _interval_domain(float("-inf"), 0.5),
        _outside_domain(-0.5, 0.5),
        ot.LevelSet(ot.SymbolicFunction("y", "y^2 - 1"), ot.Less(), 0.0),
    ]:
        qo = otpli.DomainProbability(domain)
        ref = np.asarray(
            domain.contains(Y.reshape(-1, 1)),
            dtype=bool,
        )
        assert qo(Y, w) == float(np.sum(w[ref]) / np.sum(w))


def test_wrapped_domain_behaves_like_interval():
    _, _, Y = _toy(n=500)
    w = np.full(len(Y), 1.0 / len(Y))
    plain = otpli.DomainProbability(ot.Interval(0.5, float("inf")))
    wrapped = otpli.DomainProbability(ot.Domain(ot.Interval(0.5, float("inf"))))
    assert wrapped(Y, w) == plain(Y, w)
    assert np.array_equal(wrapped.event_mask(Y), plain.event_mask(Y))


def test_domain_guards():
    try:
        otpli.DomainProbability(ot.Interval([0.0, 0.0], [1.0, 1.0]))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        otpli.DomainProbability("not-a-domain")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_domain_bootstrap_disabled():
    dist, _, _ = _toy()
    try:
        otpli.PerturbedLawIndices(
            dist,
            otpli.DomainProbability(_interval_domain(2.0, float("inf"))),
            n_bootstrap=30,
            seed=0,
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_domain_integration_analytic():
    dist, X, Y = _toy()
    Ys = ot.Sample(Y.reshape(-1, 1).tolist())
    pli = otpli.PerturbedLawIndices(
        dist,
        otpli.DomainProbability(_interval_domain(2.0, float("inf"))),
        n_bootstrap=0,
        seed=0,
    )
    res = pli.compute_mean(X, Ys, shifts=np.array([-1.0, 0.0, 1.0]))
    assert res.pli.shape == (3, 2)
    # Analytic CLT intervals (not bootstrap) are filled.
    assert res.pli_low is not None and res.pli_high is not None
    assert res.quantity_low is not None and res.quantity_high is not None
    # Perturbing X0 upward raises P(Y > 2) via the X0 channel.
    assert res.pli[2, 0] > 0.0
    # Nominal exceedance probability recovered (up to float summation order).
    assert np.allclose(res.quantity_nominal[0], np.mean(Y > 2.0))
