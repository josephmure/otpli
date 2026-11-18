import numpy as np
import openturns as ot

import otpli


def _toy(n=4000, seed=0):
    ot.RandomGenerator.SetSeed(seed)
    dist = ot.JointDistribution([ot.Normal(0.0, 1.0)] * 2)
    X = dist.getSample(n)
    # Y = X0: analytic case Q8-B(3); tilted law N(delta, 1)
    Y = ot.Sample(np.asarray(X)[:, [0]].tolist())
    return dist, X, Y


def test_mean_quantile_analytic():
    # Q8: Y = U0, target quantile delta + z_alpha, nominal z_alpha
    alpha = 0.95
    dist, X, Y = _toy()
    pli = otpli.PerturbedLawIndices(dist, otpli.Quantile(alpha),
                                    n_bootstrap=0, seed=0)
    shifts = np.array([-1.0, 0.0, 1.0])
    res = pli.compute_mean(X, Y, shifts)
    assert res.mode == "mean"
    assert res.pli.shape == (3, 2)
    assert res.ess_ratio.shape == (3, 2)
    assert np.allclose(res.kl, shifts**2 / 2.0)
    # PLI(0) = 0 exactly by construction (Q5-A, single estimator)
    assert abs(res.pli[1, 0]) == 0.0
    # Only variable 0 acts on Y = X0
    assert abs(res.pli[2, 0]) > abs(res.pli[2, 1])
    # Analytic reference: (delta + z)/z - 1, loose Monte Carlo tolerance
    z = ot.Normal().computeQuantile(alpha)[0]
    assert abs(res.pli[2, 0] - ((1.0 + z) / z - 1.0)) < 0.15


def test_var_ratio_nominal():
    dist, X, Y = _toy(n=2000)
    pli = otpli.PerturbedLawIndices(dist, otpli.Mean(), n_bootstrap=0)
    res = pli.compute_var(X, Y, ratios=np.array([0.5, 1.0, 2.0]))
    assert res.mode == "var"
    assert res.pli[1, 0] == 0.0  # r = 1 nominal
    assert np.allclose(res.kl[1], 0.0)
    assert res.kl[0] > 0.0 and res.kl[2] > 0.0
