"""Toy Gaussian PLI curve (Y = X0 + X1)."""
import numpy as np
import openturns as ot

import otpli

ot.RandomGenerator.SetSeed(0)
dist = ot.JointDistribution([ot.Normal(0.0, 1.0)] * 2)
X = dist.getSample(3000)
Y = ot.Sample((np.asarray(X).sum(axis=1)).reshape(-1, 1).tolist())

for mode, grid, fn in [
    ("MEAN", np.linspace(-2, 2, 9), "compute_mean"),
    ("VAR", np.array([0.25, 0.5, 1.0, 2.0, 4.0]), "compute_var"),
]:
    pli = otpli.PerturbedLawIndices(dist, otpli.Quantile(0.95),
                                    n_bootstrap=200, seed=0)
    kw = {"shifts": grid} if mode == "MEAN" else {"ratios": grid}
    res = getattr(pli, fn)(X, Y, **kw)
    print(f"--- {mode} ---")
    print("grid      :", res.grid)
    print("KL        :", np.round(res.kl, 4))
    print("PLI X0    :", np.round(res.pli[:, 0], 3))
    print("ESS/N X0  :", np.round(res.ess_ratio[:, 0], 3))
