"""Toy Gaussian PLI curves (Y = X0 + X1), with plot."""
import matplotlib.pyplot as plt
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
    pli = otpli.PerturbedLawIndices(
        dist, otpli.Quantile(0.95), n_bootstrap=200, seed=0
    )
    kw = {"shifts": grid} if mode == "MEAN" else {"ratios": grid}
    res = getattr(pli, fn)(X, Y, **kw)
    print(f"--- {mode} ---")
    print("grid      :", res.grid)
    print("KL        :", np.round(res.kl, 4))
    print("PLI X0    :", np.round(res.pli[:, 0], 3))
    print("ESS/N X0  :", np.round(res.ess_ratio[:, 0], 3))
    if res.pli_low is not None:
        print("PLI_low X0:", np.round(res.pli_low[:, 0], 3))

    fig, ax1 = plt.subplots()
    ax1.errorbar(
        res.grid,
        res.pli[:, 0],
        yerr=[
            res.pli[:, 0] - res.pli_low[:, 0],
            res.pli_high[:, 0] - res.pli[:, 0],
        ]
        if res.pli_low is not None
        else None,
        marker="o",
        label="PLI X0",
    )
    ax1.set_xlabel("shifts" if mode == "MEAN" else "ratios")
    ax1.set_ylabel("PLI")
    ax2 = ax1.twinx()
    ax2.plot(res.grid, res.ess_ratio[:, 0], "s--", label="ESS/N X0")
    ax2.set_ylabel("ESS/N")
    fig.suptitle(f"PLI {mode} (Y=X0+X1, Quantile 0.95)")
    fig.tight_layout()
    out = f"pli_{mode.lower()}.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close(fig)
