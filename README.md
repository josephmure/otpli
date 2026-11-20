# otpli

Perturbed-Law based Indices (PLI) based on isoprobabilistic transformation
to the standard space.

This means that the reference distribution, which if multivariate must have an independent copula,
is mapped to the standard (multivariate) normal distribution.
Perturbations of the mean or the variance are then applied to this normal distribution.

## Installation

```bash
pixi install
pixi run test
pixi run plot
# or dedicated shell :
pixi shell
```

`pip install -e .` remains possible (outside pixi).

## Minimal example

```python
import openturns as ot
import numpy as np
import otpli

dist = ot.JointDistribution([ot.Normal(0.0, 1.0)] * 2)
X = dist.getSample(2000)
Y = ot.Sample(np.asarray(X)[:, [0]].tolist())  # Y = X0, analytic case

pli = otpli.PerturbedLawIndices(dist, otpli.Quantile(alpha=0.95))
res = pli.compute_mean(X, Y, shifts=np.array([-1.0, 0.0, 1.0]))
print(res.pli[:, 0], res.ess_ratio[:, 0], res.kl)
```
