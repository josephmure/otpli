"""otpli module: explicit exports + version."""
from .indices import PerturbedLawIndices, PerturbedLawIndicesResult
from .perturbation import kl_mean, kl_var, logweights_mean, logweights_var
from .quantity import (
    DomainProbability,
    Mean,
    Quantile,
    QuantityOfInterest,
    SuperQuantile,
    Variance,
)
from .utils import ess_ratio_from_logweights

__all__ = [
    "DomainProbability",
    "Mean",
    "PerturbedLawIndices",
    "PerturbedLawIndicesResult",
    "Quantile",
    "QuantityOfInterest",
    "SuperQuantile",
    "Variance",
    "ess_ratio_from_logweights",
    "kl_mean",
    "kl_var",
    "logweights_mean",
    "logweights_var",
]
__version__ = "0.1"
