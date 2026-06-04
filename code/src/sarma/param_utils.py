from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any


@dataclass
class EstimatorConfig:
    """Configuration container for `SARMAEstimator` constructor.

    This mirrors the key constructor arguments of `SARMAEstimator` and
    provides a `to_kwargs()` method that returns a dict suitable for
    passing directly to `SARMAEstimator(**kwargs)`.

    Only the most commonly used options are exposed here; advanced users
    can still pass a dict to the estimator directly or update the
    returned kwargs before constructing the estimator.
    """

    P: int = 150
    n_iter: int = 500
    stop_thres: float = 1e-5

    grid_mode: str = "auto"
    n_random: int = 30
    seed: Optional[int] = None
    use_multistart: bool = True
    multistart_parallel: bool = False
    multistart_n_jobs: int = 1

    method: str = "ls"  # 'ls' or 'mle'

    verbose: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a dict of kwargs compatible with `SARMAEstimator.__init__()`."""
        return asdict(self)


@dataclass
class BICConfig:
    """Configuration container for BIC-based order selection and refinement.

    This captures options that control whether BIC should run, which branch
    to use ('ls' or 'mle'), and the optional refinement step (top-k).
    """

    run_bic: bool = True
    bic_branch: str = "ls"  # 'ls'|'mle'|'both'

    # Top-k refinement settings
    refine_top_k: int = 0
    refine_n_random: int = 20
    refine_n_iter: int = 100
    refine_grid_mode: str = "auto"
    refine_stop_thres: float = 1e-3
    refine_parallel: bool = False
    refine_n_jobs: int = 1

    # Upper caps used by fit() when orders are missing
    bic_caps: Tuple[int, Optional[int], Optional[int]] = (3, 3, 2)

    def to_kwargs(self) -> Dict[str, Any]:
        """Return a dict of BIC-related settings.

        These keys are meant to be merged with estimator kwargs or passed
        separately to control order selection.
        """
        d = asdict(self)
        # Keep bic_caps separate because `fit()` expects bic_caps as a positional kw
        bic_caps = d.pop("bic_caps")
        d["bic_caps"] = bic_caps
        return d


def make_estimator_kwargs(**overrides) -> Dict[str, Any]:
    """Convenience factory that returns estimator kwargs with sensible defaults.

    Example:
        est_kwargs = make_estimator_kwargs(method='mle', use_multistart=False)
        bic_kwargs = make_bic_kwargs(run_bic=True, refine_top_k=3)

    The returned dict can be passed to `SARMAEstimator(**est_kwargs)`.
    """
    cfg = EstimatorConfig()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise TypeError(f"Unknown estimator kwarg: {k}")
    return cfg.to_kwargs()


def make_bic_kwargs(**overrides) -> Dict[str, Any]:
    """Convenience factory that returns BIC/refinement/order-selection kwargs.

    The returned dict contains keys such as `run_bic`, `bic_branch`,
    `refine_top_k`, and `bic_caps`. These are intended to control the
    model selection behavior and are kept separate from the estimator
    constructor options.
    """
    cfg = BICConfig()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise TypeError(f"Unknown BIC kwarg: {k}")
    return cfg.to_kwargs()


__all__ = ["EstimatorConfig", "BICConfig", "make_estimator_kwargs", "make_bic_kwargs "]
