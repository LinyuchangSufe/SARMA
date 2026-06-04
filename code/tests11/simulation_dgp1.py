"""
Simulation study for SARMA estimation (DGP1).

The data are generated from a VARMA(1,1) model that admits an exact SARMA representation:
    DGP1: N=6, (p,r,s)=(1,1,1), (λ₀₁, γ₀₁, φ₀₁) = (-0.8, 0.8, π/4)

Innovations follow either:
  - Multivariate normal N(0, Σ₀)
  - Student's t₅ with zero mean and covariance Σ₀

Covariance matrices:
  - Σ₀ = I_N (identity)
  - Σ₀ = 0.5(1_N 1_N' + I_N) (compound symmetry)

Sample sizes: T ∈ {500, 1000}
Replications: 1000 per configuration
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from typing import Dict, Tuple, List, Optional
import time
import pickle
from pathlib import Path
from scipy.stats import ortho_group

# Import SARMA utilities (imports are optional for data generation only)
try:
    from src.sarma.estimator import SARMAEstimator
    from src.sarma.optim import multi_start_BCD
    from src.sarma.optim import BCD_SARMA
except Exception:
    try:
        from estimator import SARMAEstimator
        from optim import multi_start_BCD
        from optim import BCD_SARMA
    except Exception:
        # Not needed for pure data generation
        SARMAEstimator = None
        multi_start_BCD = None
        BCD_SARMA = None


# ============================================================================
# DGP1: VARMA(1,1) → SARMA(1,1,1) Data Generation
# ============================================================================

def _rot2(gamma: float, phi: float) -> np.ndarray:
    """
    2×2 rotation-scaling block for a pair of complex conjugate roots.
    
    Parameters
    ----------
    gamma : float
        Modulus (decay rate) of the complex roots.
    phi : float
        Phase angle in (0, π).
    
    Returns
    -------
    R : ndarray, shape (2, 2)
        Rotation-scaling matrix.
    """
    return np.array([
        [gamma * np.cos(phi),  gamma * np.sin(phi)],
        [-gamma * np.sin(phi), gamma * np.cos(phi)]
    ])


def _phi_diag_values(N: int, phi_design: str) -> np.ndarray:
    """Return stable AR eigenvalues for the requested simulation design."""
    if phi_design == "rank2":
        vals = np.zeros(N)
        if N < 2:
            raise ValueError("rank2 Phi design requires N >= 2")
        vals[0] = 0.5
        vals[1] = -0.5
        return vals

    if phi_design == "fullrank_stable":
        if N != 6:
            raise ValueError("fullrank_stable Phi design is defined for N=6")
        return np.array([-0.60, 0.50, 0.40, -0.30, 0.20, -0.10], dtype=float)

    if phi_design == "fullrank_symmetric":
        if N != 6:
            raise ValueError("fullrank_symmetric Phi design is defined for N=6")
        return np.array([-0.60, 0.60, -0.50, 0.50, -0.40, 0.40], dtype=float)

    if phi_design == "fullrank_symmetric_strong":
        if N != 6:
            raise ValueError("fullrank_symmetric_strong Phi design is defined for N=6")
        return np.array([-0.80, 0.80, -0.70, 0.70, -0.60, 0.60], dtype=float)

    if phi_design == "diag_balanced_pm05":
        if N != 6:
            raise ValueError("diag_balanced_pm05 Phi design is defined for N=6")
        return np.array([0.50, 0.50, 0.50, -0.50, -0.50, -0.50], dtype=float)

    raise ValueError("phi_design must be one of {'rank2', 'fullrank_stable', 'fullrank_symmetric', 'fullrank_symmetric_strong', 'diag_balanced_pm05'}")


def generate_dgp1(
    T: int = 500,
    N: int = 6,
    p: int = 1,
    r: int = 1,
    s: int = 1,
    lam_0: float = -0.8,
    gam_0: float = 0.8,
    phi_0: float = np.pi / 4,
    innovation_dist: str = "normal",  # 'normal' or 't5'
    sigma_type: str = "identity",      # 'identity' or 'compound'
    phi_design: str = "rank2",
    seed: Optional[int] = None,
    burn: int = 1000,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Generate synthetic VARMA(1,1) data with exact SARMA(p,r,s) representation.
    
    Model:
        y_t = Phi * y_{t-1} + ε_t - Theta * ε_{t-1}
    where:
      - Phi is constructed from AR with decay φ_strength ≈ 0.8
      - Theta is constructed from seasonal parameters (λ, γ, φ) via similarity transform
      - ε_t ~ N(0, Σ₀) or t₅(0, Σ₀)
    
    Parameters
    ----------
    T : int
        Sample size (after burn-in).
    N : int
        Number of variables (dimension).
    p, r, s : int
        SARMA orders: AR(p), seasonal magnitudes (r), seasonal pairs (s).
    lam_0, gam_0, phi_0 : float
        True SARMA parameters:
          - λ₀: real decay parameter for seasonal AR
          - γ₀: modulus for seasonal complex pair
          - φ₀: phase angle for seasonal complex pair
    innovation_dist : {'normal', 't5'}
        Distribution of innovations.
    sigma_type : {'identity', 'compound'}
        Covariance structure: identity or compound symmetry.
    phi_design : {'rank2', 'fullrank_stable', 'fullrank_symmetric', 'fullrank_symmetric_strong'}
        AR spectrum design. The legacy rank-2 design uses eigenvalues
        (0.5, -0.5, 0, ..., 0). The full-rank stable design for N=6 uses
        (-0.60, 0.50, 0.40, -0.30, 0.20, -0.10). The full-rank symmetric
        design uses paired eigenvalues (-0.60, 0.60, -0.50, 0.50, -0.40, 0.40).
        The strong symmetric design uses (-0.80, 0.80, -0.70, 0.70, -0.60, 0.60).
    seed : int or None
        Random seed for reproducibility.
    burn : int
        Burn-in length to discard.
    
    Returns
    -------
    y : ndarray, shape (T, N)
        Generated time series.
    params : dict
        Dictionary containing:
          - 'Phi': AR coefficient matrix (N × N)
          - 'Theta': MA coefficient matrix (N × N)
          - 'Sigma': Innovation covariance (N × N)
          - 'lam_0', 'gam_0', 'phi_0': True SARMA parameters
          - 'settings': Config dict
    """
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)
    b_phi = np.eye(N) if phi_design == "diag_balanced_pm05" else ortho_group.rvs(N, random_state=seed)
    b_phi_inv = np.linalg.inv(b_phi)

    # Validate dimensions
    if r + 2 * s > N:
        raise ValueError(f"Need N >= r + 2s = {r + 2*s}, got N = {N}.")
    
    # ---------- 1. Construct Phi (AR part) ----------
    # Keep rotated-Phi style and make the spectrum explicit/reproducible.
    phi_diag_vals = _phi_diag_values(N, phi_design)
    phi_diag = np.diag(phi_diag_vals)
    Phi = b_phi @ phi_diag @ b_phi_inv
    
    # ---------- 2. Construct Theta (MA/Seasonal part) ----------
    # Build J: Jordan-like block with real and complex eigenvalues
    J = np.zeros((N, N))
    idx = 0
    
    # r real blocks (1×1) with eigenvalues λ₀
    for _ in range(r):
        J[idx, idx] = lam_0
        idx += 1
    
    # s complex blocks (2×2) with eigenvalues γ₀*exp(±i*φ₀)
    for _ in range(s):
        if idx + 1 >= N:
            raise ValueError(f"Dimension mismatch: cannot place {s} complex blocks in {N}-dim space.")
        J[idx:idx+2, idx:idx+2] = _rot2(gam_0, phi_0)
        idx += 2
    
    # Similarity transform: Theta = B_theta @ J @ B_theta^{-1}
    b_theta = ortho_group.rvs(N, random_state=None if seed is None else seed + 1)
    b_theta_inv = np.linalg.inv(b_theta)
    Theta = b_theta @ J @ b_theta_inv
    
    # ---------- 3. Construct Sigma (innovation covariance) ----------
    if sigma_type == "identity":
        Sigma = np.eye(N)
    elif sigma_type == "compound":
        # Σ = 0.5 * (1_N 1_N' + I_N)
        ones = np.ones((N, 1))
        Sigma = 0.5 * (ones @ ones.T + np.eye(N))
    else:
        raise ValueError(f"Unknown sigma_type: {sigma_type}")
    
    # ---------- 4. Generate innovations ----------
    T_total = T + burn
    
    if innovation_dist == "normal":
        # Multivariate normal
        eps = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T_total)
    elif innovation_dist == "t5":
        # Student's t₅: scale to match covariance Σ
        # For t_ν, Var = ν/(ν-2) when properly scaled
        # t₅ has Var = 5/3 ≈ 1.667
        df = 5
        scale_factor = np.sqrt((df - 2) / df)  # ≈ 0.7746
        Z = rng.standard_normal((T_total, N))
        chi2_vals = rng.chisquare(df, T_total) / df  # Var ≈ 2/df = 0.4
        eps_std = (Z.T / np.sqrt(chi2_vals)).T * scale_factor
        # Apply Cholesky decomposition of Sigma
        L = np.linalg.cholesky(Sigma)
        eps = eps_std @ L.T
    else:
        raise ValueError(f"Unknown innovation_dist: {innovation_dist}")
    
    # ---------- 5. Simulate VARMA(1,1) ----------
    # y_t = Phi * y_{t-1} + ε_t - Theta * ε_{t-1}
    y = np.zeros((T_total, N))
    for t in range(1, T_total):
        y[t] = y[t-1] @ Phi.T + eps[t] - eps[t-1] @ Theta.T
    
    # Discard burn-in
    y = y[burn:, :]
    
    # ---------- 6. Package parameters ----------
    params = {
        "Phi": Phi,
        "Theta": Theta,
        "Sigma": Sigma,
        "J": J,
        "B": b_theta,          # Backward compatibility for existing scripts
        "B_theta": b_theta,
        "B_phi": b_phi,
        "Phi_diag": phi_diag,
        "Phi_diag_vals": phi_diag_vals,
        "lam_0": lam_0,
        "gam_0": gam_0,
        "phi_0": phi_0,
        "settings": {
            "T": T,
            "N": N,
            "p": p,
            "r": r,
            "s": s,
            "innovation_dist": innovation_dist,
            "sigma_type": sigma_type,
            "phi_design": phi_design,
            "burn": burn,
            "seed": seed,
        },
    }
    
    return y, params


def simulate_with_params(
    Phi: np.ndarray,
    Theta: np.ndarray,
    Sigma: np.ndarray,
    T: int,
    innovation_dist: str = "normal",
    seed: Optional[int] = None,
    burn: int = 1000,
) -> np.ndarray:
    """
    Simulate VARMA dynamics y_t = Phi y_{t-1} + eps_t - Theta eps_{t-1}
    given fixed Phi, Theta and innovation covariance Sigma. Only innovations
    differ across replications when using this function.
    """
    if seed is not None:
        np.random.seed(seed)
    rng = np.random.default_rng(seed)

    T_total = T + burn
    N = Phi.shape[0]

    if innovation_dist == "normal":
        eps = rng.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T_total)
    elif innovation_dist == "t5":
        df = 5
        scale_factor = np.sqrt((df - 2) / df)
        Z = rng.standard_normal((T_total, N))
        chi2_vals = rng.chisquare(df, T_total) / df
        eps_std = (Z.T / np.sqrt(chi2_vals)).T * scale_factor
        L = np.linalg.cholesky(Sigma)
        eps = eps_std @ L.T
    else:
        raise ValueError(f"Unknown innovation_dist: {innovation_dist}")

    y = np.zeros((T_total, N))
    for t in range(1, T_total):
        y[t] = y[t-1] @ Phi.T + eps[t] - eps[t-1] @ Theta.T

    return y[burn:, :]


# ============================================================================
# Simulation Experiment
# ============================================================================

class DGP1Simulation:
    """
    Simulation study runner for DGP1.
    
    Generates multiple replications of VARMA(1,1) data under different
    configurations and optionally fits SARMA models to each replication.
    """
    
    def __init__(
        self,
        n_reps: int = 1000,
        sample_sizes: List[int] = None,
        innovation_dists: List[str] = None,
        sigma_types: List[str] = None,
        seed_base: int = 12345,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        n_reps : int
            Number of replications per configuration.
        sample_sizes : list of int
            Sample sizes to test. Default: [500, 1000]
        innovation_dists : list of str
            Innovation distributions. Default: ['normal', 't5']
        sigma_types : list of str
            Covariance types. Default: ['identity', 'compound']
        seed_base : int
            Base seed for reproducibility.
        verbose : bool
            Print progress messages.
        """
        self.n_reps = n_reps
        self.sample_sizes = sample_sizes or [500, 1000]
        self.innovation_dists = innovation_dists or ["normal", "t5"]
        self.sigma_types = sigma_types or ["identity", "compound"]
        self.seed_base = seed_base
        self.verbose = verbose
        
        self.results = {}  # Store results by config
        self.config_names = []
    
    def generate_replications(
        self,
        config_name: str = "default",
        T: int = 500,
        N: int = 6,
        p: int = 1,
        r: int = 1,
        s: int = 1,
        lam_0: float = -0.8,
        gam_0: float = 0.8,
        phi_0: float = np.pi / 4,
        innovation_dist: str = "normal",
        sigma_type: str = "identity",
        phi_design: str = "rank2",
        dgp_seed: Optional[int] = None,
        fixed_dgp: bool = True,
        fit_on_generate: bool = False,
        estimator_kwargs: Optional[dict] = None,
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate n_reps replications of DGP1.
        
        Parameters
        ----------
        config_name : str
            Name for this configuration.
        T, N, p, r, s : int
            DGP parameters.
        lam_0, gam_0, phi_0 : float
            True SARMA parameters.
        innovation_dist, sigma_type : str
            Distribution and covariance settings.
        
        Returns
        -------
        replications : list of (y, params) tuples
        """
        replications = []
        estimates = []  # store fit results when fit_on_generate=True

        # If fixed_dgp is requested, generate Phi/Theta/J/Q2/Sigma once here
        if fixed_dgp:
            base_seed = self.seed_base if dgp_seed is None else int(dgp_seed)
            # call generate_dgp1 once to obtain matrices (we discard returned y)
            _, base_params = generate_dgp1(
                T=1, N=N, p=p, r=r, s=s,
                lam_0=lam_0, gam_0=gam_0, phi_0=phi_0,
                innovation_dist=innovation_dist, sigma_type=sigma_type,
                phi_design=phi_design,
                seed=base_seed, burn=0,
            )
            Phi_base = base_params["Phi"]
            Theta_base = base_params["Theta"]
            Sigma_base = base_params["Sigma"]
            J_base = base_params["J"]
            B_base = base_params["B"]
            B_theta_base = base_params.get("B_theta", B_base)
            B_phi_base = base_params.get("B_phi")
            Phi_diag_base = base_params.get("Phi_diag")
            Phi_diag_vals_base = base_params.get("Phi_diag_vals")
        else:
            Phi_base = Theta_base = Sigma_base = J_base = B_base = None
            B_theta_base = B_phi_base = Phi_diag_base = Phi_diag_vals_base = None
        
        if self.verbose:
            print(f"\n[{config_name}] Generating {self.n_reps} replications...")
            print(f"  T={T}, N={N}, (p,r,s)=({p},{r},{s})")
            print(f"  innovation_dist={innovation_dist}, sigma_type={sigma_type}")
        
        start_time = time.time()
        
        for rep in range(self.n_reps):
            seed = self.seed_base + rep  # Unique seed per replication
            if fixed_dgp:
                # simulate using base matrices but new innovations
                y = simulate_with_params(
                    Phi=Phi_base,
                    Theta=Theta_base,
                    Sigma=Sigma_base,
                    T=T,
                    innovation_dist=innovation_dist,
                    seed=seed,
                    burn=1000,
                )
                params = {
                    "Phi": Phi_base,
                    "Theta": Theta_base,
                    "Sigma": Sigma_base,
                    "J": J_base,
                    "B": B_base,
                    "B_theta": B_theta_base,
                    "B_phi": B_phi_base,
                    "Phi_diag": Phi_diag_base,
                    "Phi_diag_vals": Phi_diag_vals_base,
                    "lam_0": lam_0,
                    "gam_0": gam_0,
                    "phi_0": phi_0,
                    "settings": {
                        "T": T,
                        "N": N,
                        "p": p,
                        "r": r,
                        "s": s,
                        "innovation_dist": innovation_dist,
                        "sigma_type": sigma_type,
                        "phi_design": phi_design,
                        "dgp_seed": base_seed,
                        "seed": seed,
                    },
                }
            else:
                y, params = generate_dgp1(
                    T=T,
                    N=N,
                    p=p,
                    r=r,
                    s=s,
                    lam_0=lam_0,
                    gam_0=gam_0,
                    phi_0=phi_0,
                    innovation_dist=innovation_dist,
                    sigma_type=sigma_type,
                    phi_design=phi_design,
                    seed=seed,
                    burn=1000,
                )
            # store only seed and params so the series can be reproduced on demand
            replications.append({"seed": seed, "params": params})
            # Optionally fit SARMA immediately and store estimates (simplified)
            if fit_on_generate:
                ek = estimator_kwargs or {}
                if SARMAEstimator is None:
                    if self.verbose:
                        print("Warning: SARMAEstimator not available; skipping fit and recording placeholder.")
                    estimates.append({
                        "lmbd": None,
                        "eta": None,
                        "G": None,
                        "Sigma": None,
                        "AsyVar": None,
                        "time_elapsed": 0.0,
                        "error": "SARMAEstimator_not_available",
                    })
                else:
                    t0 = time.time()
                    try:
                        est_local = SARMAEstimator(**ek)
                        est_local.fit(y, p=p, r=r, s=s)
                        elapsed = time.time() - t0
                        # get_params returns a dict with keys including lmbd, eta, G, Sigma, AsyVar
                        params_est = est_local.get_params()
                        est_dict = {
                            "lmbd": np.asarray(params_est.get("lmbd")) if params_est.get("lmbd") is not None else None,
                            "eta": np.asarray(params_est.get("eta")) if params_est.get("eta") is not None else None,
                            "G": np.asarray(params_est.get("G")) if params_est.get("G") is not None else None,
                            "Sigma": np.asarray(params_est.get("Sigma")) if params_est.get("Sigma") is not None else None,
                            "AsyVar": np.asarray(params_est.get("AsyVar")) if params_est.get("AsyVar") is not None else None,
                            "time_elapsed": elapsed,
                        }
                    except Exception as e:
                        elapsed = time.time() - t0
                        est_dict = {
                            "lmbd": None,
                            "eta": None,
                            "G": None,
                            "Sigma": None,
                            "AsyVar": None,
                            "time_elapsed": elapsed,
                            "error": str(e),
                        }
                    estimates.append(est_dict)
            
            if self.verbose and (rep + 1) % max(100, self.n_reps // 10) == 0:
                elapsed = time.time() - start_time
                print(f"  Generated {rep + 1}/{self.n_reps} replications ({elapsed:.1f}s)")
        
        elapsed = time.time() - start_time
        if self.verbose:
            print(f"  Completed in {elapsed:.1f}s")
        
        self.results[config_name] = {
            "replications": replications,
            "estimates": estimates if fit_on_generate else None,
            "config": {
                "T": T,
                "N": N,
                "p": p,
                "r": r,
                "s": s,
                "lam_0": lam_0,
                "gam_0": gam_0,
                "phi_0": phi_0,
                "innovation_dist": innovation_dist,
                "sigma_type": sigma_type,
            }
        }
        self.config_names.append(config_name)
        
        return replications
    
    def save_replications(self, path: str) -> None:
        """Save all generated replications to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\nSaving replications to {path}...")
        
        with open(path, "wb") as f:
            pickle.dump(self.results, f)
        
        if self.verbose:
            print(f"  Saved successfully.")
    
    def load_replications(self, path: str) -> None:
        """Load replications from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            self.results = pickle.load(f)
        self.config_names = list(self.results.keys())
        if self.verbose:
            print(f"Loaded replications from {path}")
            print(f"  Configurations: {self.config_names}")
    
    def summarize(self) -> pd.DataFrame:
        """
        Summarize all configurations and replications.
        
        Returns
        -------
        summary : DataFrame
            Summary statistics per configuration.
        """
        summaries = []
        
        for config_name in self.config_names:
            cfg_data = self.results[config_name]
            config = cfg_data["config"]
            reps = cfg_data["replications"]
            
            # Compute summary stats from replications
            data_stats = {
                "config": config_name,
                "T": config["T"],
                "N": config["N"],
                "(p,r,s)": f"({config['p']},{config['r']},{config['s']})",
                "innovation_dist": config["innovation_dist"],
                "sigma_type": config["sigma_type"],
                "n_reps": len(reps),
            }
            
            # Compute mean/std of data properties
            y_means = []
            y_stds = []
            for y, _ in reps:
                y_means.append(np.mean(y))
                y_stds.append(np.std(y))
            
            data_stats.update({
                "mean(y_mean)": np.mean(y_means),
                "std(y_mean)": np.std(y_means),
                "mean(y_std)": np.mean(y_stds),
                "std(y_std)": np.std(y_stds),
            })
            
            summaries.append(data_stats)
        
        return pd.DataFrame(summaries)


# ============================================================================
# Example: Run full simulation suite
# ============================================================================

if __name__ == "__main__":
    # Create simulation runner
    sim = DGP1Simulation(
        n_reps=10,  # Small for testing; use 1000 for production
        seed_base=12345,
        verbose=True,
    )
    
    print("="*70)
    print("DGP1 VARMA(1,1) → SARMA(1,1,1) Simulation")
    print("="*70)
    
    # Configuration 1: T=500, normal innovations, identity Sigma
    sim.generate_replications(
        config_name="T500_normal_identity",
        T=500, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="normal",
        sigma_type="identity",
        fit_on_generate=True,
        estimator_kwargs={"P":150, "n_iter":20, "stop_thres":1e-4, "esti_method":"ls"},
    )
    
    # Configuration 2: T=500, normal innovations, compound Sigma
    sim.generate_replications(
        config_name="T500_normal_compound",
        T=500, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="normal",
        sigma_type="compound",
        fit_on_generate=True,
        estimator_kwargs={"P":150, "n_iter":20, "stop_thres":1e-4, "esti_method":"ls"},
    )
    
    # Configuration 3: T=500, t5 innovations, identity Sigma
    sim.generate_replications(
        config_name="T500_t5_identity",
        T=500, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="t5",
        sigma_type="identity",
    )
    
    # Configuration 4: T=500, t5 innovations, compound Sigma
    sim.generate_replications(
        config_name="T500_t5_compound",
        T=500, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="t5",
        sigma_type="compound",
    )
    
    # Configuration 5: T=1000, normal innovations, identity Sigma
    sim.generate_replications(
        config_name="T1000_normal_identity",
        T=1000, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="normal",
        sigma_type="identity",
        fit_on_generate=True,
        estimator_kwargs={"P":150, "n_iter":20, "stop_thres":1e-4, "esti_method":"ls"},
    )
    
    # Configuration 6: T=1000, normal innovations, compound Sigma
    sim.generate_replications(
        config_name="T1000_normal_compound",
        T=1000, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="normal",
        sigma_type="compound",
    )
    
    # Configuration 7: T=1000, t5 innovations, identity Sigma
    sim.generate_replications(
        config_name="T1000_t5_identity",
        T=1000, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="t5",
        sigma_type="identity",
    )
    
    # Configuration 8: T=1000, t5 innovations, compound Sigma
    sim.generate_replications(
        config_name="T1000_t5_compound",
        T=1000, N=3, p=1, r=1, s=1,
        lam_0=-0.8, gam_0=0.8, phi_0=np.pi/4,
        innovation_dist="t5",
        sigma_type="compound",
    )
    
    # Print summary
    print("\n" + "="*70)
    print("Summary of Replications")
    print("="*70)
    summary_df = sim.summarize()
    print(summary_df.to_string())
    
    # Save replications
    output_path = "tests11/data_dgp1_reps.pkl"
    sim.save_replications(output_path)
    
    print("\n" + "="*70)
    print("Simulation Complete")
    print("="*70)
