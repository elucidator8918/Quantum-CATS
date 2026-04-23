"""
Configuration and constants for the Quantum Grid Expansion project.
E.ON Global Quantum + AI Challenge 2026
"""
import os
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
CATS_MATPOWER = PROJECT_ROOT.parent / "CATS-CaliforniaTestSystem" / "MATPOWER" / "CaliforniaTestSystem.m"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# === Grid parameters ===
CATS_FREQ_HZ = 60  # North American frequency
BASE_MVA = 100.0

# === Subgrid extraction ===
# We extract subgrids of varying size for benchmarking.
# Seed bus is chosen from a well-connected region of CATS.
SUBGRID_SIZES = [15, 30, 50, 80]  # number of buses
SEED_BUS = 18  # Starting bus - well-connected region with gens+loads

# === Congestion simulation ===
LOAD_MULTIPLIER = 1.5        # Scale loads to induce congestion
THERMAL_LIMIT_FRACTION = 0.8  # Lines above 80% loading are "near-congested"
CONGESTION_THRESHOLD = 1.0    # Lines above 100% loading are congested

# === Candidate line generation ===
MAX_CANDIDATE_DISTANCE_KM = 50.0  # Max distance for candidate new lines
CANDIDATE_LINE_COST_PER_KM = 1.0  # Normalized cost per km
CANDIDATE_CAPACITY_MW = 100.0     # Rated capacity of candidate lines (MW)

# === QUBO / Optimization ===
PENALTY_CONGESTION = 10.0    # Penalty weight for unresolved congestion
PENALTY_BUDGET = 5.0         # Penalty weight for budget violation
DEFAULT_BUDGET = None        # None = no budget constraint

# === QAOA parameters ===
QAOA_REPS = 1               # Number of QAOA layers (p)
QAOA_OPTIMIZER = "COBYLA"    # Classical optimizer
QAOA_MAXITER = 200           # Max optimizer iterations
QAOA_SHOTS = 4096            # Number of measurement shots