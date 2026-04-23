# Quantum-CATS: Quantum-Enabled Grid Expansion Planning

**E.ON Global Quantum + AI Challenge 2026**

Quantum optimization solution for distribution network congestion reduction using the CATS (California Test System) dataset.

## Architecture

```
root/
├── src/
│   ├── config.py              # All hyperparameters and paths
│   ├── phase1_grid_data.py    # Grid loading, subgrid extraction, congestion detection
│   ├── phase2_qubo.py         # BLP formulation, QUBO conversion, classical solver
│   ├── phase3_quantum.py      # QAOA and VQE solvers, benchmarking
│   └── main.py                # Orchestrator pipeline
├── data/
│   └── cats_full.json         # Cached CATS network (fast reload)
├── results/
│   └── cats_benchmark.json    # Benchmark results
└── README.md
```

## Problem Formulation

We formulate grid expansion as a **Binary Linear Program** (BLP) following Cuenca et al. [4]:

**Decision variables**: x_e ∈ {0,1} for each candidate line e (build or not)

**Objective**: min Σ c_e · x_e (minimize total investment cost)

**Constraints**: Σ A[l,e] · x_e ≥ 1 for each congested line l (resolve all congestion)

The BLP is converted to a **QUBO** (Quadratic Unconstrained Binary Optimization) by:
1. Adding slack variables for inequality → equality conversion
2. Encoding constraints as quadratic penalty terms in the objective

The QUBO maps directly to an **Ising Hamiltonian** for quantum execution.

## Pipeline

| Phase | Module | Description |
|-------|--------|-------------|
| **1** | `phase1_grid_data.py` | Load CATS, extract subgrid via BFS, run DC power flow, detect congestion, generate candidate lines, compute PTDF and influence scores |
| **2** | `phase2_qubo.py` | Build BLP from influence matrix, convert to QUBO via Qiskit, extract Ising Hamiltonian, solve classically (benchmark) |
| **3** | `phase3_quantum.py` | Solve with QAOA (p=1,2,...) and VQE, compare approximation ratios |

## Results (CATS 20-bus subgrid)

| Method | Objective Value | Approx Ratio | Time (s) | Lines Built |
|--------|----------------|--------------|----------|-------------|
| Classical (exact) | 2.512 | 1.000 | 0.01 | 1 |
| QAOA p=1 | 2.512 | 1.000 | 6.3 | 1 |
| QAOA p=2 | 2.512 | 1.000 | 15.9 | 1 |

QAOA finds the exact optimum on the 6-qubit instance. The 30-bus instance produces a 24-qubit QUBO (8 congested lines, 6 candidates, 18 slack variables) — suitable for NISQ hardware benchmarking.

## Scaling Study

| Subgrid | Candidates | Congested | QUBO Qubits | Classical Hard? |
|---------|-----------|-----------|-------------|-----------------|
| 20 bus | 4 | 1 | 6 | No (0.01s) |
| 25 bus | 6 | 2 | ~12 | No |
| 30 bus | 6 | 8 | 24 | Interesting regime |
| 30 bus | 10 | 8 | ~30+ | Yes — exponential |

## Dependencies

```bash
pip install pandapower numpy scipy networkx matplotlib
pip install qiskit qiskit-aer qiskit-algorithms qiskit-optimization
pip install matpowercaseframes
```

## Quick Start

```python
# Load cached CATS and run full pipeline
import pandapower as pp
from phase1_grid_data import extract_subgrid, run_dc_powerflow, detect_congestion
from phase1_grid_data import generate_candidates, compute_ptdf_matrix, compute_influence_scores
from phase2_qubo import build_blp, blp_to_qubo, solve_classical

net = pp.from_json('data/cats_full.json')
sub, bmap = extract_subgrid(net, seed_bus=18, n_buses=20)
sub = run_dc_powerflow(sub, load_multiplier=2.0)
cong = detect_congestion(sub, threshold_pct=100.0)
near = detect_congestion(sub, threshold_pct=80.0)
candidates = generate_candidates(sub, near, max_candidates=4)
ptdf, bus_idx = compute_ptdf_matrix(sub)
A_bin, _ = compute_influence_scores(sub, cong, candidates, ptdf, bus_idx)

qp = build_blp(candidates, A_bin, len(cong))
classical = solve_classical(qp)
```

## References

1. Gust et al., "Designing electricity distribution networks", EJOR 315(1), 2024
2. Duan & Yu, "Power distribution system optimization by capacitated Steiner tree", IJEPES, 2003
3. Khodabakhsh et al., "A submodular approach for electricity distribution network reconfiguration", 2017
4. Cuenca et al., "Event-informed identification and allocation of distribution network planning candidates", IEEE TPWRS, 2024
5. Koch et al., "Quantum Optimization Benchmark Library — The Intractable Decathlon", 2025
6. Kotil et al., "Quantum approximate multi-objective optimization", Nature Comp Sci, 2025