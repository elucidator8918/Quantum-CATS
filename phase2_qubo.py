"""
Phase 2: BLP Formulation & QUBO Conversion

Formulates the grid expansion problem as a Binary Linear Program (BLP)
following Cuenca et al. [4], then converts to QUBO for quantum solvers.

BLP Formulation:
  min  sum_e  c_e * x_e  +  lambda * sum_l  s_l
  s.t. sum_e  A[l,e] * x_e  +  s_l  >= 1   for each congested line l
       s_l >= 0                               (slack for uncovered congestion)
       sum_e  c_e * x_e  <= Budget            (optional)
       x_e in {0, 1}                          (build or not)

QUBO Conversion:
  H = sum c_e x_e  +  P1 * sum_l (1 - sum_e A[l,e] x_e)^2
                    +  P2 * (sum c_e x_e - B)^2  [if budget]

References:
  [4] Cuenca et al., IEEE Trans. Power Systems 40(1), 2024
  [5] Koch et al., QOBLIB, arXiv:2504.03832, 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple
import time

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from config import PENALTY_CONGESTION, PENALTY_BUDGET, DEFAULT_BUDGET


# ============================================================
# 1. BUILD BINARY LINEAR PROGRAM
# ============================================================

def build_blp(candidates: 'pd.DataFrame',
              influence_matrix: np.ndarray,
              congested_count: int,
              budget: Optional[float] = None
              ) -> QuadraticProgram:
    """
    Build the Binary Linear Program for grid expansion.

    Variables:
      x_e in {0,1} for each candidate line e (build or not)

    Objective:
      min sum_e c_e * x_e   (minimize investment cost)

    Constraints:
      For each congested line l:
        sum_e A[l,e] * x_e >= 1   (congestion must be resolved)
      Optional:
        sum_e c_e * x_e <= budget
    """
    n_candidates = len(candidates)
    qp = QuadraticProgram("grid_expansion")

    # Decision variables
    costs = candidates['cost'].values
    for e in range(n_candidates):
        qp.binary_var(name=f"x_{e}")

    # Objective: minimize total cost
    linear_obj = {f"x_{e}": float(costs[e]) for e in range(n_candidates)}
    qp.minimize(linear=linear_obj)

    # Constraint: each congested line must be covered
    for l in range(congested_count):
        row = influence_matrix[l]
        coeffs = {f"x_{e}": float(row[e]) for e in range(n_candidates) if row[e] != 0}
        if len(coeffs) > 0:
            qp.linear_constraint(
                linear=coeffs,
                sense=">=",
                rhs=1.0,
                name=f"cover_congestion_{l}"
            )

    # Optional budget constraint
    if budget is not None:
        budget_coeffs = {f"x_{e}": float(costs[e]) for e in range(n_candidates)}
        qp.linear_constraint(
            linear=budget_coeffs,
            sense="<=",
            rhs=budget,
            name="budget"
        )

    print(f"[BLP] Built: {qp.get_num_vars()} variables, "
          f"{qp.get_num_linear_constraints()} constraints")
    print(f"[BLP] Objective coefficients range: "
          f"[{costs.min():.2f}, {costs.max():.2f}]")

    return qp


# ============================================================
# 2. CONVERT BLP -> QUBO
# ============================================================

def blp_to_qubo(qp: QuadraticProgram,
                penalty: float = PENALTY_CONGESTION
                ) -> Tuple[QuadraticProgram, 'QuadraticProgramToQubo']:
    """
    Convert the constrained BLP to an unconstrained QUBO.

    Qiskit's QuadraticProgramToQubo handles:
    1. Inequality -> equality (adds slack variables)
    2. Constraints -> penalty terms in objective
    3. The result is min x^T Q x + c^T x  with x in {0,1}^n

    The penalty parameter controls constraint enforcement strength.
    Too low: constraints violated. Too high: objective swamped.
    """
    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)

    n_orig = qp.get_num_vars()
    n_qubo = qubo.get_num_vars()
    n_slack = n_qubo - n_orig

    print(f"[QUBO] Converted: {n_orig} original vars + {n_slack} slack vars "
          f"= {n_qubo} total qubits")
    print(f"[QUBO] Constraints: {qubo.get_num_linear_constraints()} "
          f"(should be 0 for QUBO)")

    return qubo, converter


# ============================================================
# 3. EXTRACT ISING HAMILTONIAN
# ============================================================

def qubo_to_ising(qubo: QuadraticProgram) -> Tuple:
    """
    Convert QUBO to Ising Hamiltonian for quantum circuits.

    H_ising = sum_{ij} J_{ij} Z_i Z_j + sum_i h_i Z_i + offset

    Returns:
      operator: SparsePauliOp (the Ising Hamiltonian)
      offset: float (constant energy shift)
    """
    operator, offset = qubo.to_ising()

    print(f"[Ising] Hamiltonian: {operator.num_qubits} qubits, "
          f"{len(operator)} Pauli terms, offset={offset:.4f}")

    return operator, offset


# ============================================================
# 4. CLASSICAL BENCHMARK (exact solver)
# ============================================================

def solve_classical(qp: QuadraticProgram) -> Dict:
    """
    Solve the BLP exactly using Qiskit's NumPy eigensolver.
    This gives the ground truth optimal solution for benchmarking.
    """
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import NumPyMinimumEigensolver

    start = time.time()
    exact_solver = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(exact_solver)

    # Need QUBO form for eigensolver
    converter = QuadraticProgramToQubo(penalty=PENALTY_CONGESTION)
    result = optimizer.solve(qp)
    elapsed = time.time() - start

    # Extract solution
    x_vals = result.x[:qp.get_num_vars()]  # Original vars only
    obj_val = sum(
        qp.objective.linear.to_dict().get(i, 0) * x_vals[i]
        for i in range(len(x_vals))
    )

    print(f"[Classical] Solved in {elapsed:.2f}s")
    print(f"[Classical] Objective value: {result.fval:.4f}")
    print(f"[Classical] Lines to build: {[i for i, v in enumerate(x_vals) if v > 0.5]}")

    return {
        'x': x_vals,
        'fval': result.fval,
        'obj_val': obj_val,
        'time_s': elapsed,
        'status': str(result.status),
        'n_lines_built': sum(1 for v in x_vals if v > 0.5),
    }


# ============================================================
# 5. FULL PHASE 2 PIPELINE
# ============================================================

def run_phase2(phase1_result: Dict,
               budget: Optional[float] = DEFAULT_BUDGET,
               penalty: float = PENALTY_CONGESTION
               ) -> Dict:
    """
    Execute the complete Phase 2 pipeline.
    Takes Phase 1 output, returns QUBO + Ising + classical solution.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: BLP -> QUBO -> Ising Formulation")
    print("=" * 60)

    candidates = phase1_result['candidates']
    A_binary = phase1_result['influence_binary']
    n_congested = len(phase1_result['congested_lines'])

    if len(candidates) == 0:
        print("[WARN] No candidate lines. Cannot formulate optimization.")
        return {}

    # Step 1: Build BLP
    print("\n--- Step 1: Building BLP ---")
    qp = build_blp(candidates, A_binary, n_congested, budget)

    # Step 2: Convert to QUBO
    print("\n--- Step 2: Converting to QUBO ---")
    qubo, converter = blp_to_qubo(qp, penalty)

    # Step 3: Extract Ising Hamiltonian
    print("\n--- Step 3: Extracting Ising Hamiltonian ---")
    operator, offset = qubo_to_ising(qubo)

    # Step 4: Classical exact solution (benchmark)
    print("\n--- Step 4: Classical exact solution ---")
    classical_result = solve_classical(qp)

    result = {
        'qp': qp,
        'qubo': qubo,
        'converter': converter,
        'ising_operator': operator,
        'ising_offset': offset,
        'classical_result': classical_result,
        'n_qubits_qubo': qubo.get_num_vars(),
        'n_qubits_original': qp.get_num_vars(),
    }

    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE")
    print(f"  Original variables: {qp.get_num_vars()}")
    print(f"  QUBO variables (qubits): {qubo.get_num_vars()}")
    print(f"  Classical optimum: {classical_result['fval']:.4f}")
    print(f"  Lines to build: {classical_result['n_lines_built']}")
    print(f"{'='*60}")

    return result