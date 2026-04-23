"""
Phase 3: Quantum Algorithms — QAOA and VQE

Implements the quantum optimization pipeline:
  1. QAOA with configurable depth (p layers)
  2. VQE with hardware-efficient ansatz
  3. Simulator execution (Aer statevector / qasm)
  4. Benchmarking against classical solution

Compatible with:
  - Qiskit Aer (local simulation)
  - Qiskit Runtime (IBM Quantum hardware, future)

References:
  [5] Koch et al., QOBLIB, arXiv:2504.03832, 2025
  [6] Kotil et al., Nature Comp. Sci., 2025
"""

import numpy as np
import time
from typing import Dict, Optional, List

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.circuit.library import QAOAAnsatz, EfficientSU2

from config import (
    QAOA_REPS, QAOA_OPTIMIZER, QAOA_MAXITER, QAOA_SHOTS,
    PENALTY_CONGESTION
)


# ============================================================
# 1. QAOA SOLVER
# ============================================================

def solve_qaoa(qp: QuadraticProgram,
               reps: int = QAOA_REPS,
               optimizer_name: str = QAOA_OPTIMIZER,
               maxiter: int = QAOA_MAXITER,
               penalty: float = PENALTY_CONGESTION
               ) -> Dict:
    """
    Solve the grid expansion BLP using QAOA.

    QAOA circuit: |ψ(γ,β)⟩ = ∏_{p} e^{-iβ_p H_M} e^{-iγ_p H_C} |+⟩^n

    where H_C is the cost Hamiltonian (from QUBO/Ising) and
    H_M = Σ_i X_i is the transverse-field mixer.

    Args:
        qp: QuadraticProgram (the BLP, before QUBO conversion)
        reps: Number of QAOA layers (p). Higher = better quality, deeper circuit.
        optimizer_name: Classical optimizer for variational parameters.
        maxiter: Maximum optimization iterations.
        penalty: Constraint penalty for QUBO conversion.
    """
    print(f"[QAOA] Starting: p={reps}, optimizer={optimizer_name}, "
          f"maxiter={maxiter}")

    # Select classical optimizer
    optimizers = {
        'COBYLA': COBYLA(maxiter=maxiter),
        'SPSA': SPSA(maxiter=maxiter),
        'NELDER_MEAD': NELDER_MEAD(maxiter=maxiter),
    }
    optimizer = optimizers.get(optimizer_name, COBYLA(maxiter=maxiter))

    # Track convergence
    convergence_log = []
    def callback(eval_count, parameters, mean, std):
        convergence_log.append({
            'eval': eval_count,
            'mean': mean,
            'std': std,
        })
        if eval_count % 20 == 0:
            print(f"  [QAOA] Iter {eval_count}: E = {mean:.4f} ± {std:.4f}")

    # Build QAOA instance
    sampler = StatevectorSampler()
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        callback=callback,
    )

    # Solve via MinimumEigenOptimizer (handles BLP -> QUBO -> Ising internally)
    start = time.time()
    min_eigen_optimizer = MinimumEigenOptimizer(
        qaoa,
        penalty=penalty,
    )
    result = min_eigen_optimizer.solve(qp)
    elapsed = time.time() - start

    # Extract solution for original variables
    x_vals = result.x[:qp.get_num_vars()]

    print(f"[QAOA] Solved in {elapsed:.2f}s")
    print(f"[QAOA] Objective: {result.fval:.4f}")
    print(f"[QAOA] Lines to build: {[i for i, v in enumerate(x_vals) if v > 0.5]}")
    print(f"[QAOA] Status: {result.status}")

    return {
        'x': x_vals,
        'fval': result.fval,
        'time_s': elapsed,
        'status': str(result.status),
        'n_lines_built': sum(1 for v in x_vals if v > 0.5),
        'convergence': convergence_log,
        'method': f'QAOA_p{reps}_{optimizer_name}',
        'reps': reps,
    }


# ============================================================
# 2. SAMPLING VQE SOLVER
# ============================================================

def solve_vqe(qp: QuadraticProgram,
              reps: int = 2,
              optimizer_name: str = "COBYLA",
              maxiter: int = QAOA_MAXITER,
              penalty: float = PENALTY_CONGESTION
              ) -> Dict:
    """
    Solve using Sampling VQE with EfficientSU2 ansatz.
    This is an alternative to QAOA that uses a hardware-efficient
    parameterized circuit instead of the problem-specific QAOA ansatz.
    """
    print(f"[VQE] Starting: reps={reps}, optimizer={optimizer_name}")

    optimizers = {
        'COBYLA': COBYLA(maxiter=maxiter),
        'SPSA': SPSA(maxiter=maxiter),
    }
    optimizer = optimizers.get(optimizer_name, COBYLA(maxiter=maxiter))

    convergence_log = []
    def callback(eval_count, parameters, mean, std):
        convergence_log.append({'eval': eval_count, 'mean': mean, 'std': std})

    # Need number of qubits from the QUBO
    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)
    n_qubits = qubo.get_num_vars()

    ansatz = EfficientSU2(n_qubits, reps=reps, entanglement='linear')
    sampler = StatevectorSampler()

    vqe = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    start = time.time()
    min_eigen_optimizer = MinimumEigenOptimizer(vqe, penalty=penalty)
    result = min_eigen_optimizer.solve(qp)
    elapsed = time.time() - start

    x_vals = result.x[:qp.get_num_vars()]

    print(f"[VQE] Solved in {elapsed:.2f}s")
    print(f"[VQE] Objective: {result.fval:.4f}")
    print(f"[VQE] Lines to build: {[i for i, v in enumerate(x_vals) if v > 0.5]}")

    return {
        'x': x_vals,
        'fval': result.fval,
        'time_s': elapsed,
        'status': str(result.status),
        'n_lines_built': sum(1 for v in x_vals if v > 0.5),
        'convergence': convergence_log,
        'method': f'VQE_reps{reps}_{optimizer_name}',
        'reps': reps,
    }


# ============================================================
# 3. BENCHMARK SUITE
# ============================================================

def run_benchmark(qp: QuadraticProgram,
                  classical_result: Dict,
                  qaoa_depths: List[int] = [1, 2, 3],
                  penalty: float = PENALTY_CONGESTION
                  ) -> Dict:
    """
    Run systematic benchmarking: classical vs QAOA at various depths vs VQE.
    Computes approximation ratios and timing comparisons.
    """
    print("\n" + "=" * 60)
    print("BENCHMARKING: Classical vs Quantum")
    print("=" * 60)

    results = {'classical': classical_result}
    optimal_val = classical_result['fval']

    # QAOA at different depths
    for p in qaoa_depths:
        print(f"\n--- QAOA p={p} ---")
        try:
            qaoa_result = solve_qaoa(qp, reps=p, penalty=penalty)
            if optimal_val != 0:
                qaoa_result['approx_ratio'] = optimal_val / qaoa_result['fval']
            else:
                qaoa_result['approx_ratio'] = 1.0 if qaoa_result['fval'] == 0 else 0.0
            results[f'qaoa_p{p}'] = qaoa_result
            print(f"  Approx ratio: {qaoa_result['approx_ratio']:.4f}")
        except Exception as e:
            print(f"  [ERROR] QAOA p={p} failed: {e}")
            results[f'qaoa_p{p}'] = {'error': str(e)}

    # VQE
    print(f"\n--- VQE ---")
    try:
        vqe_result = solve_vqe(qp, penalty=penalty)
        if optimal_val != 0:
            vqe_result['approx_ratio'] = optimal_val / vqe_result['fval']
        else:
            vqe_result['approx_ratio'] = 1.0 if vqe_result['fval'] == 0 else 0.0
        results['vqe'] = vqe_result
        print(f"  Approx ratio: {vqe_result['approx_ratio']:.4f}")
    except Exception as e:
        print(f"  [ERROR] VQE failed: {e}")
        results['vqe'] = {'error': str(e)}

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Method':<20} {'Obj Value':<12} {'Approx Ratio':<14} "
          f"{'Time (s)':<10} {'Lines Built':<12}")
    print(f"{'-'*60}")
    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<20} {'FAILED':<12}")
        else:
            ar = res.get('approx_ratio', 1.0)
            print(f"{name:<20} {res['fval']:<12.4f} {ar:<14.4f} "
                  f"{res['time_s']:<10.2f} {res['n_lines_built']:<12}")
    print(f"{'='*60}")

    return results


# ============================================================
# 4. FULL PHASE 3 PIPELINE
# ============================================================

def run_phase3(phase2_result: Dict,
               qaoa_depths: List[int] = [1, 2],
               ) -> Dict:
    """
    Execute the complete Phase 3 quantum pipeline.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Quantum Optimization")
    print("=" * 60)

    qp = phase2_result['qp']
    classical_result = phase2_result['classical_result']
    penalty = PENALTY_CONGESTION

    # Run benchmark suite
    benchmark = run_benchmark(qp, classical_result, qaoa_depths, penalty)

    return {
        'benchmark': benchmark,
        'qp': qp,
    }