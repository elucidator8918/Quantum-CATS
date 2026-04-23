"""
Main Pipeline: Quantum Grid Expansion Planning
E.ON Global Quantum + AI Challenge 2026

Orchestrates all phases:
  Phase 1: Grid data loading, congestion detection, candidate generation
  Phase 2: BLP formulation, QUBO conversion, classical benchmark
  Phase 3: Quantum optimization (QAOA, VQE), benchmarking
  Phase 4: Results analysis and visualization

Usage:
  python main.py
"""

import sys
import os
import json
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from phase1_grid_data import run_phase1
from phase2_qubo import run_phase2
from phase3_quantum import run_phase3
from config import RESULTS_DIR


def run_full_pipeline(seed_bus: int = 100,
                      n_buses: int = 30,
                      load_mult: float = 1.5,
                      max_candidates: int = 15,
                      qaoa_depths: list = [1, 2],
                      budget: float = None):
    """
    Run the complete quantum grid expansion pipeline.
    """
    pipeline_start = time.time()

    print("=" * 70)
    print("  QUANTUM GRID EXPANSION PLANNING")
    print("  E.ON Global Quantum + AI Challenge 2026")
    print("  Using CATS (California Test System) Dataset")
    print("=" * 70)
    print(f"\nConfig: {n_buses} buses, load x{load_mult}, "
          f"max {max_candidates} candidates, QAOA p={qaoa_depths}")

    # ========== PHASE 1 ==========
    p1 = run_phase1(
        seed_bus=seed_bus,
        n_buses=n_buses,
        load_mult=load_mult,
        max_candidates=max_candidates,
    )

    if len(p1['candidates']) == 0:
        print("\n[ABORT] No candidate lines generated. "
              "Try different seed_bus or higher load_mult.")
        return None

    # ========== PHASE 2 ==========
    p2 = run_phase2(p1, budget=budget)

    if not p2:
        print("\n[ABORT] Phase 2 failed.")
        return None

    # ========== PHASE 3 ==========
    p3 = run_phase3(p2, qaoa_depths=qaoa_depths)

    # ========== RESULTS SUMMARY ==========
    total_time = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Dataset: CATS (California Test System)")
    print(f"  Subgrid: {len(p1['network'].bus)} buses, "
          f"{len(p1['network'].line)} lines")
    print(f"  Congested lines: {len(p1['congested_lines'])}")
    print(f"  Candidate lines: {len(p1['candidates'])}")
    print(f"  QUBO qubits: {p2['n_qubits_qubo']}")
    print(f"\n  Classical optimal cost: "
          f"{p2['classical_result']['fval']:.4f}")
    print(f"  Classical lines to build: "
          f"{p2['classical_result']['n_lines_built']}")

    # Best quantum result
    benchmark = p3['benchmark']
    best_quantum = None
    for key, res in benchmark.items():
        if key == 'classical' or 'error' in res:
            continue
        if best_quantum is None or res['fval'] < best_quantum['fval']:
            best_quantum = res
            best_quantum['method_name'] = key

    if best_quantum:
        print(f"\n  Best quantum result ({best_quantum['method_name']}):")
        print(f"    Cost: {best_quantum['fval']:.4f}")
        print(f"    Approx ratio: {best_quantum.get('approx_ratio', 'N/A')}")
        print(f"    Lines to build: {best_quantum['n_lines_built']}")
        print(f"    Time: {best_quantum['time_s']:.2f}s")

    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print("=" * 70)

    # Save results summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {
        'config': {
            'seed_bus': seed_bus,
            'n_buses': n_buses,
            'load_mult': load_mult,
            'max_candidates': max_candidates,
            'qaoa_depths': qaoa_depths,
        },
        'grid': {
            'n_buses': len(p1['network'].bus),
            'n_lines': len(p1['network'].line),
            'n_congested': len(p1['congested_lines']),
            'n_candidates': len(p1['candidates']),
        },
        'qubo': {
            'n_qubits_original': p2['n_qubits_original'],
            'n_qubits_total': p2['n_qubits_qubo'],
        },
        'classical': {
            'fval': p2['classical_result']['fval'],
            'n_lines_built': p2['classical_result']['n_lines_built'],
            'time_s': p2['classical_result']['time_s'],
        },
        'quantum_results': {},
        'total_time_s': total_time,
    }

    for key, res in benchmark.items():
        if key == 'classical':
            continue
        if 'error' in res:
            summary['quantum_results'][key] = {'error': res['error']}
        else:
            summary['quantum_results'][key] = {
                'fval': res['fval'],
                'approx_ratio': res.get('approx_ratio'),
                'n_lines_built': res['n_lines_built'],
                'time_s': res['time_s'],
            }

    with open(RESULTS_DIR / 'pipeline_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR / 'pipeline_results.json'}")

    return {
        'phase1': p1,
        'phase2': p2,
        'phase3': p3,
        'summary': summary,
    }


# ============================================================
# SCALING STUDY: Find classically hard instances
# ============================================================

def run_scaling_study(sizes: list = [10, 15, 20, 25, 30],
                      seed_bus: int = 100,
                      load_mult: float = 2.0):
    """
    Run the pipeline at increasing subgrid sizes to identify
    the scaling behavior of classical vs quantum solvers.
    This is the PRIMARY OBJECTIVE of the challenge.
    """
    print("\n" + "=" * 70)
    print("  SCALING STUDY: Finding classically hard instances")
    print("=" * 70)

    scaling_results = []

    for n_buses in sizes:
        max_cands = min(n_buses, 20)
        print(f"\n{'='*40}")
        print(f"  Instance: {n_buses} buses, max {max_cands} candidates")
        print(f"{'='*40}")

        try:
            result = run_full_pipeline(
                seed_bus=seed_bus,
                n_buses=n_buses,
                load_mult=load_mult,
                max_candidates=max_cands,
                qaoa_depths=[1],
            )
            if result:
                scaling_results.append(result['summary'])
        except Exception as e:
            print(f"  [ERROR] Instance {n_buses} failed: {e}")
            scaling_results.append({
                'config': {'n_buses': n_buses},
                'error': str(e),
            })

    # Save scaling results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_DIR / 'scaling_study.json', 'w') as f:
        json.dump(scaling_results, f, indent=2, default=str)

    print(f"\nScaling study saved to {RESULTS_DIR / 'scaling_study.json'}")
    return scaling_results


if __name__ == "__main__":
    # Single instance run
    result = run_full_pipeline(
        seed_bus=18,
        n_buses=30,
        load_mult=1.5,
        max_candidates=12,
        qaoa_depths=[1, 2],
    )