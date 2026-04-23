"""
Phase 1: Grid Data Pipeline
- Load CATS network via pandapower
- Extract manageable subgrids using BFS
- Run DC power flow under stressed conditions
- Detect congested lines
- Generate candidate expansion lines
- Compute PTDF-based influence scores

References:
  [4] Cuenca et al., IEEE Trans. Power Systems, 2024
  [5] Koch et al., QOBLIB, arXiv:2504.03832, 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import deque
from itertools import combinations
from typing import Tuple, Dict, List, Optional

import pandapower as pp
import pandapower.networks as pn
from pandapower.converter.matpower import from_mpc
import pandapower.topology as top

from config import (
    CATS_MATPOWER, CATS_FREQ_HZ, BASE_MVA,
    LOAD_MULTIPLIER, THERMAL_LIMIT_FRACTION, CONGESTION_THRESHOLD,
    MAX_CANDIDATE_DISTANCE_KM, CANDIDATE_LINE_COST_PER_KM,
    CANDIDATE_CAPACITY_MW
)


# ============================================================
# 1. LOAD CATS NETWORK
# ============================================================

def load_cats_network() -> pp.pandapowerNet:
    """Load the full CATS network from the MATPOWER .m file."""
    net = from_mpc(str(CATS_MATPOWER), f_hz=CATS_FREQ_HZ)
    print(f"[CATS] Loaded: {len(net.bus)} buses, {len(net.line)} lines, "
          f"{len(net.gen)} generators, {len(net.load)} loads")
    return net


# ============================================================
# 2. SUBGRID EXTRACTION (BFS-based)
# ============================================================

def extract_subgrid(net: pp.pandapowerNet,
                    seed_bus: int,
                    n_buses: int) -> pp.pandapowerNet:
    """
    Extract a connected subgrid of n_buses around seed_bus using BFS.
    This gives us controllable problem sizes for benchmarking.
    Returns a new pandapower network.
    """
    # Build adjacency from line table
    G = nx.Graph()
    for idx, row in net.line.iterrows():
        if row['in_service']:
            G.add_edge(int(row['from_bus']), int(row['to_bus']), line_idx=idx)

    if seed_bus not in G.nodes:
        # Find the closest valid bus
        valid_buses = list(G.nodes)
        seed_bus = valid_buses[0]
        print(f"[WARN] Seed bus not in graph, using bus {seed_bus}")

    # BFS to collect n_buses
    visited = set()
    queue = deque([seed_bus])
    while queue and len(visited) < n_buses:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                queue.append(neighbor)

    selected_buses = sorted(visited)
    print(f"[Subgrid] Extracted {len(selected_buses)} buses around seed {seed_bus}")

    # Create subnetwork using pandapower select
    # We need to find all lines within the selected buses
    sub_net = pp.create_empty_network(f_hz=CATS_FREQ_HZ)
    sub_net["baseMVA"] = BASE_MVA

    # Bus mapping: old_idx -> new_idx
    bus_map = {}
    for new_idx, old_idx in enumerate(selected_buses):
        row = net.bus.loc[old_idx]
        bus_map[old_idx] = pp.create_bus(
            sub_net,
            vn_kv=row['vn_kv'],
            name=f"bus_{old_idx}",
            max_vm_pu=row.get('max_vm_pu', 1.06),
            min_vm_pu=row.get('min_vm_pu', 0.94),
        )

    # Add lines (only those with both endpoints in subgrid)
    line_count = 0
    for idx, row in net.line.iterrows():
        fb, tb = int(row['from_bus']), int(row['to_bus'])
        if fb in bus_map and tb in bus_map and row['in_service']:
            pp.create_line_from_parameters(
                sub_net,
                from_bus=bus_map[fb],
                to_bus=bus_map[tb],
                length_km=max(row['length_km'], 0.1),
                r_ohm_per_km=max(row['r_ohm_per_km'], 1e-6),
                x_ohm_per_km=max(row['x_ohm_per_km'], 1e-4),
                c_nf_per_km=row.get('c_nf_per_km', 0),
                max_i_ka=row['max_i_ka'],
                name=f"line_{idx}",
            )
            line_count += 1

    # Add loads
    load_count = 0
    for idx, row in net.load.iterrows():
        bus = int(row['bus'])
        if bus in bus_map:
            pp.create_load(
                sub_net,
                bus=bus_map[bus],
                p_mw=row['p_mw'],
                q_mvar=row['q_mvar'],
                name=f"load_{idx}",
            )
            load_count += 1

    # Add generators
    gen_count = 0
    for idx, row in net.gen.iterrows():
        bus = int(row['bus'])
        if bus in bus_map:
            pp.create_gen(
                sub_net,
                bus=bus_map[bus],
                p_mw=row['p_mw'],
                max_p_mw=row.get('max_p_mw', row['p_mw'] * 1.2),
                min_p_mw=row.get('min_p_mw', 0),
                vm_pu=row.get('vm_pu', 1.0),
                name=f"gen_{idx}",
            )
            gen_count += 1

    # Add external grid (slack bus) at the first generator bus, or first bus
    ext_bus = bus_map[selected_buses[0]]
    # Prefer a generator bus for the slack
    for idx, row in net.gen.iterrows():
        if int(row['bus']) in bus_map:
            ext_bus = bus_map[int(row['bus'])]
            break
    for idx, row in net.ext_grid.iterrows():
        if int(row['bus']) in bus_map:
            ext_bus = bus_map[int(row['bus'])]
            break

    pp.create_ext_grid(sub_net, bus=ext_bus, vm_pu=1.0, name="slack")

    print(f"[Subgrid] Components: {len(sub_net.bus)} buses, {line_count} lines, "
          f"{load_count} loads, {gen_count} gens")

    return sub_net, bus_map


# ============================================================
# 3. DC POWER FLOW & CONGESTION DETECTION
# ============================================================

def run_dc_powerflow(net: pp.pandapowerNet,
                     load_multiplier: float = 1.0) -> pp.pandapowerNet:
    """
    Run DC power flow with optionally scaled loads to induce congestion.
    DC approximation: linearized, lossless, voltage magnitudes = 1 pu.
    This is standard for transmission planning studies.
    """
    # Scale loads
    if load_multiplier != 1.0:
        net.load['p_mw'] *= load_multiplier
        net.load['q_mvar'] *= load_multiplier

    try:
        pp.rundcpp(net)
        if net.res_line.empty:
            print("[WARN] DC power flow returned empty results")
            return net
        print(f"[DC-PF] Converged. Max line loading: "
              f"{net.res_line['loading_percent'].max():.1f}%")
    except Exception as e:
        print(f"[DC-PF] Failed: {e}")
        # Try with relaxed settings
        try:
            pp.rundcpp(net, trafo_model="pi", check_connectivity=True)
        except Exception as e2:
            print(f"[DC-PF] Also failed with relaxed: {e2}")

    return net


def detect_congestion(net: pp.pandapowerNet,
                      threshold_pct: float = 100.0
                      ) -> pd.DataFrame:
    """
    Identify congested lines (loading > threshold %).
    Returns DataFrame with line index, from/to bus, loading %, flow MW.
    """
    if net.res_line.empty:
        print("[WARN] No power flow results available")
        return pd.DataFrame()

    congested = net.res_line[net.res_line['loading_percent'] > threshold_pct].copy()
    congested = congested.join(net.line[['from_bus', 'to_bus', 'name', 'max_i_ka']])
    congested['flow_mw'] = net.res_line.loc[congested.index, 'p_from_mw'].abs()

    print(f"[Congestion] {len(congested)} lines above {threshold_pct}% loading")
    if len(congested) > 0:
        print(f"  Top 5 congested:")
        top5 = congested.nlargest(5, 'loading_percent')
        for idx, row in top5.iterrows():
            print(f"    Line {idx}: {row['loading_percent']:.1f}% "
                  f"(bus {int(row['from_bus'])} -> {int(row['to_bus'])})")

    return congested


# ============================================================
# 4. CANDIDATE LINE GENERATION
# ============================================================

def compute_bus_distances(net: pp.pandapowerNet) -> pd.DataFrame:
    """
    Compute pairwise electrical distances between buses.
    Uses graph shortest path on impedance-weighted topology.
    """
    G = nx.Graph()
    for idx, row in net.line.iterrows():
        if row['in_service']:
            # Impedance as weight (proxy for electrical distance)
            z = np.sqrt(row['r_ohm_per_km']**2 + row['x_ohm_per_km']**2) * row['length_km']
            G.add_edge(int(row['from_bus']), int(row['to_bus']),
                       weight=z, length_km=row['length_km'])
    return G


def generate_candidates(net: pp.pandapowerNet,
                        congested_lines: pd.DataFrame,
                        max_candidates: Optional[int] = None
                        ) -> pd.DataFrame:
    """
    Generate candidate new lines based on congested regions.
    Strategy (following Cuenca et al. [4]):
      1. Identify buses adjacent to congested lines
      2. Generate candidate edges between nearby buses not currently connected
      3. Assign cost proportional to electrical distance

    Returns DataFrame with columns:
      from_bus, to_bus, cost, capacity_mw, est_length_km
    """
    G = compute_bus_distances(net)

    # Collect buses involved in or near congestion
    congested_buses = set()
    if len(congested_lines) > 0:
        for _, row in congested_lines.iterrows():
            congested_buses.add(int(row['from_bus']))
            congested_buses.add(int(row['to_bus']))
            # Also add 1-hop neighbors
            for bus in [int(row['from_bus']), int(row['to_bus'])]:
                if bus in G:
                    congested_buses.update(G.neighbors(bus))

    # If few congested buses, use all buses
    if len(congested_buses) < 4:
        congested_buses = set(net.bus.index.tolist())

    # Existing edges
    existing_edges = set()
    for _, row in net.line.iterrows():
        fb, tb = int(row['from_bus']), int(row['to_bus'])
        existing_edges.add((min(fb, tb), max(fb, tb)))

    # Generate candidates: pairs of congested-region buses without a direct line
    candidates = []
    bus_list = sorted(congested_buses)

    for i, bi in enumerate(bus_list):
        for bj in bus_list[i+1:]:
            edge = (min(bi, bj), max(bi, bj))
            if edge in existing_edges:
                continue
            # Use impedance-weighted shortest path as electrical distance proxy
            if bi in G and bj in G:
                try:
                    # Impedance distance (better proxy than length_km for MATPOWER data)
                    z_dist = nx.shortest_path_length(G, bi, bj, weight='weight')
                    hop_dist = nx.shortest_path_length(G, bi, bj)
                    if hop_dist <= 4:  # Within 4 hops
                        # Cost varies by impedance distance + random perturbation
                        # to create non-trivial optimization landscape
                        cost = 1.0 + z_dist * 10.0 + hop_dist * 0.5
                        # Capacity inversely related to distance (shorter = stronger)
                        cap = CANDIDATE_CAPACITY_MW / (1 + 0.3 * hop_dist)
                        candidates.append({
                            'from_bus': bi,
                            'to_bus': bj,
                            'cost': round(cost, 3),
                            'capacity_mw': round(cap, 1),
                            'est_length_km': round(z_dist, 3),
                        })
                except nx.NetworkXNoPath:
                    continue

    candidates_df = pd.DataFrame(candidates)

    if max_candidates and len(candidates_df) > max_candidates:
        # Prioritize candidates near the most congested lines
        candidates_df = candidates_df.nsmallest(max_candidates, 'cost')

    candidates_df = candidates_df.reset_index(drop=True)
    print(f"[Candidates] Generated {len(candidates_df)} candidate lines")

    return candidates_df


# ============================================================
# 5. PTDF & INFLUENCE SCORE COMPUTATION
# ============================================================

def compute_ptdf_matrix(net: pp.pandapowerNet) -> np.ndarray:
    """
    Compute Power Transfer Distribution Factors (PTDF) matrix.
    PTDF[l, b] = sensitivity of flow on line l to injection at bus b.

    Under DC assumptions: PTDF = B_f * B_bus^{-1}
    where B_f = branch susceptance matrix, B_bus = bus susceptance matrix.
    """
    n_bus = len(net.bus)
    n_line = len(net.line[net.line['in_service']])

    bus_idx = {bus: i for i, bus in enumerate(net.bus.index)}
    slack_bus_idx = bus_idx[net.ext_grid.bus.iloc[0]]

    # Build bus admittance matrix (imaginary part only for DC)
    B_bus = np.zeros((n_bus, n_bus))
    B_f = np.zeros((n_line, n_bus))

    active_lines = net.line[net.line['in_service']]
    for line_i, (idx, row) in enumerate(active_lines.iterrows()):
        fb = bus_idx[int(row['from_bus'])]
        tb = bus_idx[int(row['to_bus'])]

        # Susceptance: b = 1/x for DC model
        x_pu = row['x_ohm_per_km'] * row['length_km'] / (
            (net.bus.loc[int(row['from_bus']), 'vn_kv'])**2 / BASE_MVA)
        if abs(x_pu) < 1e-10:
            x_pu = 1e-4  # Avoid division by zero
        b = 1.0 / x_pu

        B_bus[fb, fb] += b
        B_bus[tb, tb] += b
        B_bus[fb, tb] -= b
        B_bus[tb, fb] -= b

        B_f[line_i, fb] = b
        B_f[line_i, tb] = -b

    # Remove slack bus row/col, invert, then restore
    mask = np.ones(n_bus, dtype=bool)
    mask[slack_bus_idx] = False
    B_reduced = B_bus[np.ix_(mask, mask)]

    try:
        B_inv = np.linalg.inv(B_reduced)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for singular matrices
        B_inv = np.linalg.pinv(B_reduced)

    # Full inverse with slack row/col = 0
    B_inv_full = np.zeros((n_bus, n_bus))
    idx_map = np.where(mask)[0]
    for i, ii in enumerate(idx_map):
        for j, jj in enumerate(idx_map):
            B_inv_full[ii, jj] = B_inv[i, j]

    PTDF = B_f @ B_inv_full
    print(f"[PTDF] Computed {PTDF.shape[0]} x {PTDF.shape[1]} PTDF matrix")

    return PTDF, bus_idx


def compute_influence_scores(net: pp.pandapowerNet,
                             congested_lines: pd.DataFrame,
                             candidates: pd.DataFrame,
                             ptdf: np.ndarray,
                             bus_idx: Dict
                             ) -> np.ndarray:
    """
    Compute influence score matrix A[l, e]:
    How much does building candidate line e reduce congestion on line l?

    Based on Cuenca et al. [4]: We approximate the flow redistribution
    when a new line is added by computing the change in PTDF.

    For the BLP formulation, we binarize:
      A[l, e] = 1 if candidate e significantly reduces flow on congested line l
    """
    active_lines = net.line[net.line['in_service']]
    line_idx_map = {idx: i for i, idx in enumerate(active_lines.index)}

    n_congested = len(congested_lines)
    n_candidates = len(candidates)

    if n_congested == 0 or n_candidates == 0:
        return np.zeros((n_congested, n_candidates))

    # Influence score: for each candidate, estimate how it relieves each congestion
    A = np.zeros((n_congested, n_candidates))

    for c_idx, (_, cand) in enumerate(candidates.iterrows()):
        fb_new = bus_idx.get(cand['from_bus'])
        tb_new = bus_idx.get(cand['to_bus'])
        if fb_new is None or tb_new is None:
            continue

        # The new line creates an alternative path.
        # Approximate: flow diverted ~ proportional to PTDF difference
        # at the candidate endpoints for each congested line
        for l_idx, (cong_line_idx, cong_row) in enumerate(congested_lines.iterrows()):
            if cong_line_idx in line_idx_map:
                l_i = line_idx_map[cong_line_idx]
                # PTDF difference at candidate endpoints gives flow diversion potential
                ptdf_diff = abs(ptdf[l_i, fb_new] - ptdf[l_i, tb_new])
                # Normalize by the congestion severity
                severity = cong_row['loading_percent'] / 100.0 - 1.0  # excess fraction
                influence = ptdf_diff * severity * cand['capacity_mw'] / BASE_MVA
                A[l_idx, c_idx] = influence

    # Binarize: A[l, e] = 1 if influence is significant
    if A.max() > 0:
        threshold = np.percentile(A[A > 0], 30) if np.sum(A > 0) > 0 else 0
        A_binary = (A > threshold).astype(int)
    else:
        A_binary = A.astype(int)

    print(f"[Influence] Score matrix: {A.shape}, "
          f"non-zero entries: {np.count_nonzero(A_binary)}")

    return A_binary, A


# ============================================================
# 6. FULL PHASE 1 PIPELINE
# ============================================================

def run_phase1(seed_bus: int = 100,
               n_buses: int = 30,
               load_mult: float = 1.5,
               max_candidates: int = 20
               ) -> Dict:
    """
    Execute the complete Phase 1 pipeline.
    Returns a dictionary with all data needed for Phase 2 (QUBO formulation).
    """
    print("=" * 60)
    print("PHASE 1: Grid Data Pipeline")
    print("=" * 60)

    # Step 1: Load CATS
    print("\n--- Step 1: Loading CATS network ---")
    full_net = load_cats_network()

    # Step 2: Extract subgrid
    print(f"\n--- Step 2: Extracting {n_buses}-bus subgrid ---")
    sub_net, bus_map = extract_subgrid(full_net, seed_bus, n_buses)

    # Step 3: Run DC power flow with stress
    print(f"\n--- Step 3: DC power flow (load x{load_mult}) ---")
    sub_net = run_dc_powerflow(sub_net, load_multiplier=load_mult)

    # Step 4: Detect congestion
    print(f"\n--- Step 4: Congestion detection ---")
    congested = detect_congestion(sub_net, threshold_pct=100.0)

    # If no congestion, try higher multiplier
    if len(congested) == 0:
        print("[INFO] No congestion found. Increasing load multiplier...")
        for mult in [2.0, 2.5, 3.0]:
            # Reload to reset
            sub_net2, _ = extract_subgrid(full_net, seed_bus, n_buses)
            sub_net2 = run_dc_powerflow(sub_net2, load_multiplier=mult)
            congested = detect_congestion(sub_net2, threshold_pct=100.0)
            if len(congested) > 0:
                sub_net = sub_net2
                load_mult = mult
                break

    # Also detect near-congestion for richer candidate generation
    near_congested = detect_congestion(sub_net, threshold_pct=80.0)

    # Step 5: Generate candidate lines
    print(f"\n--- Step 5: Generating candidate lines ---")
    candidates = generate_candidates(sub_net, near_congested,
                                     max_candidates=max_candidates)

    # Step 6: Compute PTDF and influence scores
    print(f"\n--- Step 6: Computing PTDF & influence scores ---")
    ptdf, bus_idx_map = compute_ptdf_matrix(sub_net)
    A_binary, A_raw = compute_influence_scores(
        sub_net, congested, candidates, ptdf, bus_idx_map)

    # Package results
    result = {
        'network': sub_net,
        'full_network': full_net,
        'bus_map': bus_map,
        'congested_lines': congested,
        'near_congested_lines': near_congested,
        'candidates': candidates,
        'ptdf': ptdf,
        'bus_idx_map': bus_idx_map,
        'influence_binary': A_binary,
        'influence_raw': A_raw,
        'n_qubits': len(candidates),
        'load_multiplier': load_mult,
    }

    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE")
    print(f"  Subgrid: {len(sub_net.bus)} buses, {len(sub_net.line)} lines")
    print(f"  Congested lines: {len(congested)}")
    print(f"  Candidate new lines: {len(candidates)}")
    print(f"  Required qubits: {len(candidates)}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    result = run_phase1(seed_bus=100, n_buses=30, load_mult=1.5, max_candidates=20)