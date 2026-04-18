"""
Direction 1B — Non-Unit Capacities
====================================
Central question: Does the "capacity-weighted SP-balanced" condition
(equal TOTAL capacity per layer, but heterogeneous per-edge capacities)
preserve Shapley-in-core?

We also compute the Shapley–Nucleolus L1-distance to understand how
equitable the Shapley value is even when it IS in the core.

Experiment structure
---------------------
For each topology family we sweep a capacity-imbalance parameter α:

  α = 0  →  unit capacities (perfectly balanced, baseline)
  α > 0  →  within each layer, edge capacities are (1+α, 1-α, 1, 1, ...)
             so total layer capacity is preserved but per-edge is skewed
  α = 1  →  extreme: one edge in each layer has capacity 2, one has 0
             (equivalent to removing an edge → full bottleneck)

Topologies studied
-------------------
  SP-Balanced (2×2, 2×3, 3×2)  — symmetric baseline
  Capacity-Weighted SP-Balanced — equal layer totals, unequal per-edge
  Funnel variants               — how does capacity scaling change deficit?
  Mixed: one balanced, one unbalanced layer

Metrics computed per (topology, α)
------------------------------------
  1. Exact Shapley value (fractions → floats for capacity games)
  2. Core deficit
  3. Nucleolus (via iterative LP — Maschler's sequential LP method)
  4. L1 distance  ||φ − nucleolus||₁
  5. σ²(edge-cut-frequencies) under the new capacities

Key hypotheses tested
----------------------
  H1: Capacity-weighted SP-balanced (equal layer totals) is NOT sufficient
      for core stability when per-edge capacities are unequal.
  H2: The Shapley–Nucleolus L1-distance grows monotonically with α
      (more imbalance → farther from the nucleolus).
  H3: For funnel topologies, increasing the bottleneck capacity
      reduces the deficit (less bottleneck tax).
"""

from fractions import Fraction
from itertools import combinations
from collections import defaultdict, deque
import math, copy
import numpy as np
from scipy.optimize import linprog


# ═══════════════════════════════════════════════════════════════
# 1.  Max-flow (Edmonds-Karp) — generalised to real capacities
# ═══════════════════════════════════════════════════════════════

def edmonds_karp_real(capacities, nodes, source, sink):
    """
    capacities: dict (u,v) -> float
    Returns max-flow value (float).
    """
    # build residual graph
    res = defaultdict(float)
    for (u, v), c in capacities.items():
        res[(u, v)] += c   # allow multi-edges

    def bfs():
        parent = {source: None}
        queue  = deque([source])
        while queue:
            u = queue.popleft()
            for v in nodes:
                if v not in parent and res[(u, v)] > 1e-9:
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    flow = 0.0
    while True:
        parent = bfs()
        if parent is None:
            break
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, res[(u, v)])
            v = u
        v = sink
        while v != source:
            u = parent[v]
            res[(u, v)] -= path_flow
            res[(v, u)] += path_flow
            v = u
        flow += path_flow
    return flow


def max_flow_coalition(edge_list, capacities, nodes, source, sink, coalition):
    """Max-flow using only edges in coalition (set of indices)."""
    cap = defaultdict(float)
    for i in coalition:
        u, v = edge_list[i]
        cap[(u, v)] += capacities[i]
    return edmonds_karp_real(cap, nodes, source, sink)


# ═══════════════════════════════════════════════════════════════
# 2.  Characteristic function + Shapley (real-valued)
# ═══════════════════════════════════════════════════════════════

def compute_char_func(edge_list, capacities, nodes, source, sink):
    """Returns dict: frozenset(indices) -> float."""
    n = len(edge_list)
    v = {}
    for mask in range(1 << n):
        S = frozenset(i for i in range(n) if mask & (1 << i))
        v[S] = max_flow_coalition(edge_list, capacities, nodes, source, sink, S)
    return v


def shapley_real(n, v):
    """Shapley value using coalition-sum formula. Returns list of floats."""
    phi = [0.0] * n
    n_fact = math.factorial(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            S = frozenset(j for j in range(n) if mask & (1 << j))
            s = len(S)
            w = math.factorial(s) * math.factorial(n - s - 1) / n_fact
            phi[i] += w * (v[S | {i}] - v[S])
    return phi


def core_deficit_real(n, v, phi):
    """Returns max over S of (v(S) - sum_phi_i)."""
    worst = 0.0
    for mask in range(1, 1 << n):
        S = frozenset(i for i in range(n) if mask & (1 << i))
        excess = v[S] - sum(phi[i] for i in S)
        if excess > worst:
            worst = excess
    return worst


# ═══════════════════════════════════════════════════════════════
# 3.  Nucleolus via sequential LP (Maschler / Kohlberg method)
# ═══════════════════════════════════════════════════════════════

def compute_nucleolus(n, v):
    """
    Compute the nucleolus using iterative LP.

    The nucleolus lexicographically minimises the vector of
    sorted (descending) coalition excesses:
        e(S, x) = v(S) - sum_{i in S} x_i

    Algorithm (Maschler et al.):
      Repeat:
        1. Solve LP: minimise ε subject to
              v(S) - sum x_i ≤ ε   for all active coalitions S
              sum x_i = v(N)         (efficiency)
              previously-fixed constraints
        2. Find all coalitions achieving excess = ε*.
           Add constraints e(S,x) = ε* for those coalitions.
        3. Remove them from the active set.
      Until all coalitions are fixed.

    Returns nucleolus as numpy array of shape (n,).
    """
    N = frozenset(range(n))
    v_N = v[N]

    # all proper non-empty coalitions (exclude empty and grand)
    all_coalitions = [
        frozenset(i for i in range(n) if mask & (1 << i))
        for mask in range(1, (1 << n) - 1)
    ]

    # current LP variable: x = [x_0, ..., x_{n-1}, epsilon]
    # indices: 0..n-1 → payoffs, n → epsilon

    active = list(all_coalitions)
    fixed_constraints = []   # list of (coalition, rhs) meaning sum_x_i = rhs

    x_sol = np.full(n, v_N / n)   # warm start: equal split

    MAX_ITER = len(all_coalitions) + 5
    for iteration in range(MAX_ITER):
        if not active:
            break

        # ── build LP ──
        # variables: [x_0 .. x_{n-1}, epsilon]
        # minimise: epsilon  (index n)
        c_obj = np.zeros(n + 1)
        c_obj[n] = 1.0

        A_ub = []   # inequality: v(S) - sum x_i ≤ epsilon  ↔  -sum x_i - epsilon ≤ -v(S)
        b_ub = []

        for S in active:
            row = np.zeros(n + 1)
            for i in S:
                row[i] = -1.0
            row[n] = -1.0
            A_ub.append(row)
            b_ub.append(-v[S])

        A_eq = []
        b_eq = []

        # efficiency: sum x_i = v(N)
        eff_row = np.zeros(n + 1)
        eff_row[:n] = 1.0
        A_eq.append(eff_row)
        b_eq.append(v_N)

        # previously fixed coalitions
        for (S_fixed, rhs) in fixed_constraints:
            row = np.zeros(n + 1)
            for i in S_fixed:
                row[i] = -1.0
            A_eq.append(row)
            b_eq.append(-rhs)

        bounds = [(None, None)] * n + [(None, None)]  # epsilon unbounded

        result = linprog(
            c_obj,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            # fallback: return current best
            break

        x_sol  = result.x[:n]
        eps_star = result.x[n]

        # ── find tight coalitions ──
        TOL = 1e-6
        newly_fixed = []
        still_active = []
        for S in active:
            excess = v[S] - sum(x_sol[i] for i in S)
            if abs(excess - eps_star) < TOL:
                newly_fixed.append(S)
            else:
                still_active.append(S)

        if not newly_fixed:
            break   # numerical issue — stop

        for S in newly_fixed:
            rhs = v[S] - eps_star   # sum x_i = v(S) - eps*  ↔  excess = eps*
            fixed_constraints.append((S, rhs))

        active = still_active

    return x_sol


# ═══════════════════════════════════════════════════════════════
# 4.  σ²(edge-cut-freq) for real-capacity networks
# ═══════════════════════════════════════════════════════════════

def find_min_cuts_real(edge_list, capacities, nodes, source, sink):
    """
    Enumerate all minimum cuts.
    A cut (S_set, T_set) is a min-cut iff its capacity = max-flow.
    """
    # full graph max-flow
    full_cap = defaultdict(float)
    for i, (u, v) in enumerate(edge_list):
        full_cap[(u, v)] += capacities[i]
    F = edmonds_karp_real(full_cap, nodes, source, sink)

    non_st = [nd for nd in nodes if nd != source and nd != sink]
    min_cuts = []
    for mask in range(1 << len(non_st)):
        S_set = {source}
        for idx, nd in enumerate(non_st):
            if mask & (1 << idx):
                S_set.add(nd)
        T_set = set(nodes) - S_set
        if sink not in T_set:
            continue
        cap = sum(capacities[i] for i, (u, v) in enumerate(edge_list)
                  if u in S_set and v in T_set)
        if abs(cap - F) < 1e-6:
            min_cuts.append((frozenset(S_set), frozenset(T_set), cap))
    return min_cuts, F


def sigma2_real(edge_list, capacities, nodes, source, sink):
    min_cuts, _ = find_min_cuts_real(edge_list, capacities, nodes, source, sink)
    n = len(edge_list)
    if not min_cuts:
        return 0.0, 0, [0] * n
    n_cuts = len(min_cuts)
    freq = []
    for i, (u, v) in enumerate(edge_list):
        cnt = sum(1 for (S, T, _) in min_cuts if u in S and v in T)
        freq.append(cnt)
    mean = sum(freq) / n
    var  = sum((f - mean) ** 2 for f in freq) / n
    return var, n_cuts, freq


# ═══════════════════════════════════════════════════════════════
# 5.  Topology factories with capacity parameter α
# ═══════════════════════════════════════════════════════════════

def sp_balanced_caps(k, m, alpha):
    """
    k layers, m parallel paths.
    alpha = imbalance within each layer:
      path j capacity = 1 + alpha * sin(2π j / m)   <- unit-sum preserved
    Specifically we use: cap_j = 1 + alpha * (-1)^j  for simplicity,
    so layer totals are always m (equal) but per-edge varies.

    Returns (nodes, source, sink, edge_list, capacities_list).
    """
    nodes_set = {0, 99}
    edge_list = []
    capacities = []
    for j in range(m):
        prev = 0
        for layer in range(k):
            cur = 100 * (layer + 1) + j
            nodes_set.add(cur)
            # capacity: alternating perturbation, total per layer = m
            cap = 1.0 + alpha * (1 if j % 2 == 0 else -1)
            cap = max(cap, 0.0)   # clamp
            edge_list.append((prev, cur))
            capacities.append(cap)
            prev = cur
        edge_list.append((prev, 99))
        capacities.append(cap)   # same cap for whole path
    return sorted(nodes_set), 0, 99, edge_list, capacities


def sp_layer_equal_total(k, m, alpha):
    """
    Capacity-weighted SP-balanced:
    Each LAYER has equal total capacity m, but per-edge capacities vary.
    Layer j: edges have capacities [1+alpha, 1-alpha, 1, 1, ...] rotated.
    This is the key hypothesis — does equal total suffice?
    """
    nodes_set = {0, 99}
    edge_list  = []
    capacities = []
    # m paths, k layers each
    # within each layer, distribute capacity unevenly across the m paths
    # but keep sum = m
    for j in range(m):
        prev = 0
        for layer in range(k):
            cur = 100 * (layer + 1) + j
            nodes_set.add(cur)
            # path j in this layer: capacity perturbed, sum across j = m
            if j == 0:
                cap = 1.0 + alpha * (m - 1)   # one edge gets the extra
            else:
                cap = 1.0 - alpha              # others lose proportionally
            cap = max(cap, 0.01)
            edge_list.append((prev, cur))
            capacities.append(cap)
            prev = cur
        edge_list.append((prev, 99))
        cap_last = 1.0 + alpha * (m - 1) if j == 0 else max(1.0 - alpha, 0.01)
        capacities.append(cap_last)
    return sorted(nodes_set), 0, 99, edge_list, capacities


def funnel_caps(k_parallel, bottleneck_cap, entry_cap=1.0):
    """
    k_parallel entry edges (capacity entry_cap each) → 1 bottleneck (bottleneck_cap).
    Varying bottleneck_cap shows how deficit changes.
    """
    nodes = [0] + [10 + i for i in range(k_parallel)] + [50, 1]
    edges = [(0, 10+i) for i in range(k_parallel)] + \
            [(10+i, 50) for i in range(k_parallel)] + \
            [(50, 1)]
    caps  = [entry_cap] * k_parallel + [entry_cap] * k_parallel + [bottleneck_cap]
    return nodes, 0, 1, edges, caps


def mixed_layer_caps(alpha):
    """
    2-layer network: Layer A balanced, Layer B imbalanced by alpha.
    Tests whether a single unbalanced layer breaks core stability.
    s -> a1(1.0), a2(1.0) -> b1(1+alpha), b2(1-alpha) -> t
    Layer A total = 2, Layer B total = 2 always.
    """
    nodes = [0, 10, 11, 20, 21, 99]
    edges = [(0,10),(0,11),(10,20),(10,21),(11,20),(11,21),(20,99),(21,99)]
    caps  = [1.0, 1.0,           # s -> a1, s -> a2
             1.0, 1.0,           # a1 -> b1, a1 -> b2  (layer A->B, balanced)
             1.0, 1.0,           # a2 -> b1, a2 -> b2
             1.0+alpha, 1.0-alpha]  # b1->t, b2->t (layer B imbalanced)
    caps  = [max(c, 0.01) for c in caps]
    return nodes, 0, 99, edges, caps


# ═══════════════════════════════════════════════════════════════
# 6.  Single-topology analyser
# ═══════════════════════════════════════════════════════════════

def analyse_one(label, nodes, source, sink, edge_list, capacities):
    n = len(edge_list)
    if n > 13:
        return None

    v    = compute_char_func(edge_list, capacities, nodes, source, sink)
    phi  = shapley_real(n, v)
    def_ = core_deficit_real(n, v, phi)
    nuc  = compute_nucleolus(n, v)
    l1   = float(np.sum(np.abs(np.array(phi) - nuc)))
    s2, n_cuts, freqs = sigma2_real(edge_list, capacities, nodes, source, sink)

    return dict(
        label     = label,
        n         = n,
        caps      = capacities,
        max_flow  = v[frozenset(range(n))],
        phi       = phi,
        nucleolus = nuc.tolist(),
        l1_dist   = l1,
        deficit   = def_,
        in_core   = def_ < 1e-6,
        sigma2    = s2,
        n_cuts    = n_cuts,
        freqs     = freqs,
    )


# ═══════════════════════════════════════════════════════════════
# 7.  Experiment sweeps
# ═══════════════════════════════════════════════════════════════

def run_sp_imbalance_sweep():
    """
    H1 & H2: Sweep α for capacity-weighted SP-balanced (2×2 and 2×3).
    Alpha controls per-edge imbalance while keeping layer totals fixed.
    """
    print("\n" + "="*90)
    print("EXPERIMENT A — Capacity-weighted SP-balanced: sweep α")
    print("  Layer totals are EQUAL (∑ cap = m per layer), but per-edge varies")
    print("="*90)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    configs = [("2×2", 2, 2), ("2×3", 2, 3), ("3×2", 3, 2)]

    results_by_config = {}
    for cfg_name, k, m in configs:
        print(f"\n  SP-Balanced {cfg_name}  (k={k} layers, m={m} paths/layer)")
        print(f"  {'α':>6}  {'MaxFlow':>8}  {'σ²':>8}  {'Deficit':>10}  "
              f"{'InCore':>7}  {'L1(φ,nuc)':>12}  capacities")
        print("  " + "-"*85)
        rows = []
        for alpha in alphas:
            nodes, src, snk, edges, caps = sp_layer_equal_total(k, m, alpha)
            r = analyse_one(f"SP({k}x{m}) a={alpha:.1f}", nodes, src, snk, edges, caps)
            if r is None:
                continue
            rows.append((alpha, r))
            cap_s = "[" + ", ".join(f"{c:.2f}" for c in caps) + "]"
            phi_s = "[" + ", ".join(f"{p:.4f}" for p in r["phi"]) + "]"
            nuc_s = "[" + ", ".join(f"{p:.4f}" for p in r["nucleolus"]) + "]"
            print(f"  {alpha:>6.2f}  {r['max_flow']:>8.3f}  {r['sigma2']:>8.4f}  "
                  f"{r['deficit']:>10.4f}  {'Yes' if r['in_core'] else 'No':>7}  "
                  f"{r['l1_dist']:>12.6f}  caps={cap_s}")
            print(f"         phi(Shapley)  = {phi_s}")
            print(f"         eta(Nucleolus)= {nuc_s}")
            print()
        results_by_config[cfg_name] = rows

    return results_by_config


def run_funnel_sweep():
    """
    H3: How does increasing bottleneck capacity change the deficit?
    Also: does Shapley–Nucleolus distance track the deficit?
    """
    print("\n" + "="*90)
    print("EXPERIMENT B — Funnel topology: vary bottleneck capacity")
    print("  k parallel entry edges (cap=1.0 each) → 1 bottleneck (cap varies)")
    print("="*90)

    bottleneck_caps = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    k_vals = [2, 3]

    for k in k_vals:
        print(f"\n  Funnel: {k} parallel paths → 1 bottleneck")
        print(f"  {'b_cap':>6}  {'MaxFlow':>8}  {'σ²':>8}  {'Deficit':>10}  "
              f"{'InCore':>7}  {'L1(φ,nuc)':>12}  φ-bottleneck  nuc-bottleneck")
        print("  " + "-"*90)
        for bc in bottleneck_caps:
            nodes, src, snk, edges, caps = funnel_caps(k, bc)
            r = analyse_one(f"Funnel({k}+1) bc={bc:.2f}", nodes, src, snk, edges, caps)
            if r is None:
                continue
            phi_bt  = r["phi"][-1]
            nuc_bt  = r["nucleolus"][-1]
            phi_s = "[" + ", ".join(f"{p:.4f}" for p in r["phi"]) + "]"
            nuc_s = "[" + ", ".join(f"{p:.4f}" for p in r["nucleolus"]) + "]"
            print(f"  {bc:>6.2f}  {r['max_flow']:>8.3f}  {r['sigma2']:>8.4f}  "
                  f"{r['deficit']:>10.4f}  {'Yes' if r['in_core'] else 'No':>7}  "
                  f"{r['l1_dist']:>12.6f}  {phi_bt:>13.4f}  {nuc_bt:>14.4f}")
            print(f"         phi(Shapley)  = {phi_s}")
            print(f"         eta(Nucleolus)= {nuc_s}")
            print()


def run_mixed_layer_sweep():
    """
    Mixed experiment: balanced layer A, imbalanced layer B.
    Tests whether a single asymmetric layer breaks core stability.
    """
    print("\n" + "="*90)
    print("EXPERIMENT C — Mixed layers: Layer A balanced, Layer B imbalanced by α")
    print("  2-layer (2×2) network: Layer A cap=(1,1), Layer B cap=(1+α, 1-α)")
    print("  Layer totals always equal 2 for both layers")
    print("="*90)

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\n  {'α':>6}  {'MaxFlow':>8}  {'σ²':>8}  {'Deficit':>10}  "
          f"{'InCore':>7}  {'L1(φ,nuc)':>12}  φ=[...] ")
    print("  " + "-"*85)

    for alpha in alphas:
        nodes, src, snk, edges, caps = mixed_layer_caps(alpha)
        r = analyse_one(f"Mixed a={alpha:.1f}", nodes, src, snk, edges, caps)
        if r is None:
            continue
        phi_s = "[" + ", ".join(f"{p:.4f}" for p in r["phi"]) + "]"
        nuc_s = "[" + ", ".join(f"{p:.4f}" for p in r["nucleolus"]) + "]"
        print(f"  {alpha:>6.2f}  {r['max_flow']:>8.3f}  {r['sigma2']:>8.4f}  "
              f"{r['deficit']:>10.4f}  {'Yes' if r['in_core'] else 'No':>7}  "
              f"{r['l1_dist']:>12.6f}")
        print(f"         phi(Shapley)  = {phi_s}")
        print(f"         eta(Nucleolus)= {nuc_s}")
        print()


def run_unit_vs_weighted_comparison():
    """
    Direct comparison: unit capacities vs capacity-weighted SP-balanced
    for several topology sizes. Shows the phase transition clearly.
    """
    print("\n" + "="*90)
    print("EXPERIMENT D — Unit capacities vs Capacity-weighted: head-to-head")
    print("="*90)

    configs = [
        ("2×2", 2, 2, [0.0, 0.3, 0.6, 0.9]),
        ("2×3", 2, 3, [0.0, 0.2, 0.4, 0.6]),
        ("3×2", 3, 2, [0.0, 0.3, 0.6, 0.9]),
    ]

    print(f"\n  {'Config':>12}  {'α':>6}  {'σ²':>10}  {'Deficit':>10}  "
          f"{'L1(φ,nuc)':>12}  {'InCore':>7}")
    print("  " + "-"*70)

    for (cfg_name, k, m, alphas) in configs:
        for alpha in alphas:
            nodes, src, snk, edges, caps = sp_layer_equal_total(k, m, alpha)
            r = analyse_one(f"SP({cfg_name}) a={alpha:.1f}",
                            nodes, src, snk, edges, caps)
            if r is None:
                continue
            phi_s = "[" + ", ".join(f"{p:.4f}" for p in r["phi"]) + "]"
            nuc_s = "[" + ", ".join(f"{p:.4f}" for p in r["nucleolus"]) + "]"
            print(f"  {cfg_name:>12}  {alpha:>6.2f}  {r['sigma2']:>10.4f}  "
                  f"{r['deficit']:>10.4f}  {r['l1_dist']:>12.6f}  "
                  f"{'Yes' if r['in_core'] else 'No':>7}")
            print(f"  {'':>12}  phi = {phi_s}")
            print(f"  {'':>12}  eta = {nuc_s}")
            print()
        print()


def run_summary_hypothesis_test(sp_results):
    """
    Print a clean hypothesis summary from Experiment A results.
    """
    print("\n" + "="*90)
    print("HYPOTHESIS SUMMARY")
    print("="*90)

    print("\n  H1: Capacity-weighted SP-balanced (equal layer totals) ≠ sufficient for core")
    print("  ─────────────────────────────────────────────────────────────────────────────")
    all_breaks = []
    for cfg_name, rows in sp_results.items():
        first_break = next((alpha for alpha, r in rows
                            if not r["in_core"]), None)
        if first_break is not None:
            all_breaks.append((cfg_name, first_break))
            print(f"    SP({cfg_name}): core BROKEN at α = {first_break:.1f}  → H1 CONFIRMED")
        else:
            print(f"    SP({cfg_name}): core holds for all α tested  → H1 inconclusive")

    print("\n  H2: L1(φ, nucleolus) grows monotonically with α")
    print("  ─────────────────────────────────────────────────────────────────────────────")
    for cfg_name, rows in sp_results.items():
        l1_vals = [r["l1_dist"] for _, r in rows]
        monotone = all(l1_vals[i] <= l1_vals[i+1] + 1e-8
                       for i in range(len(l1_vals)-1))
        status   = "CONFIRMED" if monotone else "VIOLATED"
        print(f"    SP({cfg_name}): L1 sequence = "
              f"{[f'{v:.4f}' for v in l1_vals]}  → H2 {status}")

    print("\n  H3: Increasing bottleneck capacity reduces deficit (Funnel topology)")
    print("  ─────────────────────────────────────────────────────────────────────────────")
    print("    (See Experiment B table above for full data)")


# ═══════════════════════════════════════════════════════════════
# 8.  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Direction 1B — Non-Unit Capacities")
    print("Core stability and Shapley–Nucleolus distance under heterogeneous capacities")

    sp_results = run_sp_imbalance_sweep()
    run_funnel_sweep()
    run_mixed_layer_sweep()
    run_unit_vs_weighted_comparison()
    run_summary_hypothesis_test(sp_results)