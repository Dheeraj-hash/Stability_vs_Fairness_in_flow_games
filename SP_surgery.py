"""
Direction 3 — Graph Surgery for SP-Balancing
==============================================
Question: Can any flow network be converted to SP-balanced form
by iteratively adding bypass (parallel) edges?

Algorithm (Greedy Surgery):
  At each step:
    1. Find the edge e* with the HIGHEST cut-frequency
       (the most "over-burdened" edge — the one most responsible
        for sigma^2 > 0).
    2. Add a direct parallel edge alongside e* (same u, v).
       This costs exactly 1 new edge (player) per step.
    3. Recompute sigma^2, core deficit, L1_norm.
    4. Repeat until sigma^2 = 0 or step budget exhausted.

Key fix vs. previous attempt:
    Parallel edges are stored as duplicate (u,v) tuples in the edge list.
    All flow / Shapley computations use edge INDICES (not frozensets of
    tuples), so parallel edges retain their independent identities.

Metrics tracked per step:
    n         : number of edges (players)
    sigma^2   : cut-frequency variance
    deficit   : max coalition core violation
    L1_norm   : ||phi - eta_sym||_1 / v(N)
    in_core   : deficit < 1e-6
    phi=eta?  : L1_norm < 1e-4
"""

import sys, math
import numpy as np
from fractions import Fraction
from collections import defaultdict, deque

sys.path.insert(0, ".")
from flow_game_analysis import make_topologies, find_all_min_cuts, cut_variance
from shapley_nucleolus_L1 import symmetric_nucleolus


# =========================================================================
# Corrected max-flow using edge INDICES (handles parallel edges correctly)
# =========================================================================

def max_flow_by_index(edge_list, nodes, source, sink, coalition):
    """
    Compute max s-t flow using only the edges whose indices are in coalition.
    Parallel edges (duplicate (u,v) tuples) each contribute +1 capacity.
    """
    cap = defaultdict(int)
    for i in coalition:
        u, v = edge_list[i]
        cap[(u, v)] += 1

    res = defaultdict(int)
    for (u, v), c in cap.items():
        res[(u, v)] += c

    def bfs():
        parent = {source: None}
        queue  = deque([source])
        while queue:
            u = queue.popleft()
            for v in nodes:
                if v not in parent and res[(u, v)] > 0:
                    parent[v] = u
                    if v == sink:
                        return parent
                    queue.append(v)
        return None

    flow = 0
    while True:
        parent = bfs()
        if parent is None:
            break
        pf = float('inf')
        v  = sink
        while v != source:
            u = parent[v]; pf = min(pf, res[(u, v)]); v = u
        v = sink
        while v != source:
            u = parent[v]; res[(u, v)] -= pf; res[(v, u)] += pf; v = u
        flow += pf
    return flow


# =========================================================================
# Corrected characteristic function + Shapley + deficit using indices
# =========================================================================

def char_func(edge_list, nodes, source, sink):
    n = len(edge_list)
    v = {}
    for mask in range(1 << n):
        S    = frozenset(i for i in range(n) if mask & (1 << i))
        flow = max_flow_by_index(edge_list, nodes, source, sink, S)
        v[S] = Fraction(flow)
    return v


def shapley(n, v):
    phi    = [Fraction(0)] * n
    n_fact = math.factorial(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            S = frozenset(j for j in range(n) if mask & (1 << j))
            s = len(S)
            w = Fraction(math.factorial(s) * math.factorial(n - s - 1), n_fact)
            phi[i] += w * (v[S | {i}] - v[S])
    return phi


def deficit(n, v, phi):
    worst = Fraction(0)
    for mask in range(1, 1 << n):
        S = frozenset(i for i in range(n) if mask & (1 << i))
        ex = v[S] - sum(phi[i] for i in S)
        if ex > worst:
            worst = ex
    return worst


# =========================================================================
# Full metrics for one graph state
# =========================================================================

def full_metrics(edge_list, nodes, source, sink):
    n        = len(edge_list)
    caps     = [1] * n
    v        = char_func(edge_list, nodes, source, sink)
    phi_frac = shapley(n, v)
    phi      = np.array([float(p) for p in phi_frac])
    v_N      = float(v[frozenset(range(n))])
    def_     = float(deficit(n, v, phi_frac))
    in_core  = def_ < 1e-6

    min_cuts, _ = find_all_min_cuts(edge_list, nodes, source, sink)
    sigma2, n_cuts, freqs = cut_variance(min_cuts, edge_list)
    sigma2 = float(sigma2)

    eta_sym, vertices, spread = symmetric_nucleolus(edge_list, caps, min_cuts)
    l1_sym  = float(np.sum(np.abs(phi - eta_sym)))
    l1_norm = l1_sym / v_N if v_N > 1e-9 else 0.0

    return dict(n=n, v_N=v_N, phi=phi, phi_frac=phi_frac,
                eta_sym=eta_sym, deficit=def_, in_core=in_core,
                sigma2=sigma2, n_cuts=n_cuts, freqs=freqs,
                l1_sym=l1_sym, l1_norm=l1_norm, spread=spread)


# =========================================================================
# Graph Surgery
# =========================================================================

def graph_surgery(name, nodes, source, sink, edge_list, max_steps=20, n_limit=15):
    """
    Iteratively add a parallel edge alongside the highest-frequency edge.
    Returns list of state dicts (one per step, step 0 = original).
    """
    current_edges = list(edge_list)
    history       = []

    print(f"\n{'='*72}")
    print(f"  Surgery: {name}")
    print(f"{'='*72}")
    print(f"  {'Step':>5}  {'n':>4}  {'v(N)':>6}  {'sigma2':>8}  "
          f"{'deficit':>8}  {'L1_norm':>8}  {'Core':>5}  {'phi=eta?':>8}  Added")
    print("  " + "-"*85)

    for step in range(max_steps + 1):
        if len(current_edges) > n_limit:
            print(f"  {'':>5}  [stopped: n={len(current_edges)} > {n_limit}]")
            break

        m = full_metrics(current_edges, nodes, source, sink)
        m["step"]      = step
        m["edge_list"] = list(current_edges)
        history.append(m)

        added = "—" if step == 0 else f"parallel copy of edge {bypass_idx} {bypass_edge}"
        phi_s = "[" + ", ".join(f"{x:.3f}" for x in m["phi"]) + "]"
        eta_s = "[" + ", ".join(f"{x:.3f}" for x in m["eta_sym"]) + "]"

        print(f"  {step:>5}  {m['n']:>4}  {m['v_N']:>6.2f}  "
              f"{m['sigma2']:>8.4f}  {m['deficit']:>8.4f}  "
              f"{m['l1_norm']:>8.4f}  {'Yes' if m['in_core'] else 'No':>5}  "
              f"{'Yes' if m['l1_norm']<1e-4 else 'No':>8}  {added}")
        print(f"  {'':>5}  phi    = {phi_s}")
        print(f"  {'':>5}  eta_s  = {eta_s}")
        print(f"  {'':>5}  freqs  = {m['freqs']}")
        print()

        if m["sigma2"] < 1e-6:
            print(f"  -> sigma^2 = 0 reached at step {step}. Surgery complete.")
            break

        if step == max_steps:
            print(f"  -> Budget exhausted ({max_steps} steps).")
            break

        # choose: add parallel copy of highest-frequency edge
        freqs      = m["freqs"]
        bypass_idx = int(np.argmax(freqs))
        bypass_edge = current_edges[bypass_idx]
        current_edges.append(bypass_edge)   # direct duplicate — no extra node needed

    return history


# =========================================================================
# Summary + analysis
# =========================================================================

def print_summary(all_histories):
    print(f"\n{'='*90}")
    print("SURGERY SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Topology':<26} {'n_orig':>7} {'sigma2_0':>9} {'deficit_0':>10} "
          f"{'steps':>6} {'n_added':>8} {'sigma2_f':>9} {'deficit_f':>10} "
          f"{'core_f':>7} {'phi=eta_f':>10}")
    print("  " + "-"*100)

    for name, history in all_histories.items():
        init  = history[0]
        final = history[-1]
        converged  = final["sigma2"] < 1e-6
        steps_s    = str(final["step"]) if converged else f">{final['step']}"
        n_added    = final["n"] - init["n"]
        phi_eq_eta = final["l1_norm"] < 1e-4

        print(f"  {name:<26} {init['n']:>7} {init['sigma2']:>9.4f} "
              f"{init['deficit']:>10.4f} {steps_s:>6} {n_added:>8} "
              f"{final['sigma2']:>9.4f} {final['deficit']:>10.4f} "
              f"{'Yes' if final['in_core'] else 'No':>7} "
              f"{'Yes' if phi_eq_eta else 'No':>10}")

    print(f"\n  KEY OBSERVATIONS:")
    print(f"  -----------------")
    for name, history in all_histories.items():
        init  = history[0]
        final = history[-1]

        # find first step where deficit drops below initial
        improvement_steps = [(s["step"], s["deficit"])
                             for s in history if s["deficit"] < init["deficit"] - 1e-6]
        first_improvement = improvement_steps[0] if improvement_steps else None

        # check monotonicity of deficit
        deficits = [s["deficit"] for s in history]
        monotone = all(deficits[i] <= deficits[i-1] + 1e-6 for i in range(1, len(deficits)))

        print(f"\n  {name}:")
        print(f"    Initial: sigma2={init['sigma2']:.4f}  deficit={init['deficit']:.4f}")
        print(f"    Final:   sigma2={final['sigma2']:.4f}  deficit={final['deficit']:.4f}  "
              f"after {final['n']-init['n']} added edges")
        print(f"    Deficit monotone decreasing: {monotone}")
        if first_improvement:
            print(f"    First deficit improvement at step {first_improvement[0]} "
                  f"(deficit={first_improvement[1]:.4f})")

        # deficit trajectory
        traj = " -> ".join(f"{s['deficit']:.3f}" for s in history)
        print(f"    Deficit trajectory: {traj}")
        l1_traj = " -> ".join(f"{s['l1_norm']:.3f}" for s in history)
        print(f"    L1_norm trajectory: {l1_traj}")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    print("Direction 3 — Graph Surgery for SP-Balancing")
    print("Parallel bypass edges added to highest-frequency edge each step")

    topos = {name: (nodes, src, snk, edges)
             for name, nodes, src, snk, edges in make_topologies()}

    targets = [
        "Funnel(2+1)",
        "Funnel(3+1)",
        "Double-Bottleneck",
        "Wheatstone-Bridge",
        "Ladder-2x2",
    ]

    all_histories = {}
    for name in targets:
        if name not in topos:
            print(f"  [skip: {name} not found]")
            continue
        nodes, src, snk, edges = topos[name]
        history = graph_surgery(name, nodes, src, snk, edges,
                                max_steps=12, n_limit=14)
        all_histories[name] = history

    print_summary(all_histories)