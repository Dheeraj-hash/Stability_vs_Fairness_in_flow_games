"""
Fair Allocation in Flow Games — Direction 1A
=============================================
Investigates whether core deficit is monotonically non-decreasing in
sigma^2(G) = variance of minimum cut capacities across all s-t min-cuts.

Conjecture:  deficit = 0  iff  sigma^2 = 0,
             and deficit is non-decreasing in sigma^2.

Pipeline per topology:
  1. Build directed graph with unit capacities.
  2. Enumerate all 2^n subsets -> compute max-flow (Edmonds-Karp).
  3. Compute exact Shapley values (fractions.Fraction arithmetic).
  4. Find all minimum cuts -> compute sigma^2 over their capacities.
  5. Compute core deficit = max over all S of (v(S) - sum phi_i for i in S).
  6. Aggregate results and test monotonicity conjecture.
"""

from fractions import Fraction
from itertools import permutations, combinations
from collections import defaultdict, deque
import math
import statistics


# ---------------------------------------------------------------------------
# 1. Max-flow: Edmonds-Karp (BFS-based Ford-Fulkerson)
# ---------------------------------------------------------------------------

def edmonds_karp(graph, source, sink, nodes):
    """
    graph: dict[u] -> dict[v] -> int  (residual capacity, modified in place)
    Returns max flow value.
    """
    # deep copy so we don't mutate caller's graph
    residual = {u: dict(graph[u]) for u in nodes}
    for u in nodes:
        for v in nodes:
            if v not in residual[u]:
                residual[u][v] = 0

    def bfs():
        parent = {source: None}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in nodes:
                if v not in parent and residual[u][v] > 0:
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
        # find bottleneck
        path_flow = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual[u][v])
            v = u
        # augment
        v = sink
        while v != source:
            u = parent[v]
            residual[u][v] -= path_flow
            residual[v][u] += path_flow
            v = u
        flow += path_flow
    return flow


def max_flow_subset(base_graph, nodes, source, sink, subset_edges):
    """
    Build a graph containing only the edges in subset_edges,
    then compute max flow.
    subset_edges: frozenset of (u,v) edge tuples present in coalition.
    """
    graph = {u: {v: 0 for v in nodes} for u in nodes}
    for (u, v) in subset_edges:
        graph[u][v] += 1   # unit capacity per edge; handles multi-edges
    return edmonds_karp(graph, source, sink, nodes)


# ---------------------------------------------------------------------------
# 2. Characteristic function + Shapley value
# ---------------------------------------------------------------------------

def compute_characteristic_function(edge_list, nodes, source, sink):
    """
    Returns dict: frozenset(coalition indices) -> Fraction(max_flow).
    Coalition index i corresponds to edge_list[i].
    """
    n = len(edge_list)
    v = {}
    for mask in range(1 << n):
        coalition = frozenset(i for i in range(n) if mask & (1 << i))
        edges_in = frozenset(edge_list[i] for i in coalition)
        flow = max_flow_subset(None, nodes, source, sink, edges_in)
        v[coalition] = Fraction(flow)
    return v


def shapley_value(n, v):
    """
    Exact Shapley value using the coalition-sum formula (avoids n! iteration):
      phi_i = sum_{S not containing i} [ |S|!(n-|S|-1)!/n! * (v(S|i) - v(S)) ]
    Uses Fraction arithmetic for exactness.
    """
    phi = [Fraction(0)] * n
    n_fact = math.factorial(n)
    players = list(range(n))

    for i in players:
        for mask in range(1 << n):
            if mask & (1 << i):
                continue   # S must not contain i
            S = frozenset(j for j in range(n) if mask & (1 << j))
            s = len(S)
            weight = Fraction(math.factorial(s) * math.factorial(n - s - 1), n_fact)
            marginal = v[S | {i}] - v[S]
            phi[i] += weight * marginal

    return phi


# ---------------------------------------------------------------------------
# 3. Core deficit
# ---------------------------------------------------------------------------

def core_deficit(n, v, phi):
    """
    deficit = max over all S of (v(S) - sum_{i in S} phi_i)
    Returns Fraction.
    """
    worst = Fraction(0)
    for mask in range(1, 1 << n):
        S = frozenset(i for i in range(n) if mask & (1 << i))
        excess = v[S] - sum(phi[i] for i in S)
        if excess > worst:
            worst = excess
    return worst


# ---------------------------------------------------------------------------
# 4. Minimum cut enumeration + sigma^2
# ---------------------------------------------------------------------------

def find_all_min_cuts(edge_list, nodes, source, sink):
    """
    Enumerate all minimum (s,t)-cuts via the following approach:
      - Compute max flow value F.
      - A cut (S_set, T_set) is a min-cut iff its capacity = F.
      - We enumerate all 2^|nodes| partitions (S_set containing source,
        T_set containing sink) and collect those with capacity = F.
    Returns list of (S_set, T_set, capacity) for all min-cuts.
    """
    # Full graph
    full_edges = frozenset(edge_list)
    F = max_flow_subset(None, nodes, source, sink, full_edges)

    non_st = [nd for nd in nodes if nd != source and nd != sink]
    min_cuts = []

    # Enumerate all subsets of non-source/sink nodes to be on source side
    for mask in range(1 << len(non_st)):
        S_set = {source}
        for idx, nd in enumerate(non_st):
            if mask & (1 << idx):
                S_set.add(nd)
        T_set = set(nodes) - S_set

        if sink not in T_set:
            continue  # sink must be in T

        # capacity = edges from S_set to T_set
        cap = sum(1 for (u, v) in edge_list if u in S_set and v in T_set)
        if cap == F:
            min_cuts.append((frozenset(S_set), frozenset(T_set), cap))

    return min_cuts, F


def cut_variance(min_cuts, edge_list):
    """
    Computes sigma^2 as the VARIANCE OF PER-EDGE MIN-CUT FREQUENCIES.

    For each edge e, define freq(e) = number of min-cuts that e crosses
    (i.e., tail in S-side, head in T-side).

    sigma^2 = Var(freq(e)) across all edges.

    Interpretation:
      - High sigma^2: some edges appear in many min-cuts (central/bottleneck),
        others in few. The Shapley value over-rewards rare pivots.
      - sigma^2 = 0 and all freqs equal: every edge has equal cut-coverage.
        This is the SP-balanced condition — no bottleneck tax.
      - sigma^2 = 0 with freq=0 for all: no edges cross any min-cut
        (degenerate / disconnected). Not possible for connected graphs.

    Also returns per-edge frequencies and n_cuts.
    """
    n = len(edge_list)
    if not min_cuts:
        return Fraction(0), 0, [0] * n

    n_cuts = len(min_cuts)
    freq = []
    for e_idx, (u, v) in enumerate(edge_list):
        count = sum(1 for (S, T, cap) in min_cuts if u in S and v in T)
        freq.append(count)

    if n == 0:
        return Fraction(0), n_cuts, freq

    mean = Fraction(sum(freq), n)
    variance = sum((Fraction(f) - mean) ** 2 for f in freq) / n
    return variance, n_cuts, freq


# ---------------------------------------------------------------------------
# 5. Topology definitions
# ---------------------------------------------------------------------------

def make_topologies():
    """
    Returns list of (name, nodes, source, sink, edge_list).
    All unit-capacity edges.
    """
    topos = []

    # --- Simple path graphs ---
    for k in range(2, 6):
        nodes = list(range(k + 1))   # 0 ... k
        edges = [(i, i+1) for i in range(k)]
        topos.append((f"Path-{k}", nodes, 0, k, edges))

    # --- Parallel graphs ---
    # s -> mid_i -> t : each path is 2 edges, n=2k total
    # Keep k small to stay under n=12
    for k in range(2, 5):
        nodes = [0] + [10 + i for i in range(k)] + [1]
        edges = [(0, 10 + i) for i in range(k)] + [(10 + i, 1) for i in range(k)]
        topos.append((f"Parallel-{k}", nodes, 0, 1, edges))

    # --- SP-balanced: k layers of m parallel edges each ---
    # Structure: s -> m parallel paths, each passing through k intermediate
    # nodes (one per layer), then -> t.
    # Each "layer" consists of m parallel edges, each on a distinct path.
    # n = k*m edges total. Every s-t path uses exactly one edge per layer.
    sp_configs = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4)]
    for (k, m) in sp_configs:
        if k * m > 12:
            continue   # keep n manageable for exact Shapley
        # m parallel paths, each with k edges:
        # path j: s -> L1_j -> L2_j -> ... -> Lk_j -> t
        nodes_set = {0, 99}
        edges = []
        for j in range(m):
            prev = 0   # source
            for layer in range(k):
                cur = 100 * (layer + 1) + j
                nodes_set.add(cur)
                edges.append((prev, cur))
                prev = cur
            edges.append((prev, 99))  # last node -> sink

        label = f"SP-Balanced({k}x{m})"
        topos.append((label, sorted(nodes_set), 0, 99, edges))

    # --- Funnel: k parallel -> 1 bottleneck ---
    for k in range(2, 5):
        nodes = [0] + [10 + i for i in range(k)] + [50, 1]
        edges = [(0, 10 + i) for i in range(k)] + \
                [(10 + i, 50) for i in range(k)] + \
                [(50, 1)]
        topos.append((f"Funnel({k}+1)", nodes, 0, 1, edges))

    # --- Double bottleneck: parallel -> bottleneck -> parallel -> bottleneck ---
    nodes = [0, 10, 11, 50, 20, 21, 60, 1]
    edges = [(0,10),(0,11),(10,50),(11,50),(50,20),(50,21),(20,60),(21,60),(60,1)]
    topos.append(("Double-Bottleneck", nodes, 0, 1, edges))

    # --- FanIn: k sources -> single sink node -> t ---
    for k in [3, 4]:
        nodes = [0] + [10 + i for i in range(k)] + [50, 1]
        edges = [(0, 10 + i) for i in range(k)] + \
                [(10 + i, 50) for i in range(k)] + [(50, 1)]
        topos.append((f"FanIn({k}->1)", nodes, 0, 1, edges))

    # --- Wheatstone bridge ---
    # s->a, s->b, a->c, b->c, a->t, b->t, c->t
    nodes = [0, 1, 2, 3, 4]   # s=0, a=1, b=2, c=3, t=4
    edges = [(0,1),(0,2),(1,3),(2,3),(1,4),(2,4),(3,4)]
    topos.append(("Wheatstone-Bridge", nodes, 0, 4, edges))

    # --- Asymmetric series-parallel (unequal layers) ---
    # s -> [2 parallel] -> [3 parallel] -> t  (7 edges: 2+6+... too many)
    # Use simpler: s -> a -> c, s -> b -> c, s -> a -> d, s -> b -> d, c->t, d->t
    # i.e. 2-layer SP but layer sizes 2 and 2 — actually balanced, skip
    # Instead: funnel-like but with 2 exit edges
    # s -> a, s -> b, s -> c  (3 edges into middle)
    # a -> t, b -> t           (2 edges out) => 5 edges total, n=5
    nodes = [0, 10, 11, 12, 1]
    edges = [(0,10),(0,11),(0,12),(10,1),(11,1)]
    topos.append(("Asym-FanIn(3in-2out)", nodes, 0, 1, edges))

    # --- Unbalanced SP: 2 parallel -> 3 parallel -> 2 parallel (n=2+6+6+2=... too big) ---
    # Simpler: 2-layer SP with 2 paths on left, 3 on right, sharing nothing -> 2+3=5 edges
    # path structure: s->La1->t, s->La2->t, s->Lb1->t, s->Lb2->t, s->Lb3->t
    # but that's just parallel — boring. Make proper 2-layer unbalanced:
    # Layer A: 2 edges (a1, a2), Layer B: 3 edges (b1,b2,b3)
    # Each s-t path uses one A-edge and one B-edge
    # s->a1, s->a2, a1->b1, a1->b2, a1->b3, a2->b1, a2->b2, a2->b3, b1->t, b2->t, b3->t
    # n = 2 + 6 + 3 = 11 edges
    nodes = [0, 10, 11, 20, 21, 22, 1]
    edges  = [(0,10),(0,11)]
    edges += [(10,20),(10,21),(10,22),(11,20),(11,21),(11,22)]
    edges += [(20,1),(21,1),(22,1)]
    topos.append(("Unbalanced-SP(2x3)", nodes, 0, 1, edges))

    # --- Diamond ---
    nodes = [0,1,2,3]
    edges = [(0,1),(0,2),(1,3),(2,3)]
    topos.append(("Diamond", nodes, 0, 3, edges))

    # --- Ladder graph (2 x k) ---
    for k in [2, 3]:
        top = [100 + i for i in range(k+1)]
        bot = [200 + i for i in range(k+1)]
        nodes = [0, 99] + top + bot
        edges = []
        edges.append((0, top[0]))
        edges.append((0, bot[0]))
        for i in range(k):
            edges.append((top[i], top[i+1]))
            edges.append((bot[i], bot[i+1]))
            edges.append((top[i], bot[i]))     # rungs
        edges.append((top[-1], 99))
        edges.append((bot[-1], 99))
        n_edges = len(edges)
        if n_edges <= 12:
            topos.append((f"Ladder-2x{k}", nodes, 0, 99, edges))

    return topos


# ---------------------------------------------------------------------------
# 6. Main analysis loop
# ---------------------------------------------------------------------------

def analyze_topology(name, nodes, source, sink, edge_list):
    n = len(edge_list)
    if n > 16:
        return None  # skip — 2^n enumeration too expensive

    v = compute_characteristic_function(edge_list, nodes, source, sink)
    phi = shapley_value(n, v)
    deficit = core_deficit(n, v, phi)
    min_cuts, max_flow_val = find_all_min_cuts(edge_list, nodes, source, sink)
    variance, n_cuts, cut_sizes = cut_variance(min_cuts, edge_list)

    # Convexity check: sample marginal contributions
    # v(S u {i}) - v(S) <= v(T u {i}) - v(T) for all S ⊆ T ⊆ N\{i}
    is_convex = True
    players = list(range(n))
    for i in players:
        rest = [p for p in players if p != i]
        for r in range(len(rest) + 1):
            for S_list in combinations(rest, r):
                S = frozenset(S_list)
                for extra in combinations([p for p in rest if p not in S], 1):
                    T = S | frozenset(extra)
                    marg_S = v[S | {i}] - v[S]
                    marg_T = v[T | {i}] - v[T]
                    if marg_S > marg_T:
                        is_convex = False
                        break
                if not is_convex:
                    break
            if not is_convex:
                break
        if not is_convex:
            break

    return {
        "name": name,
        "n": n,
        "max_flow": max_flow_val,
        "phi": phi,
        "deficit": deficit,
        "in_core": deficit == 0,
        "is_convex": is_convex,
        "n_min_cuts": n_cuts,
        "edge_cut_freqs": cut_sizes,   # per-edge cut frequencies
        "sigma2": variance,
        "sigma2_float": float(variance),
        "deficit_float": float(deficit),
    }


# ---------------------------------------------------------------------------
# 7. Monotonicity test
# ---------------------------------------------------------------------------

def test_monotonicity(results):
    """
    Test the conjecture: deficit is non-decreasing in sigma^2.
    Method: for all pairs (i,j), check if sigma2_i < sigma2_j => deficit_i <= deficit_j.
    Count violations.
    """
    pairs_checked = 0
    violations = []

    for i in range(len(results)):
        for j in range(len(results)):
            if i == j:
                continue
            ri, rj = results[i], results[j]
            if ri["sigma2"] < rj["sigma2"]:
                pairs_checked += 1
                if ri["deficit"] > rj["deficit"]:
                    violations.append((ri["name"], rj["name"],
                                       ri["sigma2_float"], rj["sigma2_float"],
                                       ri["deficit_float"], rj["deficit_float"]))

    return pairs_checked, violations


# ---------------------------------------------------------------------------
# 8. Pretty output
# ---------------------------------------------------------------------------

def print_results(results):
    print("\n" + "="*120)
    print(f"{'Topology':<28} {'n':>3} {'MaxFlow':>7} {'Convex':>6} {'InCore':>6} "
          f"{'n_cuts':>6} {'σ²(edge-freq)':>14} {'Deficit':>10}")
    print("="*120)

    for r in sorted(results, key=lambda x: x["sigma2_float"]):
        conv  = "Yes" if r["is_convex"] else "No"
        core  = "Yes" if r["in_core"]   else "No"
        sigma2_str  = f"{r['sigma2_float']:.4f}"
        deficit_str = f"{r['deficit_float']:.4f}"
        print(f"{r['name']:<28} {r['n']:>3} {r['max_flow']:>7} {conv:>6} {core:>6} "
              f"{r['n_min_cuts']:>6} {sigma2_str:>14} {deficit_str:>10}")

    print("="*120)

    print("\n--- Shapley values (exact fractions) ---")
    for r in sorted(results, key=lambda x: x["sigma2_float"]):
        phi_str = ", ".join(str(p) for p in r["phi"])
        print(f"  {r['name']:<28}  φ = [{phi_str}]")

    print("\n--- Per-edge cut frequencies (how many min-cuts each edge crosses) ---")
    for r in sorted(results, key=lambda x: x["sigma2_float"]):
        freq_str = str(r["edge_cut_freqs"])
        print(f"  {r['name']:<28}  freqs={freq_str}  σ²={r['sigma2_float']:.4f}")

    print("\n--- Monotonicity conjecture: deficit non-decreasing in σ²(edge-freq) ---")
    pairs, violations = test_monotonicity(results)
    print(f"  Pairs checked (σ²_i < σ²_j): {pairs}")
    print(f"  Monotonicity violations:      {len(violations)}")
    if violations:
        print("  VIOLATIONS FOUND:")
        for (n1, n2, s1, s2, d1, d2) in violations[:20]:  # cap at 20
            print(f"    {n1} (σ²={s1:.4f}, deficit={d1:.4f}) > {n2} (σ²={s2:.4f}, deficit={d2:.4f})")
        if len(violations) > 20:
            print(f"    ... and {len(violations)-20} more")
    else:
        print("  Conjecture HOLDS across all topology pairs. No violations found.")

    print("\n--- Core stability vs σ²(edge-freq) (sorted by σ²) ---")
    print(f"  {'σ²':>10}  {'deficit':>10}  {'in_core':>8}  topology")
    for r in sorted(results, key=lambda x: x["sigma2_float"]):
        print(f"  {r['sigma2_float']:>10.4f}  {r['deficit_float']:>10.4f}  "
              f"{'Yes' if r['in_core'] else 'No':>8}  {r['name']}")

    # Pearson correlation between sigma^2 and deficit
    s2_vals = [r["sigma2_float"] for r in results]
    def_vals = [r["deficit_float"] for r in results]
    if len(set(s2_vals)) > 1 and len(set(def_vals)) > 1:
        mean_s = sum(s2_vals) / len(s2_vals)
        mean_d = sum(def_vals) / len(def_vals)
        num = sum((s - mean_s) * (d - mean_d) for s, d in zip(s2_vals, def_vals))
        denom = math.sqrt(sum((s - mean_s)**2 for s in s2_vals) *
                          sum((d - mean_d)**2 for d in def_vals))
        corr = num / denom if denom > 0 else 0
        print(f"\n  Pearson correlation(σ²(edge-freq), deficit) = {corr:.4f}")

    print("\n--- Key findings ---")
    zero_s2_nonzero_def = [r for r in results if r["sigma2"] == 0 and r["deficit"] > 0]
    nonzero_s2_zero_def = [r for r in results if r["sigma2"] > 0 and r["deficit"] == 0]
    print(f"  σ²=0 but deficit>0 (conjecture violation candidates): {len(zero_s2_nonzero_def)}")
    for r in zero_s2_nonzero_def:
        print(f"    -> {r['name']}  (deficit={r['deficit_float']:.4f}, freqs={r['edge_cut_freqs']})")
    print(f"  σ²>0 but deficit=0 (non-convex/non-trivial but in core): {len(nonzero_s2_zero_def)}")
    for r in nonzero_s2_zero_def:
        print(f"    -> {r['name']}  (σ²={r['sigma2_float']:.4f})")


# ---------------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    topologies = make_topologies()
    results = []
    print("Analyzing topologies...")
    for (name, nodes, source, sink, edges) in topologies:
        print(f"  Processing: {name}  (n={len(edges)} edges)", end="", flush=True)
        result = analyze_topology(name, nodes, source, sink, edges)
        if result is None:
            print(f"  [SKIPPED — n too large]")
            continue
        results.append(result)
        print(f"  -> σ²={result['sigma2_float']:.4f}  deficit={result['deficit_float']:.4f}")

    print_results(results)