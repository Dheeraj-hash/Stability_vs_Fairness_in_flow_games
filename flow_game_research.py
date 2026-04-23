#!/usr/bin/env python3
"""
Flow Games: Shapley Value and Core Compatibility
"""

import math
import itertools
from collections import defaultdict, deque
from fractions import Fraction
import json

# ════════════════════════════════════════════════════════════════════
# I.  MAX FLOW  (Edmonds-Karp)
# ════════════════════════════════════════════════════════════════════

def edmonds_karp(adj, s, t):
    """BFS-augmenting max flow on adjacency dict.  adj is modified in-place."""
    total = 0
    while True:
        par = {s: None}
        q = deque([s])
        while q and t not in par:
            u = q.popleft()
            for v, c in adj[u].items():
                if v not in par and c > 0:
                    par[v] = u
                    q.append(v)
        if t not in par:
            break
        # bottleneck
        bot, v = float('inf'), t
        while par[v] is not None:
            u = par[v]; bot = min(bot, adj[u][v]); v = u
        # augment residual
        v = t
        while par[v] is not None:
            u = par[v]; adj[u][v] -= bot; adj[v][u] += bot; v = u
        total += bot
    return total


def coalition_flow(edges, caps, coalition, s, t):
    adj = defaultdict(lambda: defaultdict(int))
    for i in coalition:
        u, v = edges[i]
        adj[u][v] += caps[i]
    return edmonds_karp(adj, s, t)


# ════════════════════════════════════════════════════════════════════
# II.  FLOW GAME
# ════════════════════════════════════════════════════════════════════

class FlowGame:
    """
    n players each own one directed edge.
    v(S) = max s-t flow through edges owned by S.
    """

    def __init__(self, edges, caps, s='s', t='t', name=''):
        self.edges = edges          # list of (u, v)
        self.caps  = caps           # list of int
        self.s, self.t = s, t
        self.n = len(edges)
        self.name = name
        self._cache: dict = {}

    # ── characteristic function ──────────────────────────────────────

    def v(self, S: frozenset) -> int:
        if S not in self._cache:
            self._cache[S] = coalition_flow(self.edges, self.caps, S, self.s, self.t)
        return self._cache[S]

    def grand_value(self):
        return self.v(frozenset(range(self.n)))

    def all_values(self):
        return {frozenset(i for i in range(self.n) if mask >> i & 1): None
                for mask in range(1 << self.n)}   # lazy shell

    # ── Shapley value  (exact rational arithmetic) ───────────────────

    def shapley(self):
        n = self.n
        fac = [math.factorial(k) for k in range(n + 1)]
        phi = [Fraction(0)] * n
        rest = {i: [j for j in range(n) if j != i] for i in range(n)}
        for i in range(n):
            for r in range(n):
                for T in itertools.combinations(rest[i], r):
                    S = frozenset(T)
                    mc = self.v(S | {i}) - self.v(S)
                    if mc:
                        phi[i] += Fraction(fac[r] * fac[n - r - 1], fac[n]) * mc
        return phi

    # ── core membership check ────────────────────────────────────────

    def is_in_core(self, alloc):
        """alloc: list of Fraction.  Returns (bool, violating_S_or_None, deficit)."""
        worst_S, worst_deficit = None, Fraction(0)
        for mask in range(1, 1 << self.n):
            S = frozenset(i for i in range(self.n) if mask >> i & 1)
            deficit = Fraction(self.v(S)) - sum(alloc[i] for i in S)
            if deficit > worst_deficit:
                worst_S, worst_deficit = S, deficit
        in_core = worst_deficit == 0
        return in_core, worst_S, worst_deficit

    # ── structural properties ────────────────────────────────────────

    def is_convex(self):
        """Supermodularity: v(S∪T)+v(S∩T) ≥ v(S)+v(T) for all S,T."""
        sets = [frozenset(i for i in range(self.n) if mask >> i & 1)
                for mask in range(1 << self.n)]
        for S in sets:
            for T in sets:
                if self.v(S | T) + self.v(S & T) < self.v(S) + self.v(T):
                    return False
        return True

    def edges_on_every_mincut(self):
        """e is on every min cut ⟺ removing e strictly decreases max flow."""
        val = self.grand_value()
        N = frozenset(range(self.n))
        return frozenset(i for i in range(self.n) if self.v(N - {i}) < val)

    def min_cut_edges(self):
        """Find one minimum (s,t)-cut using BFS on residual of full flow."""
        # Run max-flow and inspect residual reachability
        adj = defaultdict(lambda: defaultdict(int))
        for i in range(self.n):
            u, v = self.edges[i]
            adj[u][v] += self.caps[i]
        edmonds_karp(adj, self.s, self.t)
        # Nodes reachable from s in residual
        visited = {self.s}
        q = deque([self.s])
        while q:
            u = q.popleft()
            for v, c in adj[u].items():
                if v not in visited and c > 0:
                    visited.add(v); q.append(v)
        # Cut edges: original edges from visited to non-visited
        cut = frozenset(i for i, (u, v) in enumerate(self.edges)
                        if u in visited and v not in visited)
        return cut

    def count_min_cuts(self):
        """Count distinct minimum (s,t)-cuts.  Exponential — only for small n."""
        val = self.grand_value()
        N = frozenset(range(self.n))
        count = 0
        for mask in range(1 << self.n):
            S = frozenset(i for i in range(self.n) if mask >> i & 1)
            complement = N - S
            # S is a min-cut iff removing S reduces flow below val and |S| is minimum
            pass
        # Simpler: enumerate all node-partitions (s-side, t-side)
        # and check if the resulting edge-cut is minimum
        nodes = set()
        for u, v in self.edges:
            nodes.add(u); nodes.add(v)
        nodes = list(nodes - {self.s, self.t})
        cut_sets = set()
        for mask in range(1 << len(nodes)):
            s_side = {self.s} | {nodes[i] for i in range(len(nodes)) if mask >> i & 1}
            edge_cut = frozenset(i for i, (u, v) in enumerate(self.edges)
                                 if u in s_side and v not in s_side)
            cap = sum(self.caps[i] for i in edge_cut)
            if cap == val:
                cut_sets.add(edge_cut)
        return len(cut_sets), cut_sets

    def unique_min_cut(self):
        cnt, cuts = self.count_min_cuts()
        return cnt == 1, cuts

    def analyze(self):
        """Full analysis: Shapley, core check, structural properties."""
        phi = self.shapley()
        in_core, bad_S, deficit = self.is_in_core(phi)
        convex = self.is_convex()
        on_every = self.edges_on_every_mincut()
        unique, cuts = self.unique_min_cut()

        return {
            'name':      self.name,
            'n':         self.n,
            'edges':     self.edges,
            'caps':      self.caps,
            'grand':     self.grand_value(),
            'shapley':   [str(x) for x in phi],
            'shapley_f': [float(x) for x in phi],
            'in_core':   in_core,
            'deficit':   float(deficit),
            'violating': sorted(bad_S) if bad_S else None,
            'convex':    convex,
            'on_every_mincut': sorted(on_every),
            'n_mincuts': len(cuts),
            'unique_mincut': unique,
            'mincut_edges': [sorted(c) for c in cuts],
        }


# ════════════════════════════════════════════════════════════════════
# III.  NETWORK FAMILIES
# ════════════════════════════════════════════════════════════════════

def make_path(k):
    nodes = ['s'] + [f'v{i}' for i in range(1, k)] + ['t']
    edges = [(nodes[i], nodes[i+1]) for i in range(k)]
    return FlowGame(edges, [1]*k, name=f'Path-{k}')


def make_parallel(k):
    edges = [('s', 't') for _ in range(k)]
    return FlowGame(edges, [1]*k, name=f'Parallel-{k}')


def make_diamond():
    edges = [('s','a'), ('s','b'), ('a','t'), ('b','t')]
    return FlowGame(edges, [1]*4, name='Diamond')


def make_diamond_with_bridge():
    edges = [('s','a'), ('s','b'), ('a','b'), ('a','t'), ('b','t')]
    return FlowGame(edges, [1]*5, name='Diamond+Bridge(a→b)')


def make_series_parallel_balanced():
    edges = [('s','a'), ('a','t'), ('s','b'), ('b','t')]
    return FlowGame(edges, [1]*4, name='SP-Balanced(2paths×2)')


def make_three_paths():
    edges = [('s','a'),('a','t'), ('s','b'),('b','t'), ('s','c'),('c','t')]
    return FlowGame(edges, [1]*6, name='ThreePaths(3×2)')


def make_wheatstone():
    edges = [('s','a'),('s','b'),('a','b'),('a','t'),('b','t')]
    return FlowGame(edges, [1]*5, name='Wheatstone')


def make_grid_2x2():
    edges = [('s','a'),('s','c'),('a','b'),('a','c'),('b','t'),('c','t')]
    return FlowGame(edges, [1]*6, name='Grid-2x2(6e)')


def make_bottleneck():
    edges = [('s','a'),('s','b'),('a','m'),('b','m'),('m','t')]
    return FlowGame(edges, [1,1,1,1,2], name='Bottleneck(funnel)')


def make_two_bottlenecks():
    edges = [('s','a'),('s','b'),('a','m1'),('b','m1'),
             ('m1','m2'),('m2','c'),('m2','d'),('c','t'),('d','t')]
    return FlowGame(edges, [1]*9, name='DoubleBottleneck')


def make_asymmetric_caps():
    edges = [('s','a'),('s','b'),('a','t'),('b','t')]
    caps  = [2, 1, 1, 1]
    return FlowGame(edges, caps, name='Diamond-AsymCap(2,1,1,1)')


def make_triangle():
    edges = [('s','a'),('a','t'),('s','t')]
    return FlowGame(edges, [1]*3, name='Triangle')


def make_redundant_path():
    edges = [('s','a'),('a','b'),('b','t'),('s','t')]
    return FlowGame(edges, [1]*4, name='PathPlusShortcut')


def make_fan_in():
    edges = [('s','a'),('s','b'),('s','c'),('a','m'),('b','m'),('c','m'),('m','t')]
    return FlowGame(edges, [1]*7, name='FanIn(3→1)')


def make_k4_network():
    nodes = ['s','a','b','t']
    edges = [(u,v) for i,u in enumerate(nodes)
             for j,v in enumerate(nodes) if i < j]
    return FlowGame(edges, [1]*len(edges), name='K4-dag')


def make_series_of_parallel(k=2, m=2):
    stages = [f'v{s}' for s in range(k+1)]
    stages[0] = 's'; stages[-1] = 't'
    edges = []
    for s in range(k):
        u, v = stages[s], stages[s+1]
        for _ in range(m):
            edges.append((u, v))
    return FlowGame(edges, [1]*len(edges),
                    name=f'Series({k})ofParallel({m})')


def make_capacitated_path_bypass():
    edges = [('s','a'),('a','t'),('a','b'),('b','t'),('s','b')]
    caps  = [3, 1, 1, 1, 1]
    return FlowGame(edges, caps, name='CapacitatedBypass')

# def make_unbalanced_series_parallel():
#     """Series-parallel: (s→a→t) ∥ (s→b→t), but unbalanced capacities."""
#     k = 2
#     m = 2
#     stages = [f'v{s}' for s in range(k+1)]
#     stages[0] = 's'; stages[-1] = 't'
#     edges = []
#     for s in range(k):
#         u, v = stages[s], stages[s+1]
#         for _ in range(m):
#             edges.append((u, v))
#         # m = 2
#     return FlowGame(edges, [1,1, 2,2],
#                     name=f'Series({k})ofParallel({m})')


# ════════════════════════════════════════════════════════════════════
# IV.  STRUCTURAL THEOREMS  (analytical verification)
# ════════════════════════════════════════════════════════════════════

def verify_series_theorem(max_k=6):
    results = []
    for k in range(2, max_k+1):
        g = make_path(k)
        a = g.analyze()
        results.append({
            'k': k,
            'in_core': a['in_core'],
            'convex': a['convex'],
            'shapley': a['shapley_f'],
            'unique_mincut': a['unique_mincut'],
            'n_mincuts': a['n_mincuts'],
        })
    return results


def verify_parallel_theorem(max_k=6):
    results = []
    for k in range(2, max_k+1):
        g = make_parallel(k)
        a = g.analyze()
        results.append({
            'k': k,
            'in_core': a['in_core'],
            'convex': a['convex'],
            'shapley': a['shapley_f'],
        })
    return results


# ════════════════════════════════════════════════════════════════════
# V.  CORE CHARACTERISATION  (LP dual interpretation)
# ════════════════════════════════════════════════════════════════════

def core_extreme_points(game: FlowGame):
    """
    The core of a flow game equals the set of feasible dual solutions
    of the LP relaxation of max-flow (Kern-Paulusma 2009).
    We find the range [min, max] of each player's payoff over the core.
    Uses simple LP: minimise/maximise x_i subject to core constraints.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None

    n = game.n
    N = frozenset(range(n))
    grand = game.grand_value()

    # Build constraint matrix: for each S ⊆ N, S ≠ N, ∑_{i∈S} x_i ≥ v(S)
    # Also: ∑ x_i = grand (add as two inequalities)
    A_ub, b_ub = [], []
    for mask in range(1, (1 << n) - 1):
        S = frozenset(i for i in range(n) if mask >> i & 1)
        vs = game.v(S)
        if vs > 0:
            row = [-1 if i in S else 0 for i in range(n)]
            A_ub.append(row); b_ub.append(-vs)

    # Equality: sum = grand
    A_eq = [[1]*n]; b_eq = [grand]

    core_ranges = []
    for player in range(n):
        c_min = [0]*n; c_min[player] = 1
        c_max = [0]*n; c_max[player] = -1

        r_min = linprog(c_min, A_ub=A_ub, b_ub=b_ub,
                        A_eq=A_eq, b_eq=b_eq,
                        bounds=[(0, None)]*n, method='highs')
        r_max = linprog(c_max, A_ub=A_ub, b_ub=b_ub,
                        A_eq=A_eq, b_eq=b_eq,
                        bounds=[(0, None)]*n, method='highs')

        lo = r_min.fun if r_min.success else None
        hi = -r_max.fun if r_max.success else None
        core_ranges.append((lo, hi))

    return core_ranges


# ════════════════════════════════════════════════════════════════════
# VI.  MAIN EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════

def run_all():
    games = [
        make_path(2), make_path(3), make_path(4), make_path(5),
        make_parallel(2), make_parallel(3), make_parallel(4),
        make_diamond(),
        make_diamond_with_bridge(),
        make_series_parallel_balanced(),
        make_three_paths(),
        make_wheatstone(),
        make_grid_2x2(),
        make_bottleneck(),
        make_two_bottlenecks(),
        make_asymmetric_caps(),
        make_triangle(),
        make_redundant_path(),
        make_fan_in(),
        make_k4_network(),
        make_series_of_parallel(2, 2),
        make_series_of_parallel(3, 2),
        make_series_of_parallel(2, 3),
        make_capacitated_path_bypass(),
        # make_unbalanced_series_parallel(),
    ]

    results = []
    for g in games:
        print(f"  Analysing {g.name} (n={g.n})…")
        a = g.analyze()
        # Add core ranges
        try:
            a['core_ranges'] = core_extreme_points(g)
        except Exception:
            a['core_ranges'] = None
        results.append(a)

    return results


def summarise(results):
    in_core     = [r for r in results if r['in_core']]
    not_in_core = [r for r in results if not r['in_core']]

    print("\n" + "═"*60)
    print("  FLOW GAME RESEARCH — RESULTS SUMMARY")
    print("═"*60)

    print(f"\n✓ Shapley IN core  ({len(in_core)} / {len(results)}):")
    for r in in_core:
        print(f"    {r['name']:40s}  n={r['n']:2d}  cuts={r['n_mincuts']:3d}"
              f"  convex={r['convex']}")

    print(f"\n✗ Shapley NOT in core  ({len(not_in_core)} / {len(results)}):")
    for r in not_in_core:
        print(f"    {r['name']:40s}  n={r['n']:2d}  cuts={r['n_mincuts']:3d}"
              f"  convex={r['convex']}"
              f"  deficit={r['deficit']:.4f}"
              f"  bad_S={r['violating']}")

    print("\n" + "─"*60)
    print("  PATTERN ANALYSIS")
    print("─"*60)

    # Unique cut heuristic
    unique_and_core = sum(1 for r in in_core if r['unique_mincut'])
    unique_total    = sum(1 for r in results if r['unique_mincut'])
    print(f"\n  Unique min-cut → Shapley in core?"
          f"  {unique_and_core}/{unique_total} unique-cut games have Shapley in core.")

    # Convexity
    convex_and_core = sum(1 for r in in_core if r['convex'])
    convex_total    = sum(1 for r in results if r['convex'])
    print(f"  Convex → Shapley in core?  "
          f"  {convex_and_core}/{convex_total} convex games have Shapley in core.")

    return in_core, not_in_core


if __name__ == '__main__':
    print("Running flow game experiments…")
    results = run_all()

    print("\nRaw results for each network:")
    for r in results:
        phi_str = ', '.join(r['shapley'])
        status  = '✓ Core' if r['in_core'] else f'✗ Deficit={r["deficit"]:.4f}'
        print(f"\n{r['name']}")
        print(f"  n={r['n']}, grand={r['grand']}, cuts={r['n_mincuts']}, "
              f"unique={r['unique_mincut']}, convex={r['convex']}")
        print(f"  φ = [{phi_str}]")
        print(f"  {status}")
        if not r['in_core']:
            print(f"  Violating coalition: {r['violating']}")
        if r['core_ranges']:
            rng = ', '.join(f"[{lo:.3f},{hi:.3f}]"
                            for lo, hi in r['core_ranges'])
            print(f"  Core ranges: {rng}")

    in_core, not_in_core = summarise(results)

    # Save JSON for visualisation
    with open('/home/saksham-raj/Desktop/Semester 8/CS 6002/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /home/saksham-raj/Desktop/Semester 8/CS 6002/results.json")
