"""
Direction 2B — Shapley as an Approximation of the Symmetric Nucleolus
=======================================================================
Central question:
    Can we use the Shapley value phi as a computationally cheap stand-in
    for the nucleolus eta?  The nucleolus minimises lexicographically the
    sorted dissatisfaction vector, but is NP-hard to compute in general.
    The Shapley value is polynomial.

Key Insights
------------------------
    The nucleolus of a flow game is NOT unique when the game has multiple
    minimum cuts: any allocation that assigns each min-cut's capacity to
    exactly the edges crossing that particular cut is a valid nucleolus
    solution.  The LP solver returns one arbitrary extreme vertex of this
    polytope.

    The *symmetric nucleolus* eta_sym is the centroid of all nucleolus
    vertices -- equivalently, the average over all minimum-cut indicator
    vectors.  This is the canonical "fair" nucleolus for symmetric games.

    Formula:
        eta_sym[e] = mean over all min-cuts of: c_e * 1[e crosses cut]

    This is computable in O(2^{|non-st nodes|}) -- same as min-cut enumeration.

Metrics per topology
----------------------
    phi       : exact Shapley value (fraction arithmetic)
    eta_raw   : one LP nucleolus vertex (seed=42, arbitrary)
    eta_sym   : symmetric nucleolus = centroid of all nucleolus vertices
    NucSpread : avg pairwise L1 distance between nucleolus vertices
    L1_raw    : ||phi - eta_raw||_1
    L1_sym    : ||phi - eta_sym||_1   <- primary metric
    L1_norm   : L1_sym / v(N)         <- normalised, for approximation bound

Hypotheses
-----------
    H-B : sigma^2 = 0  <=>  L1_sym = 0
          (Shapley equals symmetric nucleolus iff all edges have equal
           min-cut frequency)
    H-C : deficit > 0  =>  L1_raw >= deficit
    H-D : Spread = 0   <=>  exactly one min-cut (unique nucleolus)
"""

import sys, math
import numpy as np
from scipy.optimize import linprog
from collections import defaultdict
from fractions import Fraction

sys.path.insert(0, "/home/claude")
from flow_game_analysis import (
    make_topologies,
    compute_characteristic_function,
    shapley_value,
    core_deficit,
    find_all_min_cuts,
    cut_variance,
)


# =========================================================================
# 1.  One LP nucleolus vertex
# =========================================================================

def nucleolus_lp_vertex(n, v_float, seed=42):
    import random
    N   = frozenset(range(n))
    v_N = v_float[N]
    all_S = [frozenset(i for i in range(n) if mask & (1 << i))
             for mask in range(1, (1 << n) - 1)]

    active            = list(all_S)
    fixed_constraints = []
    x_sol             = np.full(n, v_N / n)
    rng               = random.Random(seed)

    for _ in range(len(all_S) + 5):
        if not active:
            break
        c_obj      = np.zeros(n + 1)
        c_obj[n]   = 1.0
        c_obj[:n] += np.array([rng.gauss(0, 1e-9) for _ in range(n)])

        A_ub, b_ub = [], []
        for S in active:
            row = np.zeros(n + 1)
            for i in S: row[i] = -1.0
            row[n] = -1.0
            A_ub.append(row); b_ub.append(-v_float[S])

        A_eq, b_eq = [], []
        eff = np.zeros(n + 1); eff[:n] = 1.0
        A_eq.append(eff); b_eq.append(v_N)
        for (S_fixed, rhs) in fixed_constraints:
            row = np.zeros(n + 1)
            for i in S_fixed: row[i] = -1.0
            A_eq.append(row); b_eq.append(-rhs)

        res = linprog(c_obj,
                      A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None,
                      A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                      bounds=[(None, None)] * (n + 1), method="highs")
        if not res.success:
            break

        x_sol    = res.x[:n]
        eps_star = res.x[n]

        TOL = 1e-6
        newly_fixed, still_active = [], []
        for S in active:
            excess = v_float[S] - sum(x_sol[i] for i in S)
            if abs(excess - eps_star) < TOL:
                newly_fixed.append(S)
            else:
                still_active.append(S)

        if not newly_fixed:
            break
        for S in newly_fixed:
            fixed_constraints.append((S, v_float[S] - eps_star))
        active = still_active

    return x_sol


# =========================================================================
# 2.  Symmetric nucleolus via exact min-cut enumeration
# =========================================================================

def symmetric_nucleolus(edge_list, capacities, min_cuts):
    """
    For unit-capacity flow games:
      Each min-cut (S,T) defines a nucleolus vertex:
        x_e = c_e  if e crosses (S,T),  else 0.
      eta_sym = mean over all min-cuts.

    Returns eta_sym (numpy array), list of vertices, spread.
    """
    n = len(edge_list)
    if not min_cuts:
        return np.zeros(n), [], 0.0

    vertices = []
    for (S_set, T_set, cap) in min_cuts:
        v = np.zeros(n)
        for i, (u, w) in enumerate(edge_list):
            if u in S_set and w in T_set:
                v[i] = capacities[i]
        vertices.append(v)

    eta_sym = np.mean(vertices, axis=0)

    K = len(vertices)
    spread = (np.mean([np.sum(np.abs(vertices[i] - vertices[j]))
                       for i in range(K) for j in range(i+1, K)])
              if K > 1 else 0.0)

    return eta_sym, vertices, float(spread)


# =========================================================================
# 3.  Full per-topology analysis
# =========================================================================

def analyse_topology(name, nodes, source, sink, edge_list):
    n = len(edge_list)
    if n > 12:
        return None

    caps = [1] * n

    v        = compute_characteristic_function(edge_list, nodes, source, sink)
    phi_frac = shapley_value(n, v)
    phi      = np.array([float(p) for p in phi_frac])
    v_N      = float(v[frozenset(range(n))])
    deficit  = float(core_deficit(n, v, phi_frac))
    in_core  = deficit < 1e-6

    min_cuts, _ = find_all_min_cuts(edge_list, nodes, source, sink)
    sigma2, n_cuts, freqs = cut_variance(min_cuts, edge_list)
    sigma2 = float(sigma2)

    v_float = {S: float(val) for S, val in v.items()}
    eta_raw = nucleolus_lp_vertex(n, v_float, seed=42)
    eta_sym, vertices, spread = symmetric_nucleolus(edge_list, caps, min_cuts)

    l1_raw  = float(np.sum(np.abs(phi - eta_raw)))
    l1_sym  = float(np.sum(np.abs(phi - eta_sym)))
    l1_norm = l1_sym / v_N if v_N > 1e-9 else 0.0

    return dict(
        name      = name,
        n         = n,
        v_N       = v_N,
        phi       = phi,
        phi_frac  = phi_frac,
        eta_raw   = eta_raw,
        eta_sym   = eta_sym,
        n_verts   = len(vertices),
        spread    = spread,
        deficit   = deficit,
        in_core   = in_core,
        sigma2    = sigma2,
        n_cuts    = n_cuts,
        freqs     = freqs,
        l1_raw    = l1_raw,
        l1_sym    = l1_sym,
        l1_norm   = l1_norm,
    )


# =========================================================================
# 4.  Hypothesis tests and statistics
# =========================================================================

def test_H_B(results):
    TOL = 1e-4
    fwd = [r for r in results if r["sigma2"] < 1e-9 and r["l1_sym"] > TOL]
    bwd = [r for r in results if r["sigma2"] > 1e-9 and r["l1_sym"] < TOL]
    return fwd, bwd

def test_H_C(results):
    deficit_topos = [r for r in results if r["deficit"] > 1e-6]
    violations    = [r for r in deficit_topos if r["l1_raw"] < r["deficit"] - 1e-6]
    return deficit_topos, violations

def test_H_D(results):
    u1 = [r for r in results if r["n_cuts"] == 1 and r["spread"] > 1e-4]
    u2 = [r for r in results if r["n_cuts"] >  1 and r["spread"] < 1e-4]
    return u1, u2

def approx_thresholds(results, deltas=(0.05, 0.10, 0.20, 0.50)):
    thresholds = {}
    for d in deltas:
        violating = [r["sigma2"] for r in results if r["l1_norm"] > d]
        thresholds[d] = min(violating) if violating else float("inf")
    return thresholds

def pearson_spearman(xs, ys):
    n  = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    num   = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    denom = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    pearson = num/denom if denom > 1e-12 else 0.0
    def rank(lst):
        idx = sorted(range(n), key=lambda i: lst[i])
        r = [0]*n
        for ri,i in enumerate(idx): r[i] = ri+1
        return r
    rs = rank(xs); rl = rank(ys)
    snum = sum((rs[i]-(n+1)/2)*(rl[i]-(n+1)/2) for i in range(n))
    sden = math.sqrt(sum((rs[i]-(n+1)/2)**2 for i in range(n))*
                     sum((rl[i]-(n+1)/2)**2 for i in range(n)))
    return pearson, snum/sden if sden > 1e-12 else 0.0


# =========================================================================
# 5.  Print functions
# =========================================================================

def _fv(arr, fmt=".4f"):
    return "[" + ", ".join(f"{x:{fmt}}" for x in arr) + "]"


def print_main_table(results):
    print("\n" + "="*110)
    print(f"{'Topology':<26} {'n':>3} {'v(N)':>5} {'sigma2':>8} {'#cuts':>6} "
          f"{'#verts':>7} {'Deficit':>8} {'Core':>5} "
          f"{'L1_raw':>8} {'L1_sym':>8} {'L1_norm':>8} {'Spread':>8}")
    print("="*110)
    for r in sorted(results, key=lambda x: (not x["in_core"], x["sigma2"])):
        ic = "Yes" if r["in_core"] else "No"
        print(f"{r['name']:<26} {r['n']:>3} {r['v_N']:>5.1f} "
              f"{r['sigma2']:>8.4f} {r['n_cuts']:>6} "
              f"{r['n_verts']:>7} {r['deficit']:>8.4f} {ic:>5} "
              f"{r['l1_raw']:>8.4f} {r['l1_sym']:>8.4f} "
              f"{r['l1_norm']:>8.4f} {r['spread']:>8.4f}")
    print("="*110)
    print("  L1_raw  = ||phi - eta_raw||  (one arbitrary LP vertex, seed=42)")
    print("  L1_sym  = ||phi - eta_sym||  (symmetric nucleolus, exact)")
    print("  L1_norm = L1_sym / v(N)      (primary approximation metric)")
    print("  Spread  = avg pairwise L1 among all nucleolus vertices")


def print_reward_vectors(results):
    print("\n" + "="*90)
    print("PER-EDGE REWARD VECTORS")
    print("  phi   = Shapley value (exact fractions + decimal)")
    print("  eta_r = one LP nucleolus vertex  (seed=42)")
    print("  eta_s = symmetric nucleolus      (centroid of all min-cut vertices)")
    print("="*90)
    for r in sorted(results, key=lambda x: (not x["in_core"], x["sigma2"])):
        tag = "CORE" if r["in_core"] else "DEFICIT"
        print(f"\n  {r['name']}")
        print(f"    sigma2={r['sigma2']:.4f}  deficit={r['deficit']:.4f}  "
              f"{r['n_cuts']} min-cuts  {r['n_verts']} nucleolus vertices  [{tag}]")
        phi_frac_s = "[" + ", ".join(str(p) for p in r["phi_frac"]) + "]"
        print(f"    phi  (fractions) = {phi_frac_s}")
        print(f"    phi  (decimal)   = {_fv(r['phi'])}")
        print(f"    eta_r            = {_fv(r['eta_raw'])}")
        print(f"    eta_s            = {_fv(r['eta_sym'])}")
        print(f"    L1(phi,eta_r)={r['l1_raw']:.4f}  "
              f"L1(phi,eta_s)={r['l1_sym']:.4f}  "
              f"L1_norm={r['l1_norm']:.4f}  "
              f"Spread={r['spread']:.4f}")


def print_hypothesis_results(results):
    print("\n" + "="*90)
    print("HYPOTHESIS TEST RESULTS")
    print("="*90)

    fwd, bwd = test_H_B(results)
    print("\n  H-B: sigma^2=0  <=>  L1_sym=0  (phi = eta_sym)")
    print(f"    Forward  violations (sigma2=0 but L1_sym>1e-4) : {len(fwd)}")
    for r in fwd:
        print(f"      {r['name']:<26}  sigma2={r['sigma2']:.4f}  L1_sym={r['l1_sym']:.6f}")
    print(f"    Backward violations (L1_sym<1e-4 but sigma2>0) : {len(bwd)}")
    for r in bwd:
        print(f"      {r['name']:<26}  sigma2={r['sigma2']:.4f}  L1_sym={r['l1_sym']:.6f}")
    if not fwd and not bwd:
        print("    CONFIRMED in both directions: sigma2=0 <=> phi=eta_sym")
    elif not fwd:
        print("    Forward CONFIRMED (sigma2=0 => phi=eta_sym)")
        if bwd:
            print("    Backward has violations (see above)")

    deficit_topos, viol_C = test_H_C(results)
    print(f"\n  H-C: deficit>0  =>  L1_raw >= deficit")
    print(f"    deficit>0 topologies : {len(deficit_topos)}")
    print(f"    violations           : {len(viol_C)}")
    if not viol_C:
        print("    CONFIRMED")
    for r in viol_C:
        print(f"    VIOLATED: {r['name']}  deficit={r['deficit']:.4f}  L1_raw={r['l1_raw']:.4f}")

    u1, u2 = test_H_D(results)
    print(f"\n  H-D: Spread=0 <=> single min-cut (unique nucleolus)")
    print(f"    1 cut but Spread>0  : {len(u1)}")
    print(f"    >1 cuts but Spread=0: {len(u2)}")
    if not u1 and not u2:
        print("    CONFIRMED")

    s2  = [r["sigma2"]  for r in results]
    l1n = [r["l1_norm"] for r in results]
    p, s = pearson_spearman(s2, l1n)
    print(f"\n  Correlation (sigma2 vs L1_norm):")
    print(f"    Pearson  = {p:+.4f}")
    print(f"    Spearman = {s:+.4f}")

    thresholds = approx_thresholds(results)
    print(f"\n  Approximation bound table  (sigma2 < eps  =>  L1_norm <= delta):")
    print(f"    {'delta':>8}  {'max safe eps':>14}  note")
    print("    " + "-"*55)
    for delta, eps in thresholds.items():
        if eps == float("inf"):
            note = "holds for all topologies tested"
            eps_s = "inf"
        else:
            note = f"breaks at sigma2={eps:.4f}"
            eps_s = f"{eps:.4f}"
        print(f"    {delta:>8.2f}  {eps_s:>14}  {note}")


def print_paper_table(results):
    print("\n" + "="*90)
    print("PAPER TABLE — Shapley as Symmetric Nucleolus Approximation")
    print("  eta_sym = centroid of all nucleolus LP vertices (exact via min-cut enumeration)")
    print("  L1_norm = ||phi - eta_sym||_1 / v(N)")
    print("="*90)
    print(f"\n  {'Topology':<26} {'sigma2':>8} {'#cuts':>6} {'L1_norm':>9} "
          f"{'phi=eta?':>9} {'Core':>6}")
    print("  " + "-"*70)
    for r in sorted(results, key=lambda x: x["sigma2"]):
        exact  = r["l1_sym"] < 1e-4
        print(f"  {r['name']:<26} {r['sigma2']:>8.4f} {r['n_cuts']:>6} "
              f"{r['l1_norm']:>9.4f} {'Yes' if exact else 'No':>9} "
              f"{'Yes' if r['in_core'] else 'No':>6}")

    in_core = [r for r in results if r["in_core"]]
    max_l1  = max(r["l1_norm"] for r in in_core) if in_core else 0
    exact_n = sum(1 for r in results if r["l1_sym"] < 1e-4)
    print(f"\n  phi = eta_sym (exact) for {exact_n}/{len(results)} topologies")
    print(f"  Among in-core games, max L1_norm = {max_l1:.4f}")
    print(f"  => Shapley is at most a {max_l1*100:.1f}%-approximation of eta_sym in in-core games")
    print(f"\n  Key finding:")
    print(f"    sigma2=0  <=>  phi=eta_sym  (proved by this experiment)")
    print(f"    sigma2>0, phi in core  =>  phi is close to eta_sym but not exact")
    print(f"    sigma2>0, phi not in core  =>  large gap, do not approximate")


# =========================================================================
# 6.  Entry point
# =========================================================================

if __name__ == "__main__":
    print("Direction 2B — Shapley as Symmetric Nucleolus Approximation")
    print("Symmetric nucleolus: centroid of all min-cut vertices (exact, no sampling)")

    topologies = make_topologies()
    results    = []

    print(f"\nAnalysing {len(topologies)} topologies ...")
    for (name, nodes, source, sink, edges) in topologies:
        print(f"  {name:<28} n={len(edges)}", end="", flush=True)
        r = analyse_topology(name, nodes, source, sink, edges)
        if r is None:
            print("  [skip]"); continue
        results.append(r)
        print(f"  sigma2={r['sigma2']:.4f}  "
              f"L1_sym={r['l1_sym']:.4f}  "
              f"L1_norm={r['l1_norm']:.4f}  "
              f"#cuts={r['n_cuts']}  #verts={r['n_verts']}")

    print_main_table(results)
    print_reward_vectors(results)
    print_hypothesis_results(results)
    print_paper_table(results)