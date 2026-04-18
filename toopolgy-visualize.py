"""
Topology Visualization for Fair Allocation in Flow Games
=========================================================
Generates a multi-page PDF with one panel per topology showing:
  - The directed graph (nodes + edges, with s/t highlighted)
  - Per-edge Shapley value labels
  - Per-edge cut-frequency labels
  - Summary stats: sigma^2, core deficit, convex?, in_core?
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
from fractions import Fraction
import math

# ── reuse the analysis module ──────────────────────────────────────────────
from flow_game_analysis import (
    make_topologies,
    compute_characteristic_function,
    shapley_value,
    core_deficit,
    find_all_min_cuts,
    cut_variance,
)


# ── colour palette ─────────────────────────────────────────────────────────
C_SOURCE   = "#4C6EF5"   # blue   – source node
C_SINK     = "#F03E3E"   # red    – sink node
C_MIDDLE   = "#2F9E44"   # green  – intermediate nodes
C_EDGE_OK  = "#2F9E44"   # green  – edges in core (in_core=True)
C_EDGE_BAD = "#F03E3E"   # red    – edges causing deficit
C_EDGE_DEF = "#495057"   # gray   – default edge colour
C_BG_OK    = "#EBFBEE"   # light green panel bg
C_BG_BAD   = "#FFF5F5"   # light red panel bg
C_BG_NEU   = "#F8F9FA"   # neutral panel bg


def convexity_check(n, v):
    """Return True if game is convex (supermodular)."""
    from itertools import combinations
    players = list(range(n))
    for i in players:
        rest = [p for p in players if p != i]
        for r in range(len(rest) + 1):
            for S_list in combinations(rest, r):
                S = frozenset(S_list)
                for extra in combinations([p for p in rest if p not in S], 1):
                    T = S | frozenset(extra)
                    if v[S | {i}] - v[S] > v[T | {i}] - v[T]:
                        return False
    return True


def analyze(name, nodes, source, sink, edge_list):
    n = len(edge_list)
    if n > 13:
        return None
    v      = compute_characteristic_function(edge_list, nodes, source, sink)
    phi    = shapley_value(n, v)
    deficit= core_deficit(n, v, phi)
    min_cuts, mf = find_all_min_cuts(edge_list, nodes, source, sink)
    sigma2, n_cuts, freqs = cut_variance(min_cuts, edge_list)
    is_convex = convexity_check(n, v)
    return dict(name=name, nodes=nodes, source=source, sink=sink,
                edge_list=edge_list, n=n, max_flow=mf,
                phi=phi, deficit=deficit, in_core=(deficit==0),
                is_convex=is_convex, n_cuts=n_cuts,
                sigma2=sigma2, sigma2_f=float(sigma2),
                deficit_f=float(deficit), freqs=freqs)


def _layout(nodes, source, sink, edge_list):
    """
    Return a {node: (x,y)} pos dict.
    Uses a left-to-right layered layout via BFS from source,
    then falls back to spring layout if layers are awkward.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_list)

    # BFS layers from source
    layers = {}
    visited = {source: 0}
    queue = [source]
    while queue:
        nxt = []
        for u in queue:
            for v in G.successors(u):
                if v not in visited:
                    visited[v] = visited[u] + 1
                    nxt.append(v)
        queue = nxt
    for nd in nodes:
        if nd not in visited:
            visited[nd] = 0

    # Force sink to last layer
    max_layer = max(visited.values()) if visited else 0
    visited[sink] = max_layer

    from collections import defaultdict
    by_layer = defaultdict(list)
    for nd, lyr in visited.items():
        by_layer[lyr].append(nd)

    pos = {}
    n_layers = max(by_layer.keys()) + 1
    for lyr, nds in by_layer.items():
        x = lyr / max(n_layers - 1, 1)
        for idx, nd in enumerate(sorted(nds)):
            y = (idx + 0.5) / len(nds)
            pos[nd] = (x, y)

    return pos


def _phi_str(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def draw_topology(ax, result):
    """Draw one topology panel onto ax."""
    r = result
    edge_list = r["edge_list"]
    n = r["n"]
    source, sink = r["source"], r["sink"]

    # Build DiGraph (handle multi-edges by collapsing for drawing)
    G = nx.MultiDiGraph()
    G.add_nodes_from(r["nodes"])
    for (u, v) in edge_list:
        G.add_edge(u, v)

    pos = _layout(r["nodes"], source, sink, edge_list)

    # ── node colours ──
    node_colors = []
    node_sizes  = []
    for nd in G.nodes():
        if nd == source:
            node_colors.append(C_SOURCE)
            node_sizes.append(700)
        elif nd == sink:
            node_colors.append(C_SINK)
            node_sizes.append(700)
        else:
            node_colors.append(C_MIDDLE)
            node_sizes.append(400)

    # ── draw nodes ──
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes, alpha=0.92)

    # ── node labels: just s/t, hide internal ids ──
    labels = {}
    for nd in G.nodes():
        if nd == source:
            labels[nd] = "s"
        elif nd == sink:
            labels[nd] = "t"
        else:
            labels[nd] = ""
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_color="white", font_size=9, font_weight="bold")

    # ── draw edges with curved arrows (handle multi-edges) ──
    # Group edges by (u,v) to apply different rad for parallel edges
    from collections import defaultdict
    edge_count = defaultdict(int)
    edge_rad   = {}
    for u, v in edge_list:
        key = (min(u, v), max(u, v))
        edge_count[key] += 1

    seen = defaultdict(int)
    for idx, (u, v) in enumerate(edge_list):
        key = (min(u, v), max(u, v))
        total = edge_count[key]
        i = seen[key]
        seen[key] += 1

        # spread curves: 0, ±0.2, ±0.4 ...
        offsets = [0] if total == 1 else [
            (j - (total - 1) / 2) * 0.25 for j in range(total)
        ]
        rad = offsets[i]

        # colour by sigma² > 0: if edge is exclusive min-cut carrier → red
        freq = r["freqs"][idx]
        if not r["in_core"]:
            ec = C_EDGE_BAD if freq > 0 else C_EDGE_DEF
        else:
            ec = C_EDGE_OK

        ax.annotate("",
            xy=pos[v], xycoords="data",
            xytext=pos[u], textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=ec,
                lw=1.4,
                connectionstyle=f"arc3,rad={rad}",
                mutation_scale=14,
            ))

    # ── edge labels: φ value ──
    # Place label at midpoint along each edge (approximate for curves)
    phi = r["phi"]
    for idx, (u, v) in enumerate(edge_list):
        xu, yu = pos[u]
        xv, yv = pos[v]
        # slight perpendicular offset for curved multi-edges
        key = (min(u, v), max(u, v))
        total = edge_count[key]
        offsets = [0] if total == 1 else [
            (j - (total - 1) / 2) * 0.25 for j in range(total)
        ]
        seen2 = defaultdict(int)
        # recompute i for this edge
        i_local = 0
        for jj, (uu, vv) in enumerate(edge_list):
            kk = (min(uu, vv), max(uu, vv))
            if kk == key:
                if jj == idx:
                    i_local = seen2[kk]
                seen2[kk] += 1

        rad = offsets[i_local] if i_local < len(offsets) else 0
        # midpoint with slight curve offset
        mx = (xu + xv) / 2 + rad * (yv - yu) * 0.5
        my = (yu + yv) / 2 - rad * (xv - xu) * 0.5

        phi_s = _phi_str(phi[idx])
        freq  = r["freqs"][idx]
        label = f"φ={phi_s}\nf={freq}"

        ax.text(mx, my, label,
                fontsize=5.5, ha="center", va="center",
                color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="#cccccc", lw=0.5, alpha=0.85))

    # ── panel title & stats ──
    in_core_s  = "✓ In Core"  if r["in_core"]   else "✗ Not in Core"
    convex_s   = "Convex"     if r["is_convex"]  else "Non-convex"
    color_ic   = "#2F9E44"    if r["in_core"]    else "#F03E3E"

    ax.set_title(
        f"{r['name']}\n"
        f"n={n}  MaxFlow={r['max_flow']}  {convex_s}",
        fontsize=7.5, fontweight="bold", pad=4, color="#222222"
    )

    # bottom annotation
    stats = (f"σ²={r['sigma2_f']:.4f}   deficit={r['deficit_f']:.4f}   "
             f"#min-cuts={r['n_cuts']}")
    ax.text(0.5, -0.08, stats,
            transform=ax.transAxes, fontsize=6,
            ha="center", va="top", color="#555555")
    ax.text(0.5, -0.14, in_core_s,
            transform=ax.transAxes, fontsize=6.5, fontweight="bold",
            ha="center", va="top", color=color_ic)

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # panel background
    bg = C_BG_OK if r["in_core"] else C_BG_BAD
    ax.set_facecolor(bg)


def make_legend(fig):
    handles = [
        mpatches.Patch(color=C_SOURCE, label="Source (s)"),
        mpatches.Patch(color=C_SINK,   label="Sink (t)"),
        mpatches.Patch(color=C_MIDDLE, label="Intermediate node"),
        mpatches.Patch(color=C_EDGE_OK,  label="Edge (in-core topology)"),
        mpatches.Patch(color=C_EDGE_BAD, label="Edge crossing min-cut (deficit topology)"),
        mpatches.Patch(color=C_EDGE_DEF, label="Edge not in any min-cut"),
        mpatches.Patch(color=C_BG_OK,  label="Panel: Shapley in core"),
        mpatches.Patch(color=C_BG_BAD, label="Panel: Shapley NOT in core"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=7, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))


def run():
    topologies = make_topologies()

    print("Analyzing topologies...")
    results = []
    for (name, nodes, source, sink, edges) in topologies:
        print(f"  {name}  (n={len(edges)})", end="", flush=True)
        r = analyze(name, nodes, source, sink, edges)
        if r is None:
            print("  [skip]")
            continue
        results.append(r)
        print(f"  σ²={r['sigma2_f']:.4f}  deficit={r['deficit_f']:.4f}")

    # ── sort: in-core first, then by sigma² ──
    results.sort(key=lambda r: (not r["in_core"], r["sigma2_f"]))

    n_topo  = len(results)
    cols    = 4
    rows_per_page = 3
    per_page = cols * rows_per_page

    pages = math.ceil(n_topo / per_page)
    all_figs = []

    for page in range(pages):
        batch = results[page * per_page : (page + 1) * per_page]
        n_batch = len(batch)

        fig, axes = plt.subplots(rows_per_page, cols,
                                 figsize=(cols * 4.2, rows_per_page * 3.8 + 0.9))
        fig.patch.set_facecolor("white")

        fig.suptitle(
            "Fair Allocation in Flow Games — Topology Analysis\n"
            f"Edge labels: φ = Shapley value,  f = min-cut frequency\n"
            f"(Page {page+1}/{pages})",
            fontsize=11, fontweight="bold", y=0.99
        )

        axes_flat = axes.flatten()

        for i, r in enumerate(batch):
            draw_topology(axes_flat[i], r)

        # hide unused axes
        for i in range(n_batch, len(axes_flat)):
            axes_flat[i].set_visible(False)

        make_legend(fig)
        fig.subplots_adjust(left=0.02, right=0.98,
                            top=0.91, bottom=0.10,
                            hspace=0.55, wspace=0.25)
        all_figs.append(fig)

    # ── save as multi-page PDF ──
    from matplotlib.backends.backend_pdf import PdfPages
    out_path = "/home/claude/topology_analysis.pdf"
    with PdfPages(out_path) as pdf:
        for fig in all_figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved → {out_path}  ({pages} page(s), {n_topo} topologies)")
    return out_path


if __name__ == "__main__":
    run()