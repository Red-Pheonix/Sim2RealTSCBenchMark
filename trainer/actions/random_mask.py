"""Random valid phase-transition constraints for Domain Randomization.

Generates synthetic constraint tables by starting from a PERMISSIVE base (flexible -- all
movements allowed) and randomly DISALLOWING a fraction of the off-diagonal transitions,
subject to keeping every intersection's transition graph STRONGLY CONNECTED (every phase
still reachable from every other -- no stranded movement). Timings (min/max dwell) are
left untouched; only the allowed/disallowed structure is randomized.

This is the honest DR signal: it never reads the deployment table (barrier/cyclic), only
random subsets of the permissive graph -- so the target constraint is a member of the
randomized family by construction, never trained on directly. For reference on tempe_1x1
(intersection 99, 8 phases / 56 edges): barrier_leading_fixed disables ~43% of the edges,
cyclic ~86%, so a drop-rate range around those covers the deployment regime.
"""

import copy


def _strongly_connected(phases, edges):
    """True if the directed graph over ``phases`` with ``edges`` (set of (from, to)) is
    strongly connected: from any node, every other node is reachable, AND every node can
    reach it. Checked via reachability from one node in the graph and its reverse."""
    if len(phases) <= 1:
        return True
    adj, radj = {p: [] for p in phases}, {p: [] for p in phases}
    for f, t in edges:
        adj[f].append(t)
        radj[t].append(f)

    def reaches_all(graph):
        start = phases[0]
        seen, stack = {start}, [start]
        while stack:
            for nxt in graph[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return len(seen) == len(phases)

    return reaches_all(adj) and reaches_all(radj)


def random_constrained_table(base_table, drop_rate, rng, max_tries=200):
    """Return a copy of ``base_table`` with ``drop_rate`` of each intersection's allowed
    off-diagonal transitions disallowed at random, keeping the graph strongly connected.

    ``rng`` is a numpy Generator. Falls back to the fewest-dropped connected sample if no
    full-rate drop stays connected within ``max_tries`` (only bites at very high rates)."""
    new_table = copy.deepcopy(base_table)
    for node in new_table.values():
        _drop_node_edges(node, drop_rate, rng, max_tries)
    return new_table


def _drop_node_edges(node, drop_rate, rng, max_tries):
    trans = node["transitions"]
    phases = list(trans.keys())
    edges = [
        (f, t)
        for f, tos in trans.items()
        for t, info in tos.items()
        if f != t and int(info["allowed"])
    ]
    n_drop = int(round(drop_rate * len(edges)))
    if n_drop <= 0 or n_drop >= len(edges):
        n_drop = min(max(n_drop, 0), len(edges) - 1)  # never strand by dropping all

    best_drop = set()  # fallback: drop nothing (always connected) if nothing else works
    for _ in range(max_tries):
        idx = rng.choice(len(edges), size=n_drop, replace=False)
        drop = {edges[i] for i in idx}
        remaining = [e for e in edges if e not in drop]
        if _strongly_connected(phases, remaining):
            best_drop = drop
            break

    for (f, t) in best_drop:
        trans[f][t]["allowed"] = 0
