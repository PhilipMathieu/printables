"""Derive the 'hat' aperiodic monotile from its kite grid.

Smith, Myers, Kaplan & Goodman-Strauss (2023). The hat is a union of 8 kites
drawn on the hexagonal grid: join the midpoints of each hexagon's opposite
sides and every hexagon falls into six kites with angles 60-90-120-90.

Rather than trust a copied vertex list, the outline is searched for and then
checked against the tile's published invariants -- 13 edges, six of length
sqrt(3), six of length 1, one of length 2, area 8*sqrt(3).
"""

from __future__ import annotations

import math
from itertools import combinations

SQRT3 = math.sqrt(3.0)
KITE_AREA = SQRT3  # two right triangles with legs 1 and sqrt(3)

# Published invariants of the hat, used as the search's acceptance test.
HAT_EDGE_LENGTHS = {1.0: 6, SQRT3: 6, 2.0: 1}
HAT_AREA = 8 * KITE_AREA

MAX_INSET_UNITS = 0.72
"""Deepest inward offset the kernel survives, in hat units, with rounded
corners (``Kind.ARC``).

Measured in 0.02 steps: ARC reaches 0.74 before the offset fails, mitered
(``Kind.INTERSECTION``) only 0.62, tangent 0.80 but with heavy distortion.
Rounding wins because the hat's reflex corners are what tear a mitered offset
apart. 0.72 leaves a step of margin below the measured failure.

This is a hard geometric ceiling: the hat's widest inscribed circle is about
this radius, so past it there is simply no interior left to offset into.
"""

FAITHFUL_INSET_UNITS = 0.50
"""Deepest inset where the outline still reads as a hat rather than a blob.

Only relevant when shape fidelity matters more than how many rings fit. For
the nested fidget it does not -- nesting is the point -- so the part uses
MAX_INSET_UNITS and accepts the distortion.
"""

Point = tuple[float, float]
TOL = 1e-7


def _key(p: Point) -> tuple[int, int]:
    return (round(p[0] / TOL), round(p[1] / TOL))


def _hex_centers(rings: int = 1) -> list[Point]:
    """Centres of a hex patch. Hexagons have circumradius 2, so neighbouring
    centres sit 2*sqrt(3) apart along the 30+60k directions."""
    step = [
        (2 * SQRT3 * math.cos(math.radians(30 + 60 * k)),
         2 * SQRT3 * math.sin(math.radians(30 + 60 * k)))
        for k in range(6)
    ]
    seen = {(_key((0.0, 0.0))): (0.0, 0.0)}
    frontier = [(0.0, 0.0)]
    for _ in range(rings):
        nxt = []
        for cx, cy in frontier:
            for dx, dy in step:
                p = (cx + dx, cy + dy)
                if _key(p) not in seen:
                    seen[_key(p)] = p
                    nxt.append(p)
        frontier = nxt
    return list(seen.values())


def _kites() -> list[tuple[Point, ...]]:
    """Every kite of the patch, as 4 points: centre, midpoint, vertex, midpoint."""
    out: dict[tuple, tuple[Point, ...]] = {}
    for cx, cy in _hex_centers():
        vert = [(cx + 2 * math.cos(math.radians(60 * k)),
                 cy + 2 * math.sin(math.radians(60 * k))) for k in range(6)]
        mid = [(cx + SQRT3 * math.cos(math.radians(60 * k + 30)),
                cy + SQRT3 * math.sin(math.radians(60 * k + 30))) for k in range(6)]
        for k in range(6):
            kite = ((cx, cy), mid[k - 1], vert[k], mid[k])
            out[tuple(sorted(_key(p) for p in kite))] = kite
    return list(out.values())


def _edges(poly: tuple[Point, ...]) -> list[tuple[Point, Point]]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def _union_boundary(kites: list[tuple[Point, ...]]) -> list[Point] | None:
    """Union kites by cancelling shared edges, then walk the boundary.

    Exact because every kite edge is shared with at most one neighbour and the
    endpoints coincide to within TOL, so an edge on the outside appears once
    and an interior edge appears twice.
    """
    count: dict[frozenset, tuple[Point, Point]] = {}
    tally: dict[frozenset, int] = {}
    for kite in kites:
        for a, b in _edges(kite):
            e = frozenset((_key(a), _key(b)))
            tally[e] = tally.get(e, 0) + 1
            count[e] = (a, b)
    border = [count[e] for e, n in tally.items() if n == 1]
    if not border:
        return None

    # Each boundary vertex must have exactly two boundary edges, else the union
    # is pinched or disconnected and is not a simple polygon.
    adj: dict[tuple, list[Point]] = {}
    for a, b in border:
        adj.setdefault(_key(a), []).append(b)
        adj.setdefault(_key(b), []).append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None

    start = border[0][0]
    loop = [start]
    prev, cur = None, start
    while True:
        opts = adj[_key(cur)]
        nxt = opts[0] if (prev is None or _key(opts[0]) != _key(prev)) else opts[1]
        if _key(nxt) == _key(start):
            break
        loop.append(nxt)
        prev, cur = cur, nxt
        if len(loop) > len(border):
            return None
    return loop if len(loop) == len(border) else None


def _simplify(loop: list[Point]) -> list[Point]:
    """Drop vertices where two collinear edges meet, merging them into one."""
    out = []
    n = len(loop)
    for i in range(n):
        a, b, c = loop[i - 1], loop[i], loop[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            out.append(b)
    return out


def edge_lengths(poly: list[Point]) -> list[float]:
    return [math.dist(a, b) for a, b in _edges(tuple(poly))]


def area(poly: list[Point]) -> float:
    s = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges(tuple(poly)))
    return abs(s) / 2


def _matches_hat(poly: list[Point]) -> bool:
    if len(poly) != 13 or abs(area(poly) - HAT_AREA) > 1e-6:
        return False
    tally: dict[float, int] = {}
    for L in edge_lengths(poly):
        for want in HAT_EDGE_LENGTHS:
            if abs(L - want) < 1e-6:
                tally[want] = tally.get(want, 0) + 1
                break
        else:
            return False
    return tally == HAT_EDGE_LENGTHS


def find_hat() -> list[Point]:
    """Search connected 8-kite subsets for the one matching the hat."""
    kites = _kites()
    nbr: dict[int, set[int]] = {i: set() for i in range(len(kites))}
    edge_of: dict[frozenset, list[int]] = {}
    for i, k in enumerate(kites):
        for a, b in _edges(k):
            edge_of.setdefault(frozenset((_key(a), _key(b))), []).append(i)
    for owners in edge_of.values():
        for i, j in combinations(owners, 2):
            nbr[i].add(j)
            nbr[j].add(i)

    seen: set[frozenset] = set()
    stack = [frozenset({i}) for i in range(len(kites))]
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.add(sub)
        if len(sub) == 8:
            poly = _union_boundary([kites[i] for i in sub])
            if poly and _matches_hat(_simplify(poly)):
                return _simplify(poly)
            continue
        for i in sub:
            for j in nbr[i]:
                if j not in sub:
                    stack.append(sub | {j})
    raise RuntimeError("no 8-kite subset matched the hat's invariants")


if __name__ == "__main__":
    poly = find_hat()
    print(f"vertices: {len(poly)}  area: {area(poly):.6f} (want {HAT_AREA:.6f})")
    for i, (p, L) in enumerate(zip(poly, edge_lengths(poly))):
        print(f"  v{i:2d} ({p[0]:9.5f},{p[1]:9.5f})  edge {L:.6f}")
