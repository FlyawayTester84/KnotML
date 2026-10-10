# PD core: design notes

*2026-10-10. Implements item 1 of section 3 in `research-summary.md`
(`src/knotml/core/`). Kirby moves, encoders and the environment are not done yet.*

## Representation

A diagram with `n` crossings is stored as two flat lists plus a counter:

- `nbr` (length `4n`): crossing `c` owns **ports** `4c..4c+3`, counterclockwise
  from the incoming under-strand (the KnotInfo/Regina/SnapPy/Sage PD
  convention). `nbr[p]` is the port at the other end of the edge at `p`.
- `sign` (length `n`); `n_unknots` counts crossingless components.

So `p ^ 2` is where the strand through `p` continues, odd ports are over, and
the slot where the over-strand enters (3 or 1) is given by the sign. This is
the same information as signed Gauss code, so conversion to and from Regina is
exact. A PD code is lossy only for a component that passes over at every
crossing (`is_pd_ambiguous()`).

**Sides are ports.** Port `p` names the side of its edge that lies
counterclockwise after `p` (an edge's tail names its left side, its head its
right side). Faces are the orbits of `p -> nbr[next_ccw(p)]`. A planarity check
counts faces (`n_k + 2` per connected piece; Mastin, Thm 14).

## Moves

| Kind | Anchor | Legal when | Check |
|---|---|---|---|
| R1 remove | monogon corner `a` | `nbr[next(a)] == a` | O(1) |
| R1 add | side `a` (or `TRIVIAL`), `sign` | always | O(1) |
| R2 remove | bigon side `a` of its upper edge | 2-cycle of the corner map, distinct crossings, edge at `a` over at both ends | O(1) |
| R2 add | over side `a`, under side `b` (or `TRIVIAL`) | different edges and `face[a] == face[b]`, or different pieces of the projection | O(1) after O(n) face cache |
| R3 | triangle side `a` of its top edge | 3-cycle, distinct crossings, edge at `a` over at both ends | O(1) |
| Crossing change | crossing | always | O(1) |

Each legal move has exactly one anchor, so enumerations have no duplicates.

- **Applying a move** (`apply_inplace`) rewires O(1) ports. A deleted crossing
  is filled by moving the last crossing into its slot, and the inverse move is
  returned.
- **R3** keeps the three crossings, with their signs and slot roles, and swaps
  the order of the two crossings along each strand. The inverse is anchored at
  `a ^ 2`.
- **Undo can fail in one case:** an R2 removal that merges its two strands
  into a single edge returns `None`, because undoing it needs an edge pushed
  over itself. Regina does not offer that move either.

**Action-space support:** `move_masks()` returns all legality masks in O(n)
space. The R2-add mask is returned factored, as a face id and a piece id per
side, because the full pairwise mask has Θ(n²) entries.

**Canonical key:** walk the diagram from each of the `2n` starting edges and
keep the lexicographically smallest code. A walk stops as soon as it falls
behind the best code so far. This takes O(n²) in the worst case and is close
to linear in practice.

## Verification

- The tests compare against Regina on random diagrams:
  - legal sets for every move kind (all port pairs for R2 add);
  - the result of every legal move against Regina's `with*` output, using
    strict `sig()` with no reflection or reversal;
  - canonical-key equality against `sig()` equality;
  - Jones polynomial along random walks.
- Random walks check validity, determinant invariance and that every inverse
  restores the canonical key.

## Measured costs

Python 3.12, Apple silicon, one random diagram per size:

| n | `move_masks` | `legal_moves` (no R2 add) | `legal_moves` (all) | # R2 add | one check | apply+undo | canonical key | Regina `hasR2` over all pairs |
|---|---|---|---|---|---|---|---|---|
| 26 | 0.02 ms | 0.04 ms | 0.10 ms | 424 | 0.14 µs | 2.1 µs | 0.04 ms | 2.6 ms |
| 101 | 0.06 ms | 0.17 ms | 0.51 ms | 1,898 | 0.16 µs | 1.9 µs | 0.25 ms | 40 ms |
| 200 | 0.11 ms | 0.35 ms | 1.11 ms | 3,904 | 0.16 µs | 2.0 µs | 0.54 ms | 155 ms |
| 401 | 0.23 ms | 0.70 ms | 2.42 ms | 7,900 | 0.16 µs | 1.9 µs | 1.74 ms | — |

## Next

1. Choose the action-space design (`research-summary.md` §2) and write the
   observation encoder. `UnknotEnv` raises `NotImplementedError` at these two points.
2. Build `FramedLink` (framings, dotted circles, blow-ups, handle slides) on
   this core. `component_writhes()` already gives the blackboard framing.
