# Framed links and Kirby moves: design notes

*2026-10-10. Implements item 2 of section 3 in `research-summary.md`:
3-manifold surgery calculus in `src/knotml/kirby.py` (`FramedLink`) and
4-manifold handlebody calculus in `src/knotml/handlebody.py` (`Handlebody`).
They share the handle-slide construction (`band_slide`).*

## Data model

- **Component labels (core).** `LinkDiagram` now stores a label for every
  strand passage (`labels[2c]` = under-strand of crossing `c`, `labels[2c+1]` =
  over-strand), plus `trivial`, the labels of crossingless components. Every
  move keeps labels attached to their components:
  - new crossings inherit their strands' labels;
  - a crossing change swaps its two labels;
  - R3 keeps each crossing's pair of strands;
  - a loop that closes up moves its label to `trivial`.

  `check()` verifies that each component carries exactly one label. Anchors
  `-1 - i` (`trivial_anchor(i)`) name the `i`-th crossingless component.
- **`FramedLink`** = `LinkDiagram` + `{label: framing}`. The framing is the
  coefficient against the Seifert framing (linking number with the framing
  push-off), so Reidemeister moves leave it alone. The blackboard framing is
  the self-writhe, `diagram.component_writhes()`.
- **Invariants.**
  - `linking_matrix()` gives linking numbers off the diagonal and framings on
    it. It is the intersection form and presents H₁ of the surgery.
  - `homology()` is computed from it by Smith normal form.
  - `signature()` uses its eigenvalues.
  - `canonical_key()` adds a framing token at the start of each component's
    walk.
  - `to_snappy()` Dehn-fills the link exterior with slopes `(f, 1)`, using
    SnapPea's Seifert longitudes.

## Moves

| Move | Anchor | Legal when | Cost |
|---|---|---|---|
| Reidemeister | as in `core.moves` | as in `core.moves` (no crossing changes) | O(1) |
| `BLOW_UP(sign)` | — | always (3-manifold calculus) | O(1) |
| `BLOW_DOWN(a)` | label | `a` crossingless, framing ±1 (3-manifold calculus) | O(1) |
| `HANDLE_SLIDE(a, b)` | sides on K_i and K_j | different components; same face or different pieces (the `R2_ADD` rule) | O(n + \|f_j − w_j\|) |

By Kirby's theorem this set is complete for 3-manifolds. Split blow-downs
suffice: a ±1 unknot linked with other strands can first be freed by sliding
those strands off it. Blow-ups change the 4-manifold by #±CP², so the
4-manifold calculus lives in `Handlebody` instead (below).

### Handle slide construction (`kirby.py`)

1. **Push-off** (`pushoff`). Copy K_j as a blackboard-parallel component on
   the side of K_j that faces the band (the side named by `b`).
   - Original crossings keep their indices.
   - A crossing of K_j with another strand gains one crossing; a self-crossing
     gains three (a 2×2 grid). New crossings copy the original's sign and slot
     roles.
   - Which end of a strand the new crossing sits at follows from the side port
     (`prev_ccw`/`next_ccw` of the strand's in-port).
2. **Twists.** Insert `f_j − w_j` full twists (braid σ^±2) between K_j and the
   push-off on the edge after `b`'s edge. Their linking number is then `f_j`.
3. **Band.** Reconnect K_i's edge at `a` with the push-off's edge beside `b`,
   using the same west/east pairing as `R2_ADD`. If the orientations disagree,
   reverse the push-off first (`reverse_component`): that is handle
   subtraction, ε = −1. The band is untwisted, so ε is fixed by the sides; a
   kink on K_i gives the other sign.
4. **Labels and framing.** The push-off takes K_i's label, and
   `f_i += f_j + 2·ε·lk(K_i, K_j)`. Undoing a slide is another slide, but only
   up to isotopy, so `apply_inplace` returns `None` for slides.

## Verification (`tests/test_kirby.py`, plus the session checks below)

- **Push-off** equals Regina's `parallel(2, Framing.Blackboard)` exactly, with
  strict `sig()`, on random knot diagrams and on both sides (32/32). Its
  linking number with the original equals the writhe.
- **Every legal slide** on the framed Whitehead link, Borromean rings and
  L4a1:
  - gives a valid planar diagram (`check()`);
  - its new linking matrix is exactly `E Q Eᵀ` for an elementary row/column
    operation;
  - H₁ is unchanged.
- **SnapPy** finds an isometry before/after for all 152 slides of four
  hyperbolic surgeries (Whitehead (5,2), L6a2 (5,5), Borromean (−3,5,5),
  L7a1 (−3,−3)), with homology unchanged. The test suite checks every fifth
  slide.
- **Framing convention.** +1 surgery on the trefoil has trivial H₁; the
  blackboard convention would give Z/4.

## 4-manifolds: `Handlebody` (`handlebody.py`)

- **Data.** `LinkDiagram` + `{label: framing}` for 2-handles + a set of dotted
  labels (1-handles, Akbulut's dotted-circle notation). Dotted circles must
  form an unlink: the constructor checks pairwise linking numbers, and every
  move preserves the unlink. `closed=True` adds implicit 3- and 4-handles.
- **Invariants.**
  - χ = 1 − #1-handles + #2-handles.
  - The boundary map ∂ (rows: dotted circles, columns: 2-handles) has entries
    lk(K, D).
  - H₁(X) = coker ∂, by Smith normal form.
  - The intersection form is Bᵀ L B. Here B is a Z-basis of ker ∂ (from
    unimodular column reduction) and L is the 2-handles' linking matrix with
    framings on the diagonal.
  - Its rank, signature, nullity, |det| of the nondegenerate part, and parity.
  - ∂X = surgery with dots → 0-framed (`boundary()`, `to_snappy_boundary()`).
  - `invariants()` collects these. Without 2/3 moves (`closed=False`) it also
    includes χ and H₁(∂X).
- **Moves.**

| Move | Legal when | Effect |
|---|---|---|
| Reidemeister | as in `core.moves` | isotopy |
| `HANDLE_SLIDE(a, b)` | the `R2_ADD` face rule, and: 2-handle over 2-handle; 2-handle over dotted circle; dotted over dotted only if no crossing has two dotted strands | `band_slide`, with framing 0 for a dotted circle; f_i updated only for a 2-handle |
| `CANCEL_12(a)` | dotted `a` has exactly two crossings, both with one 2-handle K, over at one and under at the other, joined by an edge of K | remove `a` and K (`remove_components`) |
| `CREATE_12(sign=n)` | n ∈ {−1, 0, 1} | insert a split Hopf link: dotted circle + n-framed unknot |
| `CANCEL_23(a)` / `CREATE_23()` | `closed=True`; `a` a crossingless 0-framed 2-handle | remove / add it |

### Listing moves and complexity

- **Per-state cache.** `Handlebody._state()` makes one O(n) pass per diagram
  state, keyed by `LinkDiagram.stamp`, which changes on every mutation. It
  records the dotted flag of each port, whether the dotted circles are flat,
  and the cancelling partner of each dotted circle. Every `is_legal` is then
  O(1).
- **`legal_moves`** is `list(iter_legal_moves)`, O(n + output).
- **Slides** group each face's corners by component, so same-component pairs
  are skipped wholesale: O(n + output + Σ_f |f|·#components at f).
- **`count_legal_moves`** is O(n), summing bucket sizes; no pair is
  enumerated.
- **`move_masks`** is O(n): the diagram's masks plus per-side label and dotted
  flags. A slide (x, y) is legal iff the labels differ, the sides share a face
  (or lie in different pieces), and the dotted rule holds.

Measurements (Python 3.12; dotted L7a6 grown to 220 crossings with 60 extra
cancelling pairs, 939,921 legal moves):

| Operation | Time / memory |
|---|---|
| `CANCEL_12` legality check | 0.15 µs (6.7 µs before, as an O(n) scan) |
| `count_legal_moves` | 0.68 ms |
| `move_masks` | 0.32 ms |
| `iter_legal_moves` (all moves) | 0.12 MB peak |
| `legal_moves` (all moves) | 552 ms, 84 MB |

### Why dotted-over-dotted slides need the "flat" condition

The band of `band_slide` crosses nothing in the full diagram. But the spanning
disks of the dotted circles need not lie flat, so the band can thread another
dotted circle's disk.

A random walk found exactly this. The dotted circles were two components of
the Borromean rings, crossing each other. After one dotted-over-dotted slide
they no longer formed an unlink:
- before the slide, SnapPy gave a free fundamental group on 2 generators;
- after it, the dotted sublink simplified only to 10 crossings, and its group
  had 3 generators and 2 relators.

**The fix.** Allow such slides only when no crossing has two dotted strands.
Then the dotted circles are disjoint planar curves with flat disks:
- the push-off of the circle slid over has no self-crossings and needs no
  twists;
- the band is planar;
- so the result is again crossing-free among dotted circles, a certified unlink.

Reidemeister moves remain unrestricted and can untangle the dotted circles
first. Slides of 2-handles over dotted circles are isotopies in ∂(0- and
1-handles) and need no condition.

### Cancellation

`CANCEL_12` is recognised locally. The edge of K joining the dotted circle's
two crossings bounds, together with the circle's two edges, two bigons.
Nothing else can enter that side, so the circle bounds a disk pierced once,
by K only. With nothing else through the 1-handle, the pair cancels and the
other handles are unchanged; in 3D this is a slam dunk, giving (K, f − 1/0) =
(K, ∞). K may be knotted and linked with other 2-handles.

Strands that pass through a dotted circle must be slid off over K first.

### Verification (`tests/test_handlebody.py`, plus the session checks below)

- **Basic examples:** S¹×B³, the D²-bundle with Euler number n, the cancelling
  pair (B⁴), and the chain dot–K–L (cancels to L).
- **All 77 slides of 2-handles over dotted circles** on three contractible
  Mazur-type examples (L7a6, L8a8, L8a9 with one dotted component):
  - invariants unchanged;
  - SnapPy isometry of the hyperbolic boundaries.
- **Every dotted-over-dotted slide** (18) on three flat configurations:
  invariants unchanged, and the dotted sublink stays a certified unlink.
- **Random walks** (open and closed) with slides, 1/2 and 2/3 creations and
  cancellations:
  - invariants unchanged;
  - dotted circles stay unlinked (linking numbers checked every step; unlink
    certified by simplification or a free π₁).

## Next

- Non-split blow-downs (Rolfsen twists) and general 1-handle slides (bands
  certified not to thread dotted disks) as macros.
- Recognising cancelling pairs beyond the local meridian pattern.
- RL environment: done in `src/knotml/envs/kirby_env.py` (see README section 9);
  still open: the reward, start-state distributions beyond `ScrambleStart`
  (e.g. known B⁴/S³ diagrams), and policies (GNN over ports/faces).
