# Framed links and Kirby moves: design notes

*2026-10-10. Implements item 2 of section 3 in `research-summary.md`
(`src/knotml/kirby.py`). Dotted circles and handle cancellations are not
implemented yet.*

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

`calculus="4-manifold"` drops blow-ups and blow-downs, which change the
4-manifold by #±CP². By Kirby's theorem the full set is complete for
3-manifolds, and slides plus isotopy are complete for 2-handlebodies. Split
blow-downs suffice: a ±1 unknot linked with other strands can first be freed
by sliding those strands off it.

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

## Next

- Dotted circles (1-handles), 1/2- and 2/3-handle cancellation, and slides of
  1-handles over 1-handles.
- Non-split blow-downs (Rolfsen twists) as macros.
- An action space for Kirby moves. Slides use the same factored face/piece
  masks as `R2_ADD`, plus a label-difference mask.
