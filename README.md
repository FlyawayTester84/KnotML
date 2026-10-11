# KnotRL

A reinforcement learning framework for learning how to unknot mathematical knots (and eventually learn other knot invariants).

## Getting Started

### 1. Set up a Python environment (one time)

The project needs Python 3.10 or newer. On macOS the system `python3` is 3.9, so the
simplest route is [uv](https://docs.astral.sh/uv/), which downloads a suitable Python for you:

```bash
cd KnotML
python3 -m pip install --user uv          # or: curl -LsSf https://astral.sh/uv/install.sh | sh
python3 -m uv venv --python 3.12 venv     # creates ./venv (gitignored) with Python 3.12
python3 -m uv pip install --python venv/bin/python -e ".[dev,oracles]"
source venv/bin/activate                  # afterwards `python` is the venv's Python 3.12
```

Run the install step before `source venv/bin/activate`: once the venv is active,
`python3` points at the venv, which does not contain uv.

- `-e` installs in editable mode, so edits under `src/knotml/` take effect without reinstalling.
- `dev` adds pytest. `oracles` adds SnapPy/spherogram, Regina, knot Floer homology and the
  KnotInfo database; it is needed for the cross-checks, certified unknot test, loading knots
  by name, and drawing. The core works without it (`-e ".[dev]"`).
- Without activating, prefix commands with `venv/bin/`, e.g. `venv/bin/python -m pytest`.
  `deactivate` leaves the venv.

### 2. Run the tests

```bash
python -m pytest                            # whole suite: 97 passed, 1 skipped
python -m pytest tests/test_moves.py -v     # one file, verbose
python -m pytest -k regina                  # tests whose name matches "regina"
```

Tests that need the optional packages are skipped when those are missing. `UnknotEnv`'s test
is skipped until the action-space design is settled.

`tests/test_against_regina.py` checks, on random diagrams, that every move's legality and
result agree exactly with Regina. To add a knot of your own to the parametrized tests, add
`"name": (pd_code, writhe_of_that_diagram, determinant)` to `KNOTS` in `tests/_util.py` and run
`python -m pytest -k name`.

### 3. Load a knot and inspect it

`scripts/inspect_diagram.py` loads a diagram and prints its PD code, faces, writhe,
determinant and how many legal moves of each kind it has. Give the knot in whichever form
you have:

```bash
python scripts/inspect_diagram.py --pd "[[1,5,2,4],[3,1,4,6],[5,3,6,2]]"   # PD code
python scripts/inspect_diagram.py --braid 1 -2 1 -2      # braid word: i / -i = sigma_i^(+1/-1)
python scripts/inspect_diagram.py --knotinfo 3_1          # KnotInfo name (knots up to 13 crossings)
python scripts/inspect_diagram.py --knotinfo "L2a1{1}"    # LinkInfo name (quote the braces)
python scripts/inspect_diagram.py --snappy K12n123        # SnapPy table name
python scripts/inspect_diagram.py --snappy "DT: [(4,6,2)]"  # DT code
```

KnotInfo names use an underscore (`3_1`, `8_20`), not a dot. Example output:

```
LinkDiagram(crossings=3, components=1)
PD: [(6, 4, 1, 3), (2, 6, 3, 5), (4, 2, 5, 1)]
faces: 5 | writhe: 3 | determinant: 3
legal moves: {'R1_ADD': 24, 'R2_ADD': 18, 'CROSSING_CHANGE': 3}
```

**PD convention** (the same as KnotInfo, KnotAtlas, SnapPy, Regina and Sage): each 4-tuple
lists the edge labels around one crossing, starting at the incoming under-strand and going
counterclockwise. Labels can be any integers, each used exactly twice. The exported PD
renumbers edges `1..2n` along each component, so labels may differ from the input.

**Draw your own knot:** run `python -m plink.app`, draw the diagram, choose
**Info → PD code**, and paste the code into `--pd`.

### 4. Apply moves step by step

Pick a link, look at it, list the legal moves of one kind, apply one, look again, repeat.

**Command line.** Nothing is random when you use `--move`, so re-running with the same moves rebuilds
the same diagram; each run adds one more `--move` (ports in `--list` refer to the diagram after the
moves before it):

```bash
# 1. pick a link and look at it
python scripts/inspect_diagram.py --knotinfo "L4a1{0}" --html outputs/work.html && open outputs/work.html
# 2. list the legal moves of a kind (each line is a ready-to-paste --move)
python scripts/inspect_diagram.py --knotinfo "L4a1{0}" --list R2_ADD
# 3. apply one, refresh the page, and list the options for the next move
python scripts/inspect_diagram.py --knotinfo "L4a1{0}" --move "R2_ADD 0 14" --list R1_ADD --html outputs/work.html
# 4. repeat: keep the earlier --move flags and append the next one
python scripts/inspect_diagram.py --knotinfo "L4a1{0}" --move "R2_ADD 0 14" --move "R1_ADD 3 -1" --html outputs/work.html
```

For Kirby diagrams add `--framings` (e.g. `--framings 5 2`, or `--framings dot -2` for a dotted circle)
to every run, and list or apply Kirby moves the same way (`--list HANDLE_SLIDE`,
`--move "HANDLE_SLIDE 14 26"`); `--verify` checks each step.

**Python script.** The diagram stays in memory, so there is nothing to re-run:

```python
import html
import os

from knotml import Move, MoveKind
from knotml.interop import from_knotinfo
from knotml.visualization import to_svg


def show(d, kind, path="outputs/work.html"):
    """Write the diagram, its PD code and the legal moves of one kind to an HTML page."""
    moves = "\n".join(repr(m) for m in d.legal_moves([kind]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"<html><body>{to_svg(d, size=520, labels=True)}"
                f"<pre>PD: {d.to_pd()}\n\nlegal {kind.name} moves:\n{html.escape(moves)}</pre></body></html>")


d = from_knotinfo("L4a1{0}")                    # 1. pick a link
show(d, MoveKind.R2_ADD)                        # 2. open outputs/work.html: drawing + R2_ADD options
d.apply_inplace(Move(MoveKind.R2_ADD, 0, 14))   # 3. pick one
show(d, MoveKind.R1_ADD)                        # 4. refresh the page; it now lists R1_ADD options
d.apply_inplace(Move(MoveKind.R1_ADD, 3, sign=-1))
show(d, MoveKind.R3)                            # ... and repeat
```

Run it with `python step_by_step.py`, or paste it into `ipython` and keep going interactively. For
Kirby diagrams use `FramedLink`/`Handlebody` (section 8): draw `F.diagram`, pass
`dotted=H.dotted_labels()` to `to_svg`, list `F.legal_moves([KirbyKind.HANDLE_SLIDE])`, and apply
`KirbyMove(KirbyKind.HANDLE_SLIDE, a=..., b=...)`. A move's undo is returned by `apply_inplace`, or
keep `d.copy()` before experimenting.

### 5. Random walks

`--random N` applies `N` random Reidemeister moves (chosen uniformly among all legal moves)
before printing; `--seed` picks the walk and `--trace` prints every move and the PD code after
it:

```bash
python scripts/inspect_diagram.py --knotinfo 3_1 --random 30                    # summary after 30 moves
python scripts/inspect_diagram.py --knotinfo 3_1 --random 30 --seed 5 --trace   # every step
mkdir -p outputs && python scripts/inspect_diagram.py --knotinfo 3_1 --random 30 --trace > outputs/trace.txt
```

Trace lines look like `step 3: R1_ADD(a=1, sign=+1) -> 8 crossings  PD [...]`. The move's
anchors refer to the diagram *before* that step: port `4c + k` is position `k` of crossing
`c`'s PD tuple. New crossings are appended at the end of the PD list. The determinant stays
fixed along a walk; the writhe does not (R1 moves change it).

Uniform sampling is dominated by crossing-increasing moves (R1/R2 additions make up ~95% of
legal moves on a large diagram), so walks mostly grow and rarely pick R3.

### 6. Visualize

Drawings use spherogram's orthogonal layout (needs the `oracles` extra):

```bash
python scripts/inspect_diagram.py --knotinfo 3_1 --random 30 --html outputs/trefoil_walk.html
open outputs/trefoil_walk.html              # macOS; step with the slider or arrow keys
python scripts/inspect_diagram.py --knotinfo 4_1 --view    # interactive PLink window (blocks until closed)
```

The HTML page has one frame per step: the move, crossing count and determinant, the drawing
(gaps mark undercrossings, numbers are crossing indices, circled crossings were created or
rearranged by that move) and the PD code. Each frame is laid out from scratch, so pictures can
jump between steps. `outputs/` is gitignored. `--view` opens SnapPy's PLink viewer (not covered
by the automated tests).

Components are styled by label, so a component looks the same at every step. Labels 0-2 get three
hues (blue, orange, aqua) that stay distinct in every pair, including under colour-vision
deficiency. Labels 3-11 reuse those hues with dashed, dash-dot and dotted lines. Labels from 12 on
are drawn in grey. With two or more components, each one is tagged `K<label>` beside its strand
and a legend sits under the drawing. Drawings follow the system's light/dark setting, and the
page's "Light / dark" button overrides it.

From Python:

```python
from knotml.interop import from_knotinfo
from knotml.visualization import to_svg, view

svg = to_svg(from_knotinfo("3_1"), labels=True, highlight=[0])   # SVG string, no window
open("trefoil.svg", "w").write(svg)
view(from_knotinfo("3_1"))                                        # PLink window
```

### 7. Use the library

```python
import random
from knotml import LinkDiagram, Move, MoveKind, ISOTOPY_KINDS, determinant
from knotml.oracles import is_unknot, jones_polynomial      # optional deps

K = LinkDiagram.from_pd([[4, 2, 5, 1], [8, 6, 1, 5], [6, 3, 7, 4], [2, 7, 3, 8]])  # figure-eight
K = LinkDiagram.from_braid([1, -2, 1, -2])                  # the same knot as a braid closure

moves = K.legal_moves()                    # every legal Reidemeister move, each listed once
L = K.apply(Move(MoveKind.R1_ADD, 0, sign=+1))   # new diagram with a kink; K unchanged
undo = L.apply_inplace(L.legal_moves([MoveKind.R1_REMOVE])[0])   # O(1), returns the inverse
masks = K.move_masks()                     # O(n) numpy legality masks for all move kinds
key = K.canonical_key()                    # hashable key of the diagram up to relabelling

# Scramble with random moves: the knot type is unchanged
rng = random.Random(0)
J = K.copy()
for _ in range(30):
    J.apply_inplace(J.random_move(rng, ISOTOPY_KINDS))
assert determinant(J) == 5 and jones_polynomial(J) == jones_polynomial(K)

# Greedy simplification: remove kinks and bigons when possible, otherwise try an R3
for _ in range(500):
    shrink = J.legal_moves([MoveKind.R1_REMOVE, MoveKind.R2_REMOVE])
    level = J.legal_moves([MoveKind.R3])
    if not shrink and not level:
        break
    J.apply_inplace(shrink[0] if shrink else rng.choice(level))
print(J.n_crossings, J.is_isomorphic(K))   # 4 True

print(is_unknot(K))                        # False, certified via knot Floer homology
```

Other entry points: `knotml.interop` (`to_regina`, `from_regina`, `to_spherogram`,
`from_spherogram`, `from_knotinfo`), `knotml.oracles` (`is_unknot`, `seifert_genus`,
`jones_polynomial`, `alexander_polynomial`, `signature`), and on `LinkDiagram`: `to_pd`,
`to_gauss`, `faces`, `writhe`, `linking_matrix`, `component_writhes`.

**Moves and anchors.** Port `4c + k` is slot `k` of crossing `c` (slot 0 = incoming
under-strand, counterclockwise). A port also names one side of its edge, so every move is
anchored at ports:

| Move | Anchor | Legal when |
|---|---|---|
| `R1_REMOVE` | corner inside a monogon | the face is a monogon |
| `R1_ADD` | edge side + `sign` | always |
| `R2_REMOVE` | bigon side of its upper edge | bigon whose edge is over at both ends |
| `R2_ADD` | `a` = side pushed over, `b` = side pushed under | the two sides share a face |
| `R3` | triangle side of its top edge | triangle with an edge over at both ends |
| `CROSSING_CHANGE` | crossing index | always (not an isotopy) |

Legality checks are O(1), `apply_inplace` is O(1). See `planning/core-design.md` for the
details and measured timings.

### 8. Kirby calculus

There are two setups, sharing the handle-slide construction:

| | 3-manifolds: `FramedLink` (`knotml/kirby.py`) | 4-manifolds: `Handlebody` (`knotml/handlebody.py`) |
|---|---|---|
| Components | framed knots | dotted circles (1-handles) and framed knots (2-handles) |
| Describes | integral surgery on S³ | X = B⁴ ∪ 1-handles ∪ 2-handles (closed up with `closed=True`) |
| Kirby moves | `BLOW_UP`, `BLOW_DOWN`, `HANDLE_SLIDE` (`SURGERY_KINDS`) | `HANDLE_SLIDE`, `CANCEL_12`, `CREATE_12`, `CANCEL_23`, `CREATE_23` (`HANDLEBODY_KINDS`) |
| Invariants | linking matrix, H₁, signature, `to_snappy()` | χ, H₁(X), intersection form, signature, `boundary()` (∂X), `to_snappy_boundary()` |

Framings are framing coefficients (linking number with the framing push-off), not blackboard
framings, so Reidemeister moves never change them; both setups also allow every Reidemeister move.

#### 8a. 3-manifolds: `FramedLink`

```python
from knotml import FramedLink, KirbyKind, KirbyMove
from knotml.interop import from_knotinfo

F = FramedLink(from_knotinfo("L5a1{0}"), [5, 2])     # Whitehead link; framings in component-label order
F.linking_matrix(), F.homology(), F.signature()      # [[5, 0], [0, 2]], [10], 2

slide = F.legal_moves([KirbyKind.HANDLE_SLIDE])[0]   # HANDLE_SLIDE(slide=0, over=10)
G = F.apply(slide)
G.framings, G.linking_matrix(), G.homology()         # {0: 5, 1: 7}, [[5, -5], [-5, 7]], [10]

H = G.apply(KirbyMove(KirbyKind.BLOW_UP, sign=-1))   # add a split (-1)-framed unknot
H.framings, H.homology(), H.signature()              # {0: 5, 1: 7, 2: -1}, [10], 1

F.to_snappy().is_isometric_to(G.to_snappy())         # True: same surgered 3-manifold (needs snappy)
```

Moves (`F.legal_moves()`, `F.apply(move)`, `F.apply_inplace(move)`):

| Move | Anchor | Effect |
|---|---|---|
| Reidemeister `Move`s | as for diagrams | isotopy; framings unchanged |
| `BLOW_UP` | `sign` = ±1 | add a split crossingless ±1-framed unknot |
| `BLOW_DOWN` | `a` = label | remove a split crossingless ±1-framed unknot |
| `HANDLE_SLIDE` | `a` = side on K_i, `b` = side on K_j, sharing a face | K_i becomes K_i # (push-off of K_j); f_i += f_j ± 2·lk(K_i, K_j) |

Components keep their labels through every move (`F.labels()`, `F.diagram.label_of_port(p)`).
`homology()` and `signature()` come from the linking matrix; `to_snappy()` gives the surgered
manifold. From the command line:

```bash
python scripts/inspect_diagram.py --knotinfo "L5a1{0}" --framings 5 2 --random 4 --seed 2 --trace
python scripts/inspect_diagram.py --knotinfo "L5a1{0}" --framings 5 2 --random 12 --seed 1 \
    --max-crossings 60 --verify --html outputs/kirby_walk.html && open outputs/kirby_walk.html
```

**Apply specific moves.** `--list KIND` prints every legal move of one kind, ready to paste,
with what its anchors refer to (component, crossing, slot, side of edge, face; for slides
whether the handle is added or subtracted). `--move "KIND args"` applies a move; repeat it to
apply a sequence in order (illegal moves are refused). A knot alone has nothing to slide over,
so blow up an unknot, push the knot's edge across it, then slide:

```bash
python scripts/inspect_diagram.py --knotinfo 4_1 --framings 5 --move "BLOW_UP -1" --list R2_ADD
python scripts/inspect_diagram.py --knotinfo 4_1 --framings 5 --move "BLOW_UP -1" --move "R2_ADD 0 -1" --list HANDLE_SLIDE
python scripts/inspect_diagram.py --knotinfo 4_1 --framings 5 --move "BLOW_UP -1" --move "R2_ADD 0 -1" \
    --move "HANDLE_SLIDE 10 22" --verify --trace --html outputs/fig8_kirby.html
```

Move arguments: `R1_ADD a sign`, `R2_ADD over under`, `R1_REMOVE a`, `R2_REMOVE a`, `R3 a`,
`CROSSING_CHANGE c`, `BLOW_UP sign`, `BLOW_DOWN label`, `HANDLE_SLIDE slide over`. Anchors are
ports (`4·crossing + slot`, matching the crossing numbers in the drawings); negative anchors
`-1, -2, …` name crossingless components. In Python the same is
`F.apply_inplace(KirbyMove(KirbyKind.HANDLE_SLIDE, a=10, b=22))`.

`--framings` (one per component, in label order) makes the diagram a framed link: the summary
adds the linking matrix, H₁, signature and legal Kirby moves, and `--random` steps include Kirby
moves (handle slides, blow-ups/downs). `--verify` checks every step: the diagram is valid, H₁ is
unchanged and, with SnapPy, the surgered manifold is isometric to the start (for hyperbolic
surgeries; on very large diagrams SnapPy may fail to find the hyperbolic structure, reported as
"inconclusive" rather than a failure). Each slide multiplies the crossing count, so
`--max-crossings N` restricts the walk to moves that do not add crossings once it reaches `N`
(a slide taken just below `N` can still overshoot). On the HTML page each component keeps its
style, the legend under each drawing gives the framings, and captions show H₁. See
`planning/kirby-design.md` for the construction and verification.

#### 8b. 4-manifolds: `Handlebody`

Components are dotted circles (`None`), which must form an unlink, or framed 2-handles. There are
no blow-ups; instead 1/2-handle pairs (and, with `closed=True`, 2/3-handle pairs) are created and
cancelled.

```python
from knotml import Handlebody, KirbyKind, KirbyMove, LinkDiagram
from knotml.interop import from_knotinfo

X = Handlebody(from_knotinfo("L7a6{0}"), [None, -2])  # dotted circle + a -2-framed 2-handle
X.euler_characteristic(), X.homology(), X.intersection_form(), X.boundary().homology()   # 1, [], [], []

slide = X.legal_moves([KirbyKind.HANDLE_SLIDE])[0]   # HANDLE_SLIDE(slide=0, over=26): 2-handle over the dot
Y = X.apply(slide)                                    # Y.components == {0: None, 1: -4}
Y.invariants() == X.invariants()                      # True
X.to_snappy_boundary().is_isometric_to(Y.to_snappy_boundary())   # True (needs snappy)

C = Handlebody(LinkDiagram.from_braid([1, 1, 2, 2]), {1: None, 0: 2, 2: -3})   # dot - K0 - K2 chain
C.legal_moves([KirbyKind.CANCEL_12])                  # [CANCEL_12(dotted=1)]: K1 is a meridian of K0
C.apply(KirbyMove(KirbyKind.CANCEL_12, a=1)).components   # {2: -3}
X.apply(KirbyMove(KirbyKind.CREATE_12, sign=0))       # adds a split dotted circle with a 0-framed meridian
```

| Move | Anchor | Legal when |
|---|---|---|
| `HANDLE_SLIDE` | `a` = side on the slid component, `b` = side on the other, sharing a face | 2-handle over 2-handle; 2-handle over a dotted circle (as over a 0-framed unknot); dotted circle over dotted circle only while no two dotted strands cross (keeps them an unlink). Never a dotted circle over a 2-handle |
| `CANCEL_12` | `a` = dotted label | the dotted circle is a meridian of one 2-handle: two crossings, both with it, over and under, joined by an edge of it |
| `CREATE_12` | `sign` = framing in {-1, 0, 1} | always: adds a split dotted circle with a framed meridian |
| `CANCEL_23` / `CREATE_23` | `a` = label / — | `closed=True`; removes / adds a split crossingless 0-framed 2-handle |

**Listing moves.** `X.legal_moves(kinds)` returns every legal move (Reidemeister and Kirby), each
once, in O(n + output) time; legality checks (`X.is_legal(move)`) are O(1) after one O(n) pass per
diagram state. Since the number of legal moves can grow quadratically (R2 moves and slides pair up
sides), there are three cheaper views, also on `FramedLink` and `LinkDiagram`:

| Method | Gives | Cost |
|---|---|---|
| `X.iter_legal_moves(kinds)` | the same moves, lazily | O(n) memory, O(1) per move |
| `X.count_legal_moves(kinds)` | `{kind: number of legal moves}` | O(n), nothing enumerated |
| `X.move_masks()` | legality as O(n) numpy arrays (slides factored by face, piece, component, dottedness) | O(n) |

`invariants()` collects what every legal move preserves (χ, H₁(X), rank, signature, nullity,
|det| and parity of the intersection form, H₁(∂X); fewer when `closed`). From the command line,
`dot` in `--framings` selects the 4-manifold setup (`--four` forces it, `--closed` closes it up),
and `--move`, `--list`, `--random`, `--verify` and `--html` work as above (dotted circles are
drawn with a dot):

```bash
python scripts/inspect_diagram.py --knotinfo "L7a6{0}" --framings dot -2 --list HANDLE_SLIDE
python scripts/inspect_diagram.py --knotinfo "L7a6{0}" --framings dot -2 --move "CREATE_12 0" --list CANCEL_12
python scripts/inspect_diagram.py --knotinfo "L7a6{0}" --framings dot -2 --random 10 --seed 2 \
    --max-crossings 50 --verify --html outputs/mazur_walk.html
```

Move arguments add `CANCEL_12 dotted`, `CREATE_12 framing`, `CANCEL_23 label`, `CREATE_23`.

### 9. RL environment: reduce a Kirby diagram to the empty diagram

`knotml.envs.KirbyEnv` (Gymnasium ≥ 1.4, registered as `knotml/KirbyReduce-v0`) starts from a Kirby
diagram and ends (`terminated`) when no components are left: S³ for `calculus="3-manifold"`
(`FramedLink`), B⁴ for `calculus="4-manifold"` (`Handlebody`; S⁴ with `closed=True`). The reward is
left open: pass `reward_fn(transition)`, which sees the state before, the move, the state after,
whether it was legal and whether the episode ended (the default returns 0).

```python
import gymnasium as gym
import knotml.envs                                   # registers knotml/KirbyReduce-v0
from knotml.envs import CandidateActions, KirbyEnv, ScrambleStart

def reward(t):                                       # your choice; t.before, t.move, t.after, t.legal, t.terminated
    return 1.0 if t.terminated else (-0.01 if t.legal else -0.1)

env = KirbyEnv(calculus="4-manifold", max_crossings=64, max_components=8,
               reward_fn=reward, start=ScrambleStart(n_moves=(2, 8)))
obs, info = env.reset(seed=0)                        # info["scramble"]: the moves that built the start
actions = env.legal_actions()                        # (L, 4) rows: kind, anchor a, anchor b, param
obs, r, terminated, truncated, info = env.step(actions[0])   # info["legal"], info["move"]

masked = CandidateActions(gym.make("knotml/KirbyReduce-v0"), max_actions=512)   # Discrete + action_masks()
```

- **Actions** are `MultiDiscrete([kinds, anchors, anchors, 3])`: `env.kinds[kind]`; anchors
  `0 … 4·max_crossings − 1` are ports (sides of edges), the rest are components in label order;
  `param` 0/1/2 means −1/0/+1 (sign of `R1_ADD`/`BLOW_UP`, framing of `CREATE_12`). `env.decode` and
  `env.encode` convert to and from moves. An illegal action leaves the state unchanged
  (`info["legal"] = False`); moves that would exceed `max_crossings`/`max_components` count as
  illegal (a slide's size is predicted exactly).
- **Legal actions**: `env.legal_actions()` / `env.legal_moves()` (O(n + output)), `env.is_legal(move)`
  (O(1)). `CandidateActions` wraps the env as `Discrete(max_actions)` over the current legal moves,
  with their encodings in `obs["candidates"]` and `action_masks()` for masked methods such as
  sb3-contrib's `MaskablePPO`; when there are more, simplifying moves are kept and the rest sampled.
- **Observation** (padded integer arrays): `nbr`, `sign`, `strand_component`, `side_face`,
  `side_piece`, `component_type` (framed/dotted), `framing`, `crossingless`, `linking` (linking
  matrix, framings on the diagonal), `n_crossings`, `n_components`. `env.kirby` is the live
  `FramedLink`/`Handlebody`.
- **Start states**: `reset(options={"diagram": H})` starts from your own diagram; otherwise
  `ScrambleStart` applies random construction moves to the empty diagram (blow-ups or pair
  creations, Reidemeister moves, slides; the kind is sampled first), so every start is solvable.
  Pass any `start(env, rng)` callable for other distributions.
- Episodes are truncated after `max_steps` (default 200). A step takes ~0.1 ms, listing all legal
  actions 1–3 ms, for diagrams of 30–100 crossings.

### 10. Benchmark

```bash
python scripts/inspect_diagram.py --knotinfo 3_1 --random 100 --bench
```

prints the time for listing legal moves, building masks, the canonical key, and apply/undo,
plus Regina's per-pair `hasR2` enumeration for comparison.

## Layout

```
src/knotml/
  core/          pure-Python PD core (no optional dependencies)
    diagram.py     LinkDiagram: storage, PD/braid constructors, faces, components
    moves.py       Move/MoveKind, O(1) legality checks, enumeration, masks, application
    canonical.py   canonical key up to relabelling
    invariants.py  determinant (Goeritz matrix)
    ports.py       port arithmetic
  interop.py     exact conversion to/from Regina, SnapPy/spherogram, KnotInfo
  oracles.py     certified unknot test, genus, Jones, Alexander, signature (optional deps)
  kirby.py       FramedLink (3-manifolds): framings, blow-ups/downs, handle slides, surgery invariants
  handlebody.py  Handlebody (4-manifolds): dotted circles, 2-handles, slides, 1/2 and 2/3 cancellations
  visualization.py  spherogram drawings: SVG export and PLink viewer (optional deps)
  envs/          Gymnasium environments: KirbyEnv (reduce a Kirby diagram to empty), UnknotEnv (stub)
  agents/        baseline agents
scripts/
  inspect_diagram.py  load, random-walk, trace, visualize and benchmark a diagram
tests/           pytest suite (cross-checked against Regina when installed)
planning/        research and design notes
outputs/         generated traces and drawings (gitignored)
```

## Goals

1. Provide a Gym-like environment that represents knots and possible unknotting moves (Reidemeister moves).
2. Provide a simple RL agent baseline for training.
3. Expand to compute and optimize other knot invariants (e.g. crossing number, polynomial invariants).

## TODOs
Also refer to ``description.md``. 
1. ~~Determine what knot representation to use~~ (PD codes; see `planning/core-design.md`)
2. ~~Implement Reidemeister moves~~ (`knotml/core/moves.py`)
3. ~~RL environment for Kirby diagrams~~ (`knotml/envs/kirby_env.py`; reward still open);
   `UnknotEnv` (unknotting) still awaits the same treatment
4. ~~Basic visualization~~ (`knotml/visualization.py`); layouts that stay continuous across moves
5. ~~Kirby calculus~~: 3-manifolds (`knotml/kirby.py`) and 4-manifolds with dotted circles and
   handle cancellations (`knotml/handlebody.py`)