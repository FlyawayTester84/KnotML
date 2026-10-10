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
python -m pytest                            # whole suite: 45 passed, 1 skipped
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

### 4. Random walks

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

### 5. Visualize

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

From Python:

```python
from knotml.interop import from_knotinfo
from knotml.visualization import to_svg, view

svg = to_svg(from_knotinfo("3_1"), labels=True, highlight=[0])   # SVG string, no window
open("trefoil.svg", "w").write(svg)
view(from_knotinfo("3_1"))                                        # PLink window
```

### 6. Use the library

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

### 7. Benchmark

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
  visualization.py  spherogram drawings: SVG export and PLink viewer (optional deps)
  envs/          Gymnasium environment (awaiting the action-space design)
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
3. Choose the action space and observation encoder, then finish `UnknotEnv`
4. ~~Basic visualization~~ (`knotml/visualization.py`); layouts that stay continuous across moves
5. Kirby calculus: `FramedLink` with framings, blow-ups and handle slides