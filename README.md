# KnotRL

A reinforcement learning framework for learning how to unknot mathematical knots (and eventually learn other knot invariants).

## Getting Started

### Install

```bash
python -m pip install -e ".[dev]"            # core + tests
python -m pip install -e ".[dev,oracles]"    # + SnapPy, Regina, knot Floer homology, KnotInfo
```

### Run tests

```bash
python -m pytest
```

Tests that need the optional packages are skipped when those are missing.

### Quick start

```python
from knotml import LinkDiagram, MoveKind, determinant

d = LinkDiagram.from_pd([[1, 5, 2, 4], [3, 1, 4, 6], [5, 3, 6, 2]])  # trefoil
moves = d.legal_moves()             # every legal Reidemeister move, each listed once
undo = d.apply_inplace(moves[0])    # O(1); returns the inverse move
d.apply_inplace(undo)
masks = d.move_masks()              # O(n) numpy legality masks for all move kinds
key = d.canonical_key()             # hashable key of the diagram up to relabelling
```

`python scripts/inspect_diagram.py --random 100 --bench` prints a diagram's moves and timings.

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
  envs/          Gymnasium environment (awaiting the action-space design)
  agents/        baseline agents
planning/        research and design notes
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
4. Implement visualization methods
5. Kirby calculus: `FramedLink` with framings, blow-ups and handle slides