# Description
This file contains implementing details of the project. 

## Representations of knots
There are various ways to represent a knot in current literature. 
1. Knots are recorded by the crossing data. After orienting the knot, each crossing contains 4 pieces of data: the index of the crossing, the sign of the crossing, the of the previous crossing (counted cyclicly), the following crossing (counted cyclicly), and its sibling crossing (the same crossing on the other strand).
2. Adjacent matrices of dual graphs. This is an $N\times N$ matrix whose entries are 0 or $\pm 1$, recording the sign of each edge of crossing. 
3. Gauss code
4. 3D coordinates
5. As closure of a braid. The braid can be represented by its braid word as a list. 

The ``LinkDiagram`` class in `src/knotml/core/diagram.py` stores a diagram as a PD code and implements:
1. The 3 Reidemeister moves in both directions, plus crossing changes (`core/moves.py`)
2. A canonical key up to relabelling (`core/canonical.py`) and the determinant (`core/invariants.py`)
3. Jones polynomial, Khovanov homology and other invariants via Regina/SnapPy (`oracles.py`)

See `planning/core-design.md` for the conventions and complexity.

``src/knotml/visualization.py`` will implement visualizations of knots from the dual graph.
1. Visualization as a knot diagram
2. Visualization in 3D

## Environments

### Unknotting

## RL Algorithm