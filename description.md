# Description
This file contains implementing details of the project. 

## Representations of knots
There are various ways to represent a knot in current literature. 
1. Knots are recorded by the crossing data. After orienting the knot, each crossing contains 4 pieces of data: the index of the crossing, the sign of the crossing, the of the previous crossing (counted cyclicly), the following crossing (counted cyclicly), and its sibling crossing (the same crossing on the other strand).
2. Adjacent matrices of dual graphs. This is an $N\times N$ matrix whose entries are 0 or $\pm 1$, recording the sign of each edge of crossing. 
3. Gauss code
4. 3D coordinates
5. As closure of a braid. The braid can be represented by its braid word as a list. 

The ``Knot`` class in `utils/knot.py` implements various operations on knots.
1. 3 Reidemeister moves
2. Jones polynomial
3. Khovanov homology 

``utils/visualization.py`` implements visualizations of knots from the dual graph.
1. Visualization as a knot diagram
2. Visualization in 3D

## Environments

### Unknotting

## RL Algorithm