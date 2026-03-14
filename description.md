# Description
This file contains implementing details of the project. 

## Representations of knots
1. Knots are recorded by the crossing data. After orienting the knot, each crossing contains 3 pieces of data: the sign of the crossing, the previous crossing (counted cyclicly), and the following crossing (counted cyclicly).
2. Adjacent matrices of dual graphs. THis is an $N\times N$ matrix whose entries are 0 or $\pm 1$, recording the sign of each edge of crossing. 
3. Gauss code
4. 3D coordinates

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