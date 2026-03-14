# KnotRL

A reinforcement learning framework for learning how to unknot mathematical knots (and eventually learn other knot invariants).

## Getting Started

### Install

```bash
python -m pip install -e .
```

### Run tests

```bash
python -m pytest
```

## Goals

1. Provide a Gym-like environment that represents knots and possible unknotting moves (Reidemeister moves).
2. Provide a simple RL agent baseline for training.
3. Expand to compute and optimize other knot invariants (e.g. crossing number, polynomial invariants).

## TODOs
Also refer to ``description.md``. 
1. Determine what knot representation to use
2. Implement Reidemeister moves in ``utils/knot.py``
3. Implement visualization methods
4. Implement Jones polynomial and other invariants