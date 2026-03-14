# UnknotRL

A reinforcement learning framework for learning how to unknot mathematical knots (and eventually learn other knot invariants).

## Project structure

- `src/unknot_rl/` - core library code (environments, agents, utilities)
- `tests/` - unit tests

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
