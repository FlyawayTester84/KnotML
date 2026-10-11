"""Your reward function: edit :func:`my_reward` below.

Select it by the name ``my_reward``:

* training: ``reward: {name: my_reward}`` in ``configs/ppo.yaml``, or
  ``python scripts/train_agent.py --set reward.name=my_reward``;
* the play console: ``python scripts/play_env.py --reward my_reward``;
* code: ``KirbyEnv(reward_fn=make_reward("my_reward"))``.

Constants can live in the config instead of here: ``reward.kwargs`` arrives as
keyword arguments, e.g. ``reward: {name: my_reward, kwargs: {bonus: 2.0}}``
calls ``my_reward(t, bonus=2.0)``.

What a transition ``t`` holds:

* ``t.before``: the state before the step (a copy), a ``FramedLink`` or ``Handlebody``.
* ``t.move``: the move taken (``Move`` / ``KirbyMove``), or ``None`` if the action named nothing.
* ``t.after``: the state after the step (the live object: read it, do not change it).
* ``t.legal``: whether the move was applied (an illegal action changes nothing).
* ``t.terminated``: whether ``t.after`` is the empty diagram, the goal.

Useful quantities of a state ``s``: ``s.diagram.n_crossings``, ``len(s.labels())``
(components), ``s.is_empty()``, ``s.framing(label)``, ``s.diagram.writhe()``; for a
``Handlebody`` also ``s.n_1handles``, ``s.n_2handles`` and ``s.dotted_labels()``.
The move's kind is ``t.move.kind`` (e.g. ``KirbyKind.CANCEL_12``, ``MoveKind.R1_REMOVE``).

Episodes cut off at ``max_steps`` need no special case: the trainer bootstraps them.
"""

from __future__ import annotations

from .kirby_env import Transition


def my_reward(t: Transition, **params) -> float:
    """The reward for one step.  Returns 0 until you fill it in.

    For example, +1 for solving and -0.01 per step otherwise::

        if t.terminated:
            return params.get("bonus", 1.0)
        return -0.01
    """
    return 0.0
