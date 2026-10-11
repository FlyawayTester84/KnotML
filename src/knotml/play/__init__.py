"""Play a Kirby environment by hand: terminal commands, with a static HTML page that follows along.

* :class:`PlaySession`: an episode with undo, previews and a record.
* :class:`Console`: command lines -> session calls (``step 12``, ``undo``, ...).
* :func:`main`: the command-line program (``python scripts/play_env.py``).
"""

from .console import Console, main
from .page import render_page, write_page
from .session import PlaySession, Record, example_reward, load_reward

__all__ = ["PlaySession", "Record", "Console", "main", "render_page", "write_page", "example_reward", "load_reward"]
