"""The static page of a play session, rewritten after every command.

The page polls a one-line ``<name>.version.js`` next to it (a ``<script>`` tag
works from ``file://``, where ``fetch`` does not) and reloads when the version
changes.  It keeps its scroll position and open sections across reloads.
"""

from __future__ import annotations

import html
import json
import os
import time

import numpy as np

from ..handlebody import Handlebody
from ..notation import describe_move, move_crossings, move_labels, move_spec
from ..visualization.pages import THEME_BUTTON, THEME_JS, component_notes, invariants_html, page_head, write_text
from ..visualization.svg import to_svg
from .session import PlaySession, components_of

__all__ = ["render_page", "write_page", "version_path", "observation_text"]

COMMANDS = [
    ("step N", "take legal move N (or just type N)"),
    ("step SPEC", "take a move written as in the tables, e.g. step HANDLE_SLIDE 14 26 (or just type SPEC)"),
    ("action K A B P", "take a raw action vector, legal or not"),
    ("preview N | SPEC", "draw the result beside the current diagram without stepping (preview off: hide)"),
    ("random [N] [KIND]", "take N random legal moves (of one kind)"),
    ("moves [KIND]", "list the legal moves in the terminal"),
    ("undo [N]", "take back the last N steps"),
    ("reset [SEED]", "start a new episode"),
    ("obs", "print the observation arrays"),
    ("save [PATH]", "write the episode as a step-through page"),
    ("open | help | quit", ""),
]

_CSS = """
  body { max-width: 1500px; }
  h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 22px 0 8px; }
  .status { display: flex; flex-wrap: wrap; gap: 6px 18px; font-variant-numeric: tabular-nums; }
  .status b { font-weight: 600; }
  .bar { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; }
  .grid { display: grid; grid-template-columns: minmax(320px, 560px) minmax(0, 1fr); gap: 24px; margin-top: 16px; }
  .grid.has-preview { grid-template-columns: minmax(0, 1.5fr) minmax(320px, 1fr); }
  .boards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .left { position: sticky; top: 12px; align-self: start; max-height: calc(100vh - 24px); overflow: auto; }
  @media (max-width: 900px) {
    .grid, .grid.has-preview { grid-template-columns: 1fr; }
    .left { position: static; max-height: none; }
  }
  figcaption { font-weight: 400; } figcaption b { font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }
  th { color: var(--ink-2); font-weight: 600; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  td code { white-space: nowrap; } .right > h2:first-child { margin-top: 0; }
  tr.move { cursor: pointer; } tr.move:hover, tr.move:focus { background: var(--rule); outline: none; }
  details.kind > summary { cursor: pointer; padding: 4px 0; color: var(--ink); font-weight: 600; }
  .more { color: var(--ink-2); font-size: 13px; padding: 6px 8px; }
  .commands td:first-child { white-space: nowrap; }
  pre { font-size: 12px; overflow: auto; }
  #toast { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); background: var(--ink);
           color: var(--page); padding: 6px 12px; border-radius: 6px; font-size: 13px; opacity: 0;
           transition: opacity .2s; pointer-events: none; }
  #toast.on { opacity: 1; }
"""

_JS = """
  const page = JSON.parse(document.getElementById('page-data').textContent);
  const key = 'knotml-play:' + location.pathname;
  try {  // restore scroll position and open sections after a reload
    const saved = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (saved) {
      for (const [id, open] of Object.entries(saved.open)) {
        const el = document.getElementById(id);
        if (el) el.open = open;
      }
      window.scrollTo(0, saved.y);
    }
  } catch (e) {}
  function save() {
    try {
      const open = {};
      document.querySelectorAll('details[id]').forEach(d => open[d.id] = d.open);
      sessionStorage.setItem(key, JSON.stringify({y: window.scrollY, open}));
    } catch (e) {}
  }
  (function poll() {
    const s = document.createElement('script');
    s.src = page.versionFile + '?' + Date.now();
    s.onload = () => {
      s.remove();
      if (window.knotmlVersion !== undefined && window.knotmlVersion !== page.version) { save(); location.reload(); }
      else setTimeout(poll, 600);
    };
    s.onerror = () => { s.remove(); setTimeout(poll, 2000); };
    document.head.appendChild(s);
  })();
  function mark(row, on) {
    const fig = document.getElementById('current');
    for (const c of row.dataset.c.split(' ').filter(Boolean))
      fig.querySelectorAll('.spot[data-crossing="' + c + '"]').forEach(e => e.classList.toggle('on', on));
    for (const k of row.dataset.k.split(' ').filter(Boolean))
      fig.querySelectorAll('g[data-label="' + k + '"]').forEach(e => e.classList.toggle('on', on));
  }
  const toast = document.getElementById('toast');
  function copied(text) {
    toast.textContent = 'copied: ' + text;
    toast.classList.add('on');
    setTimeout(() => toast.classList.remove('on'), 1200);
  }
  function copy(text) {
    const fallback = () => {
      const t = document.createElement('textarea'); t.value = text; document.body.appendChild(t); t.select();
      try { document.execCommand('copy'); copied(text); } catch (e) {} t.remove();
    };
    if (navigator.clipboard && window.isSecureContext)
      navigator.clipboard.writeText(text).then(() => copied(text), fallback);
    else fallback();
  }
  document.querySelectorAll('tr.move').forEach(row => {
    row.onmouseenter = row.onfocus = () => mark(row, true);
    row.onmouseleave = row.onblur = () => mark(row, false);
    row.onclick = () => copy(row.dataset.cmd);
    row.onkeydown = e => { if (e.key === 'Enter') copy(row.dataset.cmd); };
  });
"""


def version_path(path: str) -> str:
    """The version file a page at ``path`` polls."""
    root, _ = os.path.splitext(path)
    return root + ".version.js"


def write_page(session: PlaySession, path: str, max_rows: int = 200) -> None:
    """Write the page and bump its version, so an open copy reloads."""
    version = time.time_ns()
    write_text(path, render_page(session, os.path.basename(version_path(path)), version, max_rows))
    write_text(version_path(path), f"window.knotmlVersion = {version};\n")


def _drawing(obj, marked, spots: bool) -> str:
    if obj.is_empty():
        return '<p class="hint">(the empty diagram)</p>'
    dotted, notes = component_notes(components_of(obj))
    return to_svg(obj.diagram, size=520, labels=True, highlight=marked, dotted=dotted, notes=notes, style=False,
                  spots=spots)


def _invariants(session: PlaySession, obj) -> str:
    calculus = "4" if isinstance(obj, Handlebody) else "3"
    return invariants_html(obj.diagram, components_of(obj), calculus, closed=session.env.closed)


def render_page(session: PlaySession, version_file: str, version: int, max_rows: int = 200) -> str:
    env, obj = session.env, session.state
    moves = session.moves()
    last = session.records[-1] if session.records else None
    d = obj.diagram

    if last and last.terminated:
        state = "solved: empty diagram"
    elif last and last.truncated:
        state = f"truncated at {env.max_steps} steps"
    else:
        state = "running"
    status = [
        f"<span>step <b>{env.steps}</b> / {env.max_steps}</span>",
        f"<span>last reward <b>{last.reward:+.3g}</b></span>" if last else "<span>last reward &ndash;</span>",
        f"<span>return <b>{session.total_return:+.3g}</b></span>",
        f"<span><b>{d.n_crossings}</b> / {env.max_crossings} crossings</span>",
        f"<span><b>{len(obj.labels())}</b> / {env.max_components} components</span>",
        f"<span><b>{len(moves)}</b> legal moves</span>",
        f"<span>status: <b>{state}</b></span>",
    ]
    lines = []
    if last:
        verdict = "legal" if last.legal else "illegal, state unchanged"
        lines.append(f"last step: <code>{html.escape(last.label)}</code> &middot; action {last.action.tolist()} "
                     f"&middot; {verdict}")
    if session.done:
        lines.append("<b>The episode is over</b>: <code>undo</code> or <code>reset</code> in the terminal.")
    if session.scramble:
        lines.append("start: scramble of " + ", ".join(f"<code>{html.escape(move_spec(m))}</code>"
                                                       for m in session.scramble))
    elif session.start is not None:
        lines.append("start: the given diagram")

    title = "empty diagram" if obj.is_empty() else f"{d.n_crossings} crossings"
    boards = [
        f'<figure id="current"><figcaption><b>Current</b> &middot; {title} &middot; {_invariants(session, obj)}'
        f'</figcaption>{_drawing(obj, last.marked if last else set(), spots=True)}</figure>'
    ]
    if session.preview is not None:
        move, after, marked = session.preview
        spec = html.escape(move_spec(move))
        if after is None:
            boards.append(f"<figure><figcaption><b>Preview</b> &middot; <code>{spec}</code> is not legal here"
                          "</figcaption></figure>")
        else:
            boards.append(
                f"<figure><figcaption><b>Preview</b> &middot; <code>{spec}</code> &rarr; "
                f"{after.diagram.n_crossings} crossings &middot; {_invariants(session, after)}</figcaption>"
                f"{_drawing(after, marked, spots=False)}</figure>"
            )

    by_kind: dict = {}
    for i, m in enumerate(moves):
        by_kind.setdefault(m.kind, []).append((i, m))
    sections = []
    for kind in env.kinds:
        rows = by_kind.get(kind, [])
        if not rows:
            continue
        body = []
        for i, m in rows[:max_rows]:
            spec = move_spec(m)
            cs = " ".join(map(str, sorted(move_crossings(d, m))))
            ks = " ".join(map(str, sorted(move_labels(obj, m))))
            body.append(
                f'<tr class="move" tabindex="0" data-cmd="step {i}" data-c="{cs}" data-k="{ks}">'
                f'<td class="num">{i}</td><td><code>{html.escape(spec)}</code></td>'
                f"<td><code>{env.encode(m).tolist()}</code></td><td>{html.escape(describe_move(obj, m))}</td></tr>"
            )
        more = (f'<div class="more">&hellip; {len(rows) - max_rows} more: <code>moves {kind.name}</code> '
                "in the terminal</div>" if len(rows) > max_rows else "")
        sections.append(
            f'<details class="kind" id="k-{kind.name}"{" open" if len(rows) <= 40 else ""}>'
            f"<summary>{kind.name} ({len(rows)})</summary><table><thead><tr><th class=num>#</th><th>move</th>"
            f"<th>action</th><th>what it does</th></tr></thead><tbody>{''.join(body)}</tbody></table>{more}</details>"
        )
    if not sections:
        sections.append('<p class="hint">No legal moves.</p>')

    history = ['<tr><td class="num">0</td><td>start</td><td></td><td></td><td class="num"></td>'
               f'<td class="num">{session.states[0].diagram.n_crossings}</td></tr>']
    for k, (record, after) in enumerate(zip(session.records, session.states[1:], strict=True), 1):
        history.append(
            f'<tr><td class="num">{k}</td><td><code>{html.escape(record.label)}</code></td>'
            f"<td><code>{record.action.tolist()}</code></td><td>{'yes' if record.legal else 'no'}</td>"
            f'<td class="num">{record.reward:+.3g}</td><td class="num">{after.diagram.n_crossings}</td></tr>'
        )

    data = json.dumps({"version": version, "versionFile": version_file})
    commands = "".join(f"<tr><td><code>{html.escape(c)}</code></td><td>{html.escape(t)}</td></tr>" for c, t in COMMANDS)
    calculus = f"{env.calculus}{' (closed)' if env.closed else ''}"
    return f"""{page_head("Kirby environment", _CSS)}
<div class="bar"><h1>KirbyReduce &middot; {calculus}</h1>{THEME_BUTTON}</div>
<div class="status">{"".join(status)}</div>
{"".join(f'<p class="hint">{line}</p>' for line in lines)}
<div class="grid{" has-preview" if session.preview is not None else ""}">
<div class="left"><div class="boards">{"".join(boards)}</div>
<p class="hint">Hover a move to circle its crossings and thicken its components; click it to copy its
<code>step</code> command. Type commands in the terminal; this page reloads when the state changes.</p></div>
<div class="right">
<h2>Legal moves</h2>{"".join(sections)}
<h2>Observation</h2>{_observation_html(session)}
<h2>History</h2><table><thead><tr><th class=num>step</th><th>move</th><th>action</th><th>legal</th>
<th class=num>reward</th><th class=num>crossings</th></tr></thead><tbody>{"".join(history)}</tbody></table>
<details id="commands"><summary>Terminal commands</summary><table class="commands">{commands}</table></details>
</div></div>
<div id="toast" role="status"></div>
<script type="application/json" id="page-data">{data}</script>
<script>{_JS}{THEME_JS}</script></body></html>
"""


def _used(obs: dict) -> dict:
    """The observation arrays cut to the crossings and components that exist (the rest is padding)."""
    n, c = int(obs["n_crossings"][0]), int(obs["n_components"][0])
    cut = {"nbr": 4 * n, "sign": n, "strand_component": 2 * n, "side_face": 4 * n, "side_piece": 4 * n,
           "component_type": c, "framing": c, "crossingless": c}
    out = {k: (v[: cut[k]] if k in cut else v) for k, v in obs.items()}
    out["linking"] = obs["linking"][:c, :c]
    return out


def observation_text(obs: dict) -> str:
    """The used part of each observation array, one per line (for the terminal)."""
    lines = []
    for k, v in _used(obs).items():
        text = np.array2string(np.asarray(v), max_line_width=110, threshold=400)
        lines.append(f"{k} {tuple(obs[k].shape)}: {text}")
    return "\n".join(lines)


def _observation_html(session: PlaySession) -> str:
    obs = _used(session.obs)
    labels = session.state.labels()
    kinds = {0: "absent", 1: "2-handle" if session.env.calculus == "4-manifold" else "framed", 2: "1-handle (dotted)"}
    rows = "".join(
        f'<tr><td class="num">{k}</td><td>K{lab}</td><td>{kinds[int(obs["component_type"][k])]}</td>'
        f'<td class="num">{"" if obs["component_type"][k] == 2 else int(obs["framing"][k])}</td>'
        f'<td>{"yes" if obs["crossingless"][k] else ""}</td></tr>'
        for k, lab in enumerate(labels)
    )
    table = ('<table><thead><tr><th class=num>slot</th><th>label</th><th>type</th><th class=num>framing</th>'
             f"<th>crossingless</th></tr></thead><tbody>{rows}</tbody></table>") if labels else ""
    link = ""
    if labels:
        head = "".join(f"<th class=num>K{lab}</th>" for lab in labels)
        body = "".join(
            f"<tr><th>K{lab}</th>" + "".join(f'<td class="num">{int(x)}</td>' for x in row) + "</tr>"
            for lab, row in zip(labels, obs["linking"], strict=True)
        )
        link = (f'<p class="hint">Linking matrix (framings on the diagonal, 0 for dotted circles):</p>'
                f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>")
    raw = html.escape(observation_text(session.obs))
    return (f"{table or '<p class=hint>No components.</p>'}{link}"
            '<details id="raw-obs"><summary>Raw observation (used part of each array)</summary>'
            f"<pre>{raw}</pre></details>")
