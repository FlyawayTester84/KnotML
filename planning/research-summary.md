# Research summary: RL on knot diagrams (PD codes) and Kirby calculus

*Compiled 2026-10-10. Package facts were checked by installing and running
snappy 3.3.2, spherogram 2.4.1, regina 7.4.1, database_knotinfo 2026.10.5 and
knot_floer_homology 1.2.2 (Python 3.9, macOS). Literature facts were checked
against arXiv/GitHub; items marked **(unverified)** could not be confirmed.*

## TL;DR

- Located Reidemeister moves on PD-style diagrams already exist in **Regina**
  (`r1/r2/r3` + `hasR*`/`withR*`) and **SnapPy/spherogram**
  (`spherogram.links.simplify`). Regina's check/perform/copy API maps directly
  onto an RL action mask + transition function.
- **Nothing usable exists for Kirby calculus** as a library — that layer must be
  written from scratch.
- **No published work runs a GNN/transformer on PD codes with Reidemeister moves
  at chosen diagram locations**, and none does RL on general Kirby diagrams.
  That is this project's niche.
- Random "backtrack" unknots are too easy for training (greedy simplification
  solves 100%); use hard-unknot datasets instead.

---

## 1. Existing packages

| Package | Install | Relevant functionality |
|---|---|---|
| **Regina 7.4** | `pip install regina` | `Link.fromPD`/`pd()`, `fromGauss`, `fromDT`, `fromKnotSig`. Moves: `r1(crossing)` / `r1(arc, side, sign)`, `r2(crossing)` / `r2(upperArc, upperSide, lowerArc, lowerSide)`, `r3(crossing, side)`; each has `hasR*` (legality check) and `withR*` (returns new diagram, non-mutating). `sig()`: canonical diagram string up to relabelling (state hashing). `rewrite(height, …)` / `simplifyExhaustive()`: BFS over the Reidemeister graph (exact optimal paths for small diagrams). `graph().cells()`: faces. `parallel(k, Framing.Blackboard \| Framing.Seifert)`, `selfFrame()`, `writheOfComponent()`, `graft(arc, arc)` (band sum), `countTrivialComponents()`, `pdAmbiguous()`, `complement()` → `Triangulation3.isSphere()`. Invariants: Jones, HOMFLY, Alexander, bracket. C++ core, fast. |
| **SnapPy + spherogram** | `pip install snappy` | `Link(pd \| braid_closure= \| 'DT:…' \| name)`, `PD_code()`, `faces()`, `dual_graph()`, `simplify('basic'\|'level'\|'pickup'\|'global')`, `backtrack()`, `many_diagrams()`, `random_link()`. Low-level: `reidemeister_I`, `reidemeister_I_and_II`, `possible_type_III_moves`, `reidemeister_III`, `reverse_type_I(link, crossing_strand, label, hand)`, `reverse_type_II(link, c, d, …)` (two strands in a common face), `backtrack`. Bands: `add_band`, `ribbon_concordant_links` (Dunfield–Gong, arXiv:2512.21825). 3-manifolds: `exterior()` → `dehn_fill`, `homology`, `fundamental_group`, `identify`, `is_isometric_to`. Drawing: `view()` (plink). Invariants: Alexander, signature, determinant, linking matrix, Jones (needs Sage), `knot_floer_homology()`. |
| **knot_floer_homology** | `pip install knot_floer_homology` | Szabó's HFK program; gives Seifert genus ⇒ **rigorous unknot certificate**. Used by DeepMind to certify hard unknots. |
| **database_knotinfo** | `pip install database_knotinfo` | `link_list()`: 12,965 knots (≤13 crossings) + 4,188 links (≤11), with `pd_notation`, braid, DT, Gauss columns and many invariants. |
| **Sage** `sage.knots.link.Link` | conda / Sage install | Invariants (Jones, Khovanov incl. Khoca, HOMFLY, Alexander, signature), `braid()`, `get_knotinfo()`. **No Reidemeister-move API.** Heavy dependency — keep optional. |
| **KLO** (Swenton) | GUI binary | Full Kirby calculus GUI (dotted 1-handles, framed 2-handles, slides, blow-ups, cancellations). No source, no scripting API, no PD export. |
| **metaknight** ([mattuff/metaknight](https://github.com/mattuff/metaknight)) | GitHub | Python circle-dot Kirby diagrams with R-moves, handle slides, creation/annihilation. Unmaintained since 2018. |
| pyknotid, Topoly | pip | 3D-curve oriented (polymer knots); low relevance. |

### PD conventions (verified)

For 3_1, 4_1, 5_1, 7_4, loading KnotInfo's `pd_notation` into spherogram and
Regina gives identical writhes, and Regina's Jones polynomial (variable x = √t)
matches KnotInfo's column. So **KnotInfo, KnotAtlas, spherogram and Regina all
use: start at the incoming under-strand, go counterclockwise.** Sage switched to
the same convention in 2023 (Issue #35665).

Pitfalls:
- spherogram's `PD_code()` uses **0-based** labels by default; Regina rejects
  them. Use `PD_code(min_strand_index=1)`.
- PD codes **cannot represent crossingless components** — track their count
  separately (cf. Regina `countTrivialComponents()`).
- Components consisting only of over-crossings have **ambiguous orientation**
  (Regina `pdAmbiguous()`).
- The crossing sign must be computed from each component's own edge successor,
  not "label + 1 mod 2n" (breaks for links and at the wrap-around edge).
- Mastin's paper (cited in `knot.py`) uses signed (component, arc) labels, not
  the integer convention above. Use the KnotInfo integer convention for interop.

### Prior ML/RL work

| Work | State representation | Actions | Notes / code |
|---|---|---|---|
| Gukov–Halverson–Ruehle–Sułkowski, *Learning to Unknot* ([2010.16263](https://arxiv.org/abs/2010.16263)) | Braid words, σᵢ^±1 ↦ ±i, zero-padded to 2·N_max | 5+N_max: SmartCollapse, shift L/R, braid relations 1/2, conjugation by generator (no stabilization) | Reward −(word length); illegal −4N_max. TRPO best (also A3C, A2C, DQN, PPO). Also Reformer/Transformer unknot classifier. No public code. |
| Gukov–Halverson–Manolescu–Ruehle, *Searching for ribbons with ML* ([2304.09304](https://arxiv.org/abs/2304.09304)) | 4 channels of N×N dual-graph matrices (signed adjacency, component map, band route, twists); N = 100 or 20 | Band as self-avoiding walk in dual graph: start, over/under (N each), twist ±, end; illegal moves masked | After each band SnapPy simplifies; win = unlink. A3C (+TRPO, AlphaZero-style, BO random walker). [ruehlef/ribbon](https://github.com/ruehlef/ribbon) is a compiled binary ("still closed source"). |
| Applebaum, Blackwell, Davies, Edlich, Juhász, Lackenby, Tomašev, Zheng — unknotting number ([2409.09032](https://arxiv.org/abs/2409.09032)) | **Invariants**, not the diagram (signature, determinant, writhe, Jones/Alexander features) of current diagram and of all one-crossing-change neighbours (one-step lookahead) | Crossing index to switch | IMPALA, diagrams up to ~200 crossings; braid-transformer variant only to ~60. Training: ~100k knots scrambled by random R-moves. Hard unknots = SnapPy `level` fails 25×; certified by `knot_floer_homology` genus. **Public data (CC-BY), PD codes:** `https://storage.googleapis.com/gdm-unknotting/` — `hard_unknots.csv` (1.0 GB, ~2.46M), `very_hard_unknots.csv` (385), counterexample files. |
| Davies et al., Nature 600:70 (2021), knot part | ~1.7M knots ≤16 crossings (Regina census), ~10⁶ random 80-crossing diagrams (`random_link`), 4–6-strand braid closures | — (supervised) | [google-deepmind/mathematics_conjectures](https://github.com/google-deepmind/mathematics_conjectures) `knot_theory.ipynb` |
| Kauffman–Russkikh–Taimanov ([2011.03498](https://arxiv.org/abs/2011.03498)) | Grid/rectangular diagrams as (X,O) permutation sequences, BiLSTM | — (classification of 36 classes ≤8 crossings) | [nerusskikh/deepknots](https://github.com/nerusskikh/deepknots) |
| Dranowski–Kabkov–Tubbenhauer ([2603.07955](https://arxiv.org/abs/2603.07955)) | 6-dim feature vector of a spherogram PD diagram | MultiDiscrete (mode × strength) over SnapPy macros (simplify modes, backtrack) — **not located moves** | Gymnasium + SB3 PPO on DeepMind hard unknots; >95% on very hard set. [dtubbenhauer/unknotter](https://github.com/dtubbenhauer/unknotter) |
| Dai–Hayman–Juhász–Morellato | Link diagrams (Nim core, [Spherogram-nim](https://github.com/ascchrvalstr/Spherogram-nim)) | Crossing changes, band moves, births; maskable PPO | [andras-juhasz/slice-genus-unknotting-rl](https://github.com/andras-juhasz/slice-genus-unknotting-rl) (Oct 2026); paper **(unverified)** |
| Cantarella–Schumacher–Shonkwiler, ReAPR ([2607.28772](https://arxiv.org/abs/2607.28772)) | — (not ML) | — | Simplifies all published hard unknots + ~2.6M new ones, ~60× faster than SnapPy. Strong baseline. Code link **(unverified)**. |
| Lindsay–Ruehle ([2502.12243](https://arxiv.org/abs/2502.12243)) | Several; braids best | — (supervised invariants) | |
| Putrov–Ri ([2305.05966](https://arxiv.org/abs/2305.05966)) | Plumbing graphs (GNN) | Neumann moves (RL) | Closest existing thing to RL on Kirby calculus. |
| Others | | | Lawrie–Mansi [2409.02126](https://arxiv.org/abs/2409.02126) (GNN on plumbings); Craven–Hughes–Jejjala–Kar [2112.00016](https://arxiv.org/abs/2112.00016) (slice genus); Khan–Vernitski–Lisitsa [2109.14502](https://arxiv.org/abs/2109.14502) (braid untangling, Gym); [lutscha/knot-rl](https://github.com/lutscha/knot-rl) (GNN+MCTS course project); [hughesmc/slice_genus_rl](https://github.com/hughesmc/slice_genus_rl). |

Not relevant despite the name: **KnotGym** (Chen–Artzi,
[2505.18028](https://arxiv.org/abs/2505.18028)) is a MuJoCo rope simulation with
image observations and force actions.

### Datasets

| Source | Size | Format | Python access |
|---|---|---|---|
| DeepMind `gdm-unknotting` | ~2.46M hard + 385 very hard unknots | PD-code CSV | HTTPS → `spherogram.Link(pd)` |
| Burton et al., *Hard diagrams of the unknot* ([2104.14076](https://arxiv.org/abs/2104.14076)) | ~20 named diagrams (Culprit, Goeritz, Ochiai I–IV, Haken, …) | Gauss codes in figure captions | `regina.Link.fromGauss` |
| KnotInfo / LinkInfo | 12,965 knots + 4,188 links | PD, DT, Gauss, braid | `database_knotinfo.link_list()` |
| SnapPy censuses | knots ≤14 (≤15 with `snappy_15_knots`), links ≤14; alt/nonalt knot exteriors ≤16 | DT | `spherogram.Link('K12a123')`; via manifold `.link()` **(unverified)** |
| Regina census ([data](https://regina-normal.github.io/data.html)) | 352,152,252 prime knots ≤19 crossings | bz2 CSV of `knot_sig` + DT | `regina.Link.fromKnotSig` / `fromDT` |
| Knot Atlas | Rolfsen ≤10, HT 11, Thistlethwaite links | RDF dumps; KnotTheory` (Mathematica) | spherogram names |

---

## 2. Reidemeister and Kirby moves in PD code; the action space

### Derived structure from a PD code

1. For each edge label: its two endpoints `(crossing, slot)`.
2. Orientation: under-strand goes slot 0 → slot 2; over-strand direction from
   per-component edge succession.
3. Crossing signs, writhe, linking numbers.
4. **Faces**: orbits of Mastin's successor map — follow an edge to its other
   endpoint, turn to the adjacent slot, repeat. Each face is a cyclic list of
   `(crossing, slot)` corners.

**Validity / planarity** (Mastin, [1309.3288](https://arxiv.org/abs/1309.3288),
JKTR 24(3) 2015): every label appears exactly twice (once in, once out); Thm 14
gives χ = #faces − n, so a **connected diagram is planar iff #faces = n + 2**
(in general #faces = n + 1 + #diagram components). (The "iff" is our inference
from Thm 14 / Prop. 17, not a numbered theorem in the paper.) Mastin Def. 20
defines R-moves directly on PD codes (1a, 1b, 2a, 3 — Polyak's minimal oriented
generating set) and shows PD codes mod PD-moves ≅ classical link theory.

### Every move is located at a face

| Move | Location | Legal when | # slots | Δ crossings |
|---|---|---|---|---|
| R1⁻ | monogon face (edge with both ends at the same crossing, adjacent slots) | always | ≤ n | −1 |
| R1⁺ | edge × side (L/R) × sign (±) | always | 8n | +1 |
| R2⁻ | bigon face | same strand is over at both crossings | ≤ n | −2 |
| R2⁺ | ordered pair of (edge, side) bounding a **common face** (which goes over) | edges distinct (same edge = two R1s) | O(n²) worst case | +2 |
| R3 | triangle face | one strand is over at both of its triangle crossings (triangle not alternating) | ≤ 2n | 0 |
| Crossing change (not isotopy; for unknotting number) | crossing | always | n | 0 |

**Recording a move:** a small frozen record `(kind, anchor, params)` with anchors
in current edge labels / crossing indices (or Regina-style arcs:
`(crossing, upper|lower)` = the edge leaving that strand). Renormalize labels
after each move; log trajectories as `(canonical_sig, move)` pairs.

### Measured legal-move counts

Regina `hasR*` checks; "legal / naive padded slots":

| Diagram | n | R1⁻ | R2⁻ | R3 | R1⁺ | R2⁺ |
|---|---|---|---|---|---|---|
| 3_1 | 3 | 0/3 | 0/3 | 0/6 | 24/24 | 18/120 |
| 4_1 | 4 | 0/4 | 0/4 | 0/8 | 32/32 | 28/224 |
| 8_20 | 8 | 0/8 | 0/8 | 3/16 | 64/64 | 76/960 |
| K12n123 | 12 | 0/12 | 0/12 | 0/24 | 96/96 | 134/2208 |
| 3_1 after 15 random moves | 21 | 3/21 | 6/21 | 2/42 | 168/168 | 480/6888 |

Takeaways: simplifying moves are rare (none on minimal diagrams); R2⁺
dominates; a flat padded space is >90% masked and quadratic in n.

### Action-space designs

1. **Flat padded `Discrete` + mask** (sb3-contrib `MaskablePPO`). Simplest; O(N²) for R2⁺.
2. **Factored / autoregressive**: move type → first anchor → second anchor, each masked conditionally.
3. **Enumerate-and-score (recommended)**: env returns the list of legal moves;
   a GNN embeds edges/faces/crossings and scores each candidate, or a value net
   scores the afterstate (transitions are cheap and deterministic — cf. the
   one-step lookahead in Applebaum et al.). No padding, relabelling-equivariant,
   compatible with MCTS/beam search using `sig()` as a transposition key. Cost:
   needs a custom PPO/DQN loop (CleanRL/TorchRL style), not stock SB3.

In all cases: cap `max_crossings` and mask R1⁺/R2⁺ that would exceed it.

### Kirby calculus in PD code

**State:** PD code + per-component integer framing coefficient (stored as the
true framing, i.e. relative to the Seifert framing, so R-moves never change it;
blackboard framing = self-writhe) + optional `dotted` flag (1-handles) +
crossingless components with their framings.

**Moves:**
- **Isotopy:** R1–R3 as above (framings unchanged).
- **Blow-up / blow-down** (Kirby's original form): add/remove a *split*
  ±1-framed crossingless unknot. Trivial in the data model; action = sign.
  General (non-split) blow-down = ∓1 full twist on the strands through the
  unknot's disk, fᵢ → fᵢ ∓ lk(Kᵢ, U)² — implement later as a macro.
- **Handle slide of Kᵢ over Kⱼ:** location = an edge of Kᵢ and an edge of Kⱼ on
  a common face, plus sign (handle addition/subtraction).
  1. Blackboard push-off Kⱼ′ of Kⱼ: each self-crossing of Kⱼ → 4 crossings,
     each crossing of Kⱼ with another component → 2.
  2. Add |fⱼ − writhe(Kⱼ)| full twists to the ribbon to fix the framing.
  3. Band-sum the two edges across the face (Regina `graft`).
  4. fᵢ′ = fᵢ + fⱼ ± 2·lk(Kᵢ, Kⱼ); lk(Kᵢ′, Kₖ) = lk(Kᵢ, Kₖ) ± lk(Kⱼ, Kₖ).

  Restricting bands to a single face is complete up to R-moves (any band can be
  shortened by isotopy first), so this action space is the same size as R2⁺.
  Crossing count grows fast — penalize in the reward.
- **4-manifolds:** 1/2-handle cancellation (2-handle passes once through a
  dotted circle), 2/3-handle cancellation (remove split 0-framed unknot); no
  blow-ups unless working up to #CP².
- **Rational surgery** (optional): slam-dunks, Rolfsen twists.

**Verification oracle:** SnapPy `Link(pd).exterior().dehn_fill([(f₁,1), …])`.
Framing n ↔ filling (n, 1), because SnapPea sets each longitude to the
homologically trivial one (`adjust_longitudes` in `link_complement.c`). Compare
`homology()`, volume, `is_isometric_to()` before/after every move; Regina
`isSphere()` certifies S³.

---

## 3. What the framework needs

1. **`LinkDiagram` core** (no RL code)
   - Constructors: `from_pd` (validate labels, orientation, face count),
     `from_braid`, `from_dt`, `from_gauss`, `from_knotinfo(name)`; `to_pd`,
     `to_gauss`, `copy`.
   - Cached derived data: edge endpoints, components + orientation, signs,
     faces, writhe, linking numbers, crossingless-component count.
   - Canonical key: `canonical()` / `__hash__` (Regina `sig()` or own).
   - Move API: `Move` dataclass; `legal_moves(kinds=…)`, `is_legal(m)`,
     non-mutating `apply(m) -> LinkDiagram`, `inverse(m)`.
   - Oracles/features: crossing count, `is_trivial_diagram`, certified
     `is_unknot` (knot Floer genus), cheap invariants (signature, determinant,
     Alexander).
2. **`FramedLink(LinkDiagram)`**: framings, dotted flags, Kirby moves,
   3-manifold oracle.
3. **Encoders** (separate from the diagram)
   - Graph: crossings as nodes with **4 ordered ports** (the rotation system is
     what encodes planarity — a plain crossing adjacency matrix loses it), or a
     crossing–face bipartite graph so face-located actions are natural.
   - Sequence: signed Gauss code for transformers.
   - Padded tensors for SB3-style baselines.
4. **Gymnasium env**
   - `reset(seed, options)` with a task/curriculum sampler.
   - `step(action)`, `action_masks()`, `info["legal_moves"]`, `info["sig"]`.
   - Reward: −1 per step + potential-based shaping Φ = −c·(crossings) (keeps the
     optimal policy unchanged even though solutions may need to go up) +
     terminal bonus.
   - Termination at goal; truncation at `max_steps` or `max_crossings`.
   - `render()` via `spherogram.Link.view()` or own planar drawing.
5. **Tasks**: unknotting; simplification to minimal crossing number; reach a
   target diagram (equivalence); unknotting number (crossing changes); ribbon
   (bands); Kirby: reduce to empty (S³) or reach a target.
6. **Data generators**: random braid closures, `spherogram.random_link`,
   `backtrack`; plus hard datasets (see below).
7. **Baselines / evaluation**: random agent; greedy (spherogram `level`/`global`,
   Regina `simplify`, ReAPR); exact BFS (Regina `rewrite`/`simplifyExhaustive`)
   for optimal path lengths on small diagrams. Metrics: success rate, path
   length vs optimal, max crossing excursion.
8. **Tests against Regina**: our `legal_moves` must match `hasR*`, `apply` must
   match `withR*` up to `sig()`, every R-move preserves Jones/Alexander, every
   Kirby move preserves the Dehn-filled manifold.

### Training-data finding

200 unknot diagrams (16–84 crossings, mean 46.9) generated from a 1-crossing
unknot by `backtrack(20–60 steps)`:

| Simplifier | Reaches 0 crossings |
|---|---|
| spherogram `basic` (R1/R2 removal only) | 167 / 200 |
| spherogram `level` (+ random R3) | 200 / 200 |
| Regina `simplify` | 200 / 200 |

The reverse of a backtrack path is monotone non-increasing in crossings, so
these are too easy. Train on the DeepMind hard unknots, Burton's named
diagrams, or generators that force the crossing count up along the way.

---

## 4. Issues in the current code

- `UnknotEnv._apply_action` applies R2 at crossings (i, i+1) and R3 at
  (i, i+1, i+2); these are not valid locations — R2/R3 are face-located.
- The observation (N×N crossing adjacency + N×4 node features) does not encode
  the cyclic port order / over-under per port.
- Tests import `unknot_rl.…`, but no package by that name exists (code is under
  `src/`). `is_unknot`, `draw_knot_diagram`, `graph_to_dot` don't exist;
  `visualization.py` does `import matplotlib as plt` (should be
  `matplotlib.pyplot`); `self.image_size` is undefined. (Tests not run.)
- This machine has only system Python 3.9.6, but `pyproject.toml` requires
  ≥3.10. All packages above install and run fine on 3.9.

## 5. Recommendation and open decisions

- **Recommended:** write our own PD core (face tracing + vectorized move
  enumeration — Regina's per-candidate `hasR2` is O(n²) Python calls, and the
  core is needed for Kirby moves anyway). Use Regina and SnapPy as test oracles
  and sources of invariants/certificates.
- **Open:** action-space design (flat masked vs enumerate-and-score); graph
  encoding (port graph vs crossing–face graph); whether to depend on Regina at
  runtime or only in tests; first target task (unknotting on hard unknots vs
  general simplification).

---

## Appendix: experiment snippets

Count legal moves (Regina):

```python
import itertools, regina, spherogram
L = regina.Link.fromPD(spherogram.Link('8_20').PD_code(min_strand_index=1))
arcs = [L.crossing(i).lower() for i in range(L.size())] + [L.crossing(i).upper() for i in range(L.size())]
r1_add = [(a, s, g) for a in arcs for s in (0, 1) for g in (1, -1) if L.hasR1(a, s, g)]
r2_add = [(a, s, b, t) for a, b in itertools.permutations(arcs, 2) for s in (0, 1) for t in (0, 1) if L.hasR2(a, s, b, t)]
r3 = [(c, s) for c in range(L.size()) for s in (0, 1) if L.hasR3(L.crossing(c), s)]
```

Convention check (KnotInfo vs spherogram vs Regina):

```python
import ast, regina, spherogram
from database_knotinfo import link_list
r = {k['name']: k for k in link_list()}['3_1']
pd = ast.literal_eval(r['pd_notation'])
assert sum(c.sign for c in spherogram.Link(pd).crossings) == regina.Link.fromPD(pd).writhe()
print(r['jones_polynomial'], regina.Link.fromPD(pd).jones())  # t+t^3-t^4  vs  -x^8+x^6+x^2 (x=√t)
```

Backtrack hardness:

```python
K = spherogram.Link(braid_closure=[1])   # 1-crossing unknot
K.backtrack(steps=40, prob_type_1=0.2, prob_type_2=0.5)
pd = K.PD_code(min_strand_index=1)
A = spherogram.Link(pd); A.simplify('level'); print(len(A.crossings))  # 0
```
