"""Policy-and-value networks over the candidate actions of :class:`~knotml.envs.CandidateActions`.

A policy maps a batch of observations to logits over the ``max_actions``
candidate slots (padding masked out) and a state value.  :class:`Policy`
supplies sampling and evaluation on top of :meth:`Policy.forward`; register new
architectures with :func:`register_policy`.

``gnn`` (:class:`GraphPolicy`): message passing on the diagram.

* **Nodes.** One per port (a crossing's 4 slots) and one per component slot.
* **Port messages.** Each port hears:
  - the port at the other end of its edge;
  - the next and previous ports around its crossing;
  - the opposite port;
  - the mean of the ports on the same face;
  - its component.
* **Component messages.** Each component hears:
  - the mean of its ports;
  - the other components, weighted by linking number;
  - a global summary.
* **Invariance.** The network uses no crossing or face index as a feature, so
  it does not depend on how crossings are numbered.
* **Policy head.** A candidate's score is an MLP of its kind, the embeddings
  of the anchors it reads (port or component), its parameter and the global
  summary.
* **Factorisation** (``factorization``).
  - ``flat`` (the default): one softmax of the scores over all candidates.
    A kind's share starts out proportional to its number of moves.
  - ``kind``: a kind head (an MLP of the kind's embedding and the global
    summary) gives P(kind) over the kinds present, and the scores are
    normalised within each kind, giving P(move | kind).  See
    :class:`KindFirst`.
* **Value head.** An MLP of mean- and max-pooled port and component embeddings.

All tensors are padded to the env's capacity, so a batch runs as dense gathers
and ``index_add_`` scatters: O(batch × (ports + candidates) × hidden) per
layer, and it runs on CUDA, MPS or CPU without extra dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import torch
from torch import nn
from torch.distributions import Categorical

from .envs import EnvSpec

__all__ = ["Policy", "GraphPolicy", "Heads", "KindFirst", "POLICIES", "register_policy", "make_policy"]

POLICIES: dict[str, type] = {}


def register_policy(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        POLICIES[name] = cls
        return cls

    return deco


def make_policy(name: str, spec: EnvSpec, params: dict) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; registered: {', '.join(POLICIES)}")
    cls = POLICIES[name]
    try:
        hp = cls.Hyperparameters(**params)
    except TypeError as err:
        raise ValueError(f"policy {name!r}: {err}") from None
    return cls(spec, hp)


class Heads(NamedTuple):
    """What a policy's ``forward`` returns."""

    logits: torch.Tensor  # (B, A) a score per candidate slot
    mask: torch.Tensor  # (B, A) bool: which slots hold a legal move
    value: torch.Tensor  # (B,)
    kind_logits: Optional[torch.Tensor] = None  # (B, K): set for a kind-first policy


class KindFirst(Categorical):
    """A distribution over the candidate slots of the form P(kind) · P(move | kind).

    * ``scores`` are normalised within each kind, giving log P(move | kind).
    * ``kind_logits`` are normalised over the kinds that have a legal move,
      giving log P(kind).
    * As a ``Categorical`` over slots, sampling from it is the same as drawing a
      kind, then a move of that kind.  ``log_prob`` and ``probs`` are exact, so
      PPO's ratios and KL need nothing special.

    :meth:`entropy` is the kind-balanced entropy that PPO's bonus uses:

        H(kind) + Σ_k P(k) · H(move | k) / log n_k   (a kind with a single move counts 1)

    It is largest for uniform kinds with uniform moves inside each, whatever
    the kinds' sizes.  The joint entropy would be largest for uniform moves,
    which pushes probability back to the kinds with many moves.
    """

    def __init__(self, scores: torch.Tensor, mask: torch.Tensor, kinds: torch.Tensor,
                 kind_logits: torch.Tensor) -> None:
        B, K = kind_logits.shape
        neg = torch.finfo(scores.dtype).min
        bins = torch.where(mask, kinds, K)  # padding goes to a spare bin K
        ones = mask.to(scores.dtype)
        counts = scores.new_zeros(B, K + 1).scatter_add_(1, bins, ones)[:, :K]
        present = counts > 0
        masked = scores.masked_fill(~mask, neg)
        peak = scores.new_full((B, K + 1), neg).scatter_reduce(1, bins, masked, "amax")
        total = scores.new_zeros(B, K + 1).scatter_add_(1, bins, (masked - peak.gather(1, bins)).exp() * ones)
        lse = peak + total.clamp_min(torch.finfo(scores.dtype).tiny).log()
        within = (masked - lse.gather(1, bins)).masked_fill(~mask, 0.0)  # log P(move | kind)
        kind_logp = kind_logits.masked_fill(~present, neg).log_softmax(-1).masked_fill(~present, 0.0)
        joint = (within + torch.cat([kind_logp, kind_logp.new_zeros(B, 1)], 1).gather(1, bins)).masked_fill(~mask, neg)
        super().__init__(logits=joint)
        self._kind_logp, self._present, self._counts = kind_logp, present, counts
        self._within, self._bins, self._mask = within, bins, mask

    @property
    def kind_probs(self) -> torch.Tensor:
        """P(kind), shape (B, K); 0 for kinds without a legal move."""
        return self._kind_logp.exp() * self._present

    def entropy(self) -> torch.Tensor:
        pk = self.kind_probs
        h_kind = -(pk * self._kind_logp).sum(-1)
        p_within = self._within.exp() * self._mask
        h = self._within.new_zeros(self._within.shape[0], pk.shape[1] + 1)
        h_within = h.scatter_add_(1, self._bins, -p_within * self._within)[:, :-1]
        n = self._counts
        normalised = torch.where(n > 1, h_within / n.clamp(min=2).log(), torch.ones_like(h_within))
        return h_kind + (pk * normalised).sum(-1)


class Policy(nn.Module):
    """Base class: subclasses implement ``forward(obs) -> Heads``.  Masked slots never get probability;
    with ``kind_logits`` the distribution is :class:`KindFirst`, otherwise one softmax over the slots."""

    @dataclass
    class Hyperparameters:
        pass

    def distribution(self, obs: dict) -> tuple[Categorical, torch.Tensor]:
        heads = self(obs)
        mask = heads.mask.clone()
        empty = ~mask.any(-1)  # no legal move: fall back to slot 0, which the env treats as a no-op
        mask[empty, 0] = True
        scores = heads.logits.float()
        if heads.kind_logits is None:
            return Categorical(logits=scores.masked_fill(~mask, torch.finfo(torch.float32).min)), heads.value
        kinds = obs["candidates"][..., 0].long().clamp(min=0)
        return KindFirst(scores, mask, kinds, heads.kind_logits.float()), heads.value

    def act(self, obs: dict, deterministic: bool = False):
        """``(action, log-prob, entropy, value)`` for each observation in the batch.  ``deterministic``
        takes the single most probable move (for kind-first policies too, not the best kind's best move)."""
        dist, value = self.distribution(obs)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: dict, action: torch.Tensor):
        """``(log-prob, entropy, value)`` of given actions (for the policy update)."""
        dist, value = self.distribution(obs)
        return dist.log_prob(action), dist.entropy(), value

    def value(self, obs: dict) -> torch.Tensor:
        return self(obs).value


def _mlp(sizes, out_scale: float = 1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        linear = nn.Linear(a, b)
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, gain=out_scale if last else math.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _gather(h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``h[b, index[b, i]]`` for ``h`` of shape (B, N, H); index -1 gives zeros."""
    valid = (index >= 0).unsqueeze(-1)
    idx = index.clamp(min=0).unsqueeze(-1).expand(*index.shape, h.shape[-1])
    return torch.gather(h, 1, idx) * valid


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Mean of ``src[b, i]`` over ``i`` with ``index[b, i] == j``, shape (B, size, H); index -1 is dropped."""
    B, _, H = src.shape
    flat = torch.where(index >= 0, index + size * torch.arange(B, device=index.device).unsqueeze(1), B * size)
    total = src.new_zeros(B * size + 1, H).index_add_(0, flat.reshape(-1), src.reshape(-1, H))
    count = src.new_zeros(B * size + 1).index_add_(0, flat.reshape(-1), src.new_ones(flat.numel()))
    return (total[:-1] / count[:-1].clamp(min=1).unsqueeze(-1)).view(B, size, H)


def _signed_log(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


class _Layer(nn.Module):
    def __init__(self, H: int) -> None:
        super().__init__()
        self.port = _mlp([7 * H, 2 * H, H])
        self.comp = _mlp([5 * H, 2 * H, H])
        self.link = nn.Linear(2 * H, H)
        self.norm_port, self.norm_comp = nn.LayerNorm(H), nn.LayerNorm(H)

    def forward(self, hp, hc, g, graph):
        P, C = hp.shape[1], hc.shape[1]
        face = _gather(_scatter_mean(hp, graph["face"], P), graph["face"])
        messages = [hp, _gather(hp, graph["nbr"]), hp[:, graph["next"]], hp[:, graph["prev"]], hp[:, graph["opp"]],
                    face, _gather(hc, graph["port_comp"])]
        hp = self.norm_port(hp + self.port(torch.cat(messages, -1))) * graph["port_valid"]
        ports = _scatter_mean(hp, graph["port_comp"], C)
        linked = self.link(torch.cat([graph["link_signed"] @ hc, graph["link_any"] @ hc], -1))
        update = self.comp(torch.cat([hc, ports, linked, g.unsqueeze(1).expand(-1, C, -1), hc * linked], -1))
        hc = self.norm_comp(hc + update) * graph["comp_valid"]
        return hp, hc


@register_policy("gnn")
class GraphPolicy(Policy):
    """Message passing on ports and components (see the module docstring)."""

    @dataclass
    class Hyperparameters:
        hidden: int = 128
        layers: int = 4
        framing_clip: int = 8  # framings beyond ±clip share an embedding (their size still enters a scalar feature)
        factorization: str = "flat"  # flat: one softmax over moves; kind: P(kind) · P(move | kind)

    def __init__(self, spec: EnvSpec, hp: Hyperparameters) -> None:
        super().__init__()
        if hp.factorization not in ("flat", "kind"):
            raise ValueError(f"policy factorization must be 'flat' or 'kind', not {hp.factorization!r}")
        self.spec, self.hp = spec, hp
        H, P = hp.hidden, spec.n_ports
        self.slot = nn.Embedding(4, H)  # position in the crossing: under-in, over, under-out, over
        self.sign = nn.Embedding(3, H)  # crossing sign -1 / +1 (0 = padding)
        self.ctype = nn.Embedding(3, H)  # absent, 2-handle / framed, dotted
        self.framing = nn.Embedding(2 * hp.framing_clip + 1, H)
        self.comp_in = _mlp([H + 3, H, H])
        self.layers = nn.ModuleList(_Layer(H) for _ in range(hp.layers))
        self.glob = nn.Linear(2 * H, H)
        self.kind = nn.Embedding(spec.n_kinds, H)
        self.param = nn.Embedding(3, H)
        self.actor = _mlp([5 * H, 2 * H, H, 1], out_scale=0.01)
        if hp.factorization == "kind":  # only then, so flat checkpoints load unchanged
            self.kind_actor = _mlp([2 * H, H, 1], out_scale=0.01)
        self.critic = _mlp([4 * H, 2 * H, H, 1], out_scale=1.0)
        ports = torch.arange(P)
        base = 4 * (ports // 4)
        self.register_buffer("next_port", base + (ports + 1) % 4, persistent=False)
        self.register_buffer("prev_port", base + (ports + 3) % 4, persistent=False)
        self.register_buffer("opp_port", ports ^ 2, persistent=False)
        self.register_buffer("strand_of_port", 2 * (ports // 4) + (ports & 1), persistent=False)
        self.register_buffer("slot_of_port", ports % 4, persistent=False)
        self.register_buffer("kind_fields", torch.tensor(spec.kind_fields, dtype=torch.float32), persistent=False)

    def _graph(self, obs: dict) -> dict:
        P, C = self.spec.n_ports, self.spec.n_components
        n = obs["n_crossings"].long().view(-1, 1)
        k = obs["n_components"].long().view(-1, 1)
        device, dtype = n.device, self.kind.weight.dtype
        port_valid = (torch.arange(P, device=device).unsqueeze(0) < 4 * n)
        strand = obs["strand_component"].long()
        port_comp = torch.gather(strand, 1, self.strand_of_port.unsqueeze(0).expand(strand.shape[0], -1))
        port_comp = torch.where(port_valid, port_comp, -1)
        link = obs["linking"].to(dtype)
        off = 1.0 - torch.eye(C, device=device, dtype=dtype)
        return {
            "nbr": torch.where(port_valid, obs["nbr"].long(), -1),
            "face": torch.where(port_valid, obs["side_face"].long(), -1),
            "port_comp": port_comp,
            "next": self.next_port, "prev": self.prev_port, "opp": self.opp_port,
            "port_valid": port_valid.unsqueeze(-1).to(dtype),
            "comp_valid": (torch.arange(C, device=device).unsqueeze(0) < k).unsqueeze(-1).to(dtype),
            "link_signed": _signed_log(link) * off,
            "link_any": (link != 0).to(dtype) * off,
        }

    def forward(self, obs: dict):
        hp_ = self.hp
        graph = self._graph(obs)
        B, P = obs["nbr"].shape
        # components: type, framing, crossingless
        dtype = self.kind.weight.dtype
        f = obs["framing"].to(dtype)
        clip = hp_.framing_clip
        dotted = (obs["component_type"] == 2).to(dtype)
        comp_feats = torch.stack([_signed_log(f), obs["crossingless"].to(dtype), dotted], -1)
        hc = self.ctype(obs["component_type"].long()) + self.framing(f.long().clamp(-clip, clip) + clip)
        hc = self.comp_in(torch.cat([hc, comp_feats], -1)) * graph["comp_valid"]
        # ports: slot in the crossing and the crossing's sign
        sign = torch.gather(obs["sign"].long(), 1, (torch.arange(P, device=hc.device) // 4).unsqueeze(0).expand(B, -1))
        hp = (self.slot(self.slot_of_port).unsqueeze(0) + self.sign(sign + 1)) * graph["port_valid"]
        g = self._global(hp, hc, graph)
        for layer in self.layers:
            hp, hc = layer(hp, hc, g, graph)
            g = self._global(hp, hc, graph)
        kind_logits = None
        if self.hp.factorization == "kind":
            kinds = self.kind.weight.unsqueeze(0).expand(B, -1, -1)
            kind_logits = self.kind_actor(torch.cat([kinds, g.unsqueeze(1).expand_as(kinds)], -1)).squeeze(-1)
        return Heads(self._logits(obs, hp, hc, g), obs["action_mask"].bool(), self._value(hp, hc, graph),
                     kind_logits)

    def _pool(self, h, valid):
        mean = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        peak = h.masked_fill(valid == 0, -1e4).amax(1) * (valid.sum(1) > 0)
        return mean, peak

    def _global(self, hp, hc, graph):
        ports, comps = self._pool(hp, graph["port_valid"])[0], self._pool(hc, graph["comp_valid"])[0]
        return self.glob(torch.cat([ports, comps], -1))

    def _value(self, hp, hc, graph):
        return self.critic(torch.cat([*self._pool(hp, graph["port_valid"]), *self._pool(hc, graph["comp_valid"])],
                                     -1)).squeeze(-1)

    def _logits(self, obs, hp, hc, g):
        P = self.spec.n_ports
        cand = obs["candidates"].long()  # (B, A, 4): kind, a, b, param; -1 rows are padding
        kind = cand[..., 0].clamp(min=0)
        use = self.kind_fields[kind]  # (B, A, 3)

        def anchor(x):  # a port (< P) or a component slot (>= P)
            port = _gather(hp, torch.where((x >= 0) & (x < P), x, -1))
            comp = _gather(hc, torch.where(x >= P, x - P, -1))
            return port + comp

        a, b = anchor(cand[..., 1]), anchor(cand[..., 2])
        param = self.param(cand[..., 3].clamp(0, 2))
        features = [self.kind(kind), a * use[..., :1], b * use[..., 1:2], param * use[..., 2:],
                    g.unsqueeze(1).expand(-1, cand.shape[1], -1)]
        return self.actor(torch.cat(features, -1)).squeeze(-1)
