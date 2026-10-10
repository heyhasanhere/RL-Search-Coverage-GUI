"""Policy and value network for env v2 (plan section 8.3 step 2).

Same encoder idea as the version 1 network (`sar/train/networks.py`): a strided
conv stack over the local crop, a conv over the global view, and a dense pass
over the scalars. Two heads come off the shared trunk:

  - policy logits over the 66 actions, masked to the legal ones,
  - a scalar value, which is the discounted return the search should expect.

The inputs are exactly what `sar.search.teacher` writes to
`runs/v2/teacher/*/shard_*.npz`, so cloning needs no glue: local is
[N, crop, crop, 7], global_view is [N, 32, 32, 5] and scalars is [N, 16].

`network_prior` returns the (logits, value) pair that `mcts` needs, so the same
function serves the cloning loss, the network-only policy and the tree.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

DT = jnp.bfloat16
NEG = -1e9
VALUE_SCALE = 10.0    # returns live in 0..0.6; the value head predicts 10x that


class Trunk(nn.Module):
    """Shared encoder. Returns a 256-d feature vector per agent."""

    width: int = 64
    embed: int = 256
    dtype: object = DT

    @nn.compact
    def __call__(self, local, glob, scalars):
        DT = self.dtype
        x = local.astype(DT)
        x = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), dtype=DT)(x))
        x = nn.relu(nn.Conv(self.width, (3, 3), strides=(2, 2), dtype=DT)(x))
        x = nn.relu(nn.Conv(self.width, (3, 3), strides=(2, 2), dtype=DT)(x))
        x = x.reshape(x.shape[:-3] + (-1,))

        y = glob.astype(DT)
        y = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), dtype=DT)(y))
        y = nn.relu(nn.Conv(self.width, (3, 3), strides=(2, 2), dtype=DT)(y))
        y = y.reshape(y.shape[:-3] + (-1,))

        s = nn.relu(nn.Dense(64, dtype=DT)(scalars.astype(DT)))
        h = jnp.concatenate([x, y, s], -1)
        return nn.relu(nn.Dense(self.embed, dtype=DT)(h))


class PolicyValue(nn.Module):
    """Policy logits and a value estimate for one drone's observation."""

    n_actions: int = 66
    width: int = 64
    embed: int = 256
    dtype: object = DT    # bfloat16 on GPU; use float32 for CPU inference
    use_gain: bool = False  # take `policies_v2.action_gain` [N, 66] as an input

    @nn.compact
    def __call__(self, local, glob, scalars, action_mask=None, gain=None):
        h = self.trunk(local, glob, scalars)
        if self.use_gain:
            g = gain.astype(jnp.float32)
            h = jnp.concatenate([h, nn.relu(nn.Dense(64, dtype=self.dtype)(g))], -1)
        h = nn.relu(nn.Dense(self.embed, dtype=self.dtype)(h))
        logits = nn.Dense(self.n_actions,
                          kernel_init=nn.initializers.orthogonal(0.01))(h.astype(jnp.float32))
        if self.use_gain:
            # residual greedy prior: beta * log(gain), beta learned, starts at 2
            beta = self.param("beta", lambda k: jnp.asarray(2.0))
            logits = logits + beta * jnp.log(g + 1e-3)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(h.astype(jnp.float32))[..., 0]
        if action_mask is not None:
            logits = jnp.where(action_mask, logits, NEG)
        return logits, value

    def trunk(self, local, glob, scalars):
        return Trunk(self.width, self.embed, self.dtype)(local, glob, scalars)


def apply_obs(net, params, obs):
    """(logits, value) from an observation dict (`env.observe` keys, plus
    `gain` from `policies_v2.action_gain` when the net uses it)."""
    glob = obs["global"] if "global" in obs else obs["global_view"]
    gain = obs.get("gain") if net.use_gain else None
    return net.apply(params, obs["local"], glob, obs["scalars"], obs["action_mask"], gain)


def network_prior(params, net, local, glob, scalars, action_mask):
    """(logits [N, A], value [N]) for `mctx`, from network observations.

    `mctx` wants rank-2 prior logits and a rank-1 value, both per tree row, so
    the leading agent axis is already in the right place.
    """
    logits, value = net.apply(params, local, glob, scalars, action_mask)
    return logits, value


def policy_targets(visits, action_mask=None):
    """Visit counts to a target distribution over the legal actions.

    Only about 13 of the 66 actions are legal in a typical state, and the search
    spreads its simulations across those, so the target is far from uniform over
    all 66. Rows where every visit is zero (an empty or fully illegal search)
    fall back to uniform over the legal set, which keeps the cross entropy
    finite and contributes no gradient to the illegal actions.
    """
    v = visits.astype(jnp.float32)
    total = v.sum(-1, keepdims=True)
    p = v / jnp.maximum(total, 1e-6)
    if action_mask is None:
        n_legal = jnp.full_like(total, float(v.shape[-1]))
    else:
        m = action_mask.astype(jnp.float32)
        n_legal = jnp.maximum(m.sum(-1, keepdims=True), 1.0)
        p = p * m
    uniform = 1.0 / n_legal
    return jnp.where(total > 0, p, uniform)