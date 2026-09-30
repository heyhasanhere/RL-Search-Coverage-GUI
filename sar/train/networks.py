"""Actor and centralized critic (plan section 5.7). Convs run in bfloat16."""

import flax.linen as nn
import jax.numpy as jnp

DT = jnp.bfloat16


class Encoder(nn.Module):
    use_local: bool = True

    @nn.compact
    def __call__(self, local, glob, scalars):
        feats = []
        if self.use_local:
            feats.append(self._local(local))
        y = glob.astype(DT)
        y = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), dtype=DT)(y))
        y = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), dtype=DT)(y))
        feats.append(y.reshape(y.shape[:-3] + (-1,)))
        feats.append(nn.relu(nn.Dense(64, dtype=DT)(scalars.astype(DT))))
        return nn.relu(nn.Dense(256, dtype=DT)(jnp.concatenate(feats, -1)))

    @nn.nowrap
    def _local(self, local):
        x = local.astype(DT)
        x = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), dtype=DT)(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), dtype=DT)(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), dtype=DT)(x))
        return x.reshape(x.shape[:-3] + (-1,))


class Actor(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, local, glob, scalars, action_mask):
        h = Encoder()(local, glob, scalars)
        h = nn.relu(nn.Dense(256, dtype=DT)(h))
        logits = nn.Dense(self.n_actions, kernel_init=nn.initializers.orthogonal(0.01))(h.astype(jnp.float32))
        return jnp.where(action_mask, logits, -1e9)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, local, glob, scalars, world):
        h = Encoder(use_local=False)(local, glob, scalars)
        w = world.astype(DT)
        w = nn.relu(nn.Conv(32, (3, 3), strides=(2, 2), dtype=DT)(w))
        w = nn.relu(nn.Conv(64, (3, 3), strides=(2, 2), dtype=DT)(w))
        w = w.reshape(w.shape[:-3] + (-1,))
        w = nn.relu(nn.Dense(256, dtype=DT)(w))
        h = nn.relu(nn.Dense(256, dtype=DT)(jnp.concatenate([h, w], -1)))
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(h.astype(jnp.float32))[..., 0]
