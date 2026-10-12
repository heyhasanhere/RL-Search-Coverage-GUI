"""Tier 0 evaluation in the JAX grid env (plan section 10.1).

Every method runs on the same seeds. Map sets: held-out prior type (blobs)
and training-style priors. Fleet: recall (forced leaves and joins) or no recall.
Metrics: discounted return, fraction found by the end, decisions to 50% and 90%.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from sar.baselines.policies import POLICIES
from sar.config import EnvConfig
from sar.env import maps
from sar.env.grid_env import SearchEnv
from sar.train.networks import Actor
from sar.train.train import STAGES

GAMMA = 0.995


def make_eval_env(grid, ring, kinds, recall, real_paths=(), t_turn=EnvConfig.t_turn):
    lo, hi = STAGES[grid]
    kw = dict(p_leave=1.0, p_join=0.5) if recall else dict(p_leave=0.0, p_join=0.0)
    cfg = EnvConfig(grid=grid, ring=ring, extent_min=lo, extent_max=hi,
                    train_prior_kinds=kinds, t_turn=t_turn, **kw)
    bank = maps.load_real_bank(real_paths, grid, seed=1) if maps.REAL_KIND in kinds else None
    return SearchEnv(cfg, real_bank=bank)


def rl_policy(actor, params):
    def act(env, state, key):
        obs = env.observe(state)
        lg = actor.apply(params, obs["local"], obs["global"], obs["scalars"], obs["action_mask"])
        return jnp.argmax(lg, -1)
    return act


def run_episodes(env, policy, keys):
    T = env.cfg.max_steps

    def ep(k):
        k1, k2 = jax.random.split(k)
        s, _ = env.reset(k1)

        def body(c, t):
            s, done, k = c
            k, ka = jax.random.split(k)
            a = policy(env, s, ka)
            s2, _, r, d = env.step(s, a)
            r = jnp.where(done, 0.0, r)
            return (s2, done | d, k), r

        _, rs = jax.lax.scan(body, (s, jnp.array(False), k2), jnp.arange(T))
        return rs

    rs = jax.jit(jax.vmap(ep))(keys)                      # [B,T]
    cum = jnp.cumsum(rs, -1)
    disc = (rs * GAMMA ** jnp.arange(rs.shape[1])).sum(-1)
    def t_to(x):
        hit = cum >= x
        return jnp.where(hit.any(-1), jnp.argmax(hit, -1), rs.shape[1]).astype(jnp.float32)
    return {"disc_return": disc, "found": cum[:, -1], "t50": t_to(0.5), "t90": t_to(0.9)}


def summarize(m):
    out = {}
    for k, v in m.items():
        v = np.asarray(v)
        out[k] = float(v.mean())
        out[k + "_se"] = float(v.std() / np.sqrt(len(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=96)
    ap.add_argument("--ring", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--actor", action="append", default=[], help="name=path/to/actor.msgpack")
    ap.add_argument("--methods", default="lawnmower,greedy,planner")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-turn", type=float, default=EnvConfig.t_turn)
    ap.add_argument("--mapsets", default="heldout,train_style", help="heldout, train_style, real_heldout")
    ap.add_argument("--real-heldout", default="data/regions/bank_heldout.npz")
    args = ap.parse_args()

    methods = {m: POLICIES[m] for m in args.methods.split(",") if m}
    for spec in args.actor:
        name, path = spec.split("=", 1)
        env0 = make_eval_env(args.grid, args.ring, (0,), False)
        actor = Actor(env0.n_actions)
        _, obs = env0.reset(jax.random.PRNGKey(0))
        tmpl = actor.init(jax.random.PRNGKey(0), obs["local"], obs["global"], obs["scalars"], obs["action_mask"])
        params = serialization.from_bytes(tmpl, Path(path).read_bytes())
        methods[name] = rl_policy(actor, params)

    results = []
    sets = {"heldout": (3,), "train_style": (0, 1, 2), "real_heldout": (maps.REAL_KIND,)}
    for mapset in args.mapsets.split(","):
        kinds = sets[mapset]
        for recall in (False, True):
            env = make_eval_env(args.grid, args.ring, kinds, recall, (args.real_heldout,), args.t_turn)
            keys = jax.random.split(jax.random.PRNGKey(12345), args.episodes)
            for name, pol in methods.items():
                parts = [run_episodes(env, pol, keys[i:i + args.batch])
                         for i in range(0, args.episodes, args.batch)]
                m = {k: jnp.concatenate([p[k] for p in parts]) for k in parts[0]}
                row = {"method": name, "mapset": mapset, "recall": recall, "grid": args.grid, **summarize(m)}
                results.append(row)
                print(f"{mapset:11s} recall={int(recall)} {name:12s} disc {row['disc_return']:.4f}±{row['disc_return_se']:.4f} "
                      f"found {row['found']:.3f} t50 {row['t50']:.1f} t90 {row['t90']:.1f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
