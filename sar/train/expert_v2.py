"""Expert iteration for env v2 (plan section 8.3, milestone 4).

Two commands:

  gen    roll out a policy on training maps and store per-decision samples:
         observations (float16), the legal mask, a policy target over the 66
         actions, and the Monte Carlo discounted team return to go.
           --policy planner  bootstrap round: target is the look-ahead planner's
                             action (one hot), executed with a small share of
                             random legal actions for state coverage
           --policy search   network-guided fleet search: target is the Gumbel
                             MuZero improved policy (`action_weights`), which is
                             much sharper than the raw visit counts
  train  fit `networks_v2.PolicyValue` on one or more gen directories:
         cross entropy to the policy target, squared error to the return.

Only rows where the drone actually decides (active, airborne, not frozen) are
stored. Steps are subsampled per episode so the data fits host memory.

Usage (from the project root):
  uv run python -m sar.train.expert_v2 gen --policy planner --episodes 512 \
      --out runs/v2/m4_data/r0
  uv run python -m sar.train.expert_v2 train --data runs/v2/m4_data/r0 \
      --out runs/v2/m4_ckpt/r0
  uv run python -m sar.train.expert_v2 gen --policy search --ckpt runs/v2/m4_ckpt/r0 \
      --episodes 128 --batch 8 --sims 32 --out runs/v2/m4_data/r1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sar.config_v2 import STAGES_V2, EnvConfigV2, TrainConfigV2
from sar.env.v2.grid_env_v2 import ACTIVE, AIRBORNE, SearchEnvV2
from sar.search import mcts, policies_v2 as P
from sar.train import networks_v2 as NN

GAMMA = 0.995
OBS_KEYS = ("local", "global_view", "scalars", "action_mask", "gain")


# ----------------------------------------------------------------- checkpoints
def save_ckpt(out, params, meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def load_ckpt(path, dtype=NN.DT):
    """(net, params, meta) from a checkpoint directory."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    net = NN.PolicyValue(n_actions=meta.get("n_actions", 66), width=meta["width"],
                         embed=meta["embed"], dtype=dtype,
                         use_gain=meta.get("use_gain", False))
    crop = meta.get("crop", 31)
    tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros((1, crop, crop, 7)),
                    jnp.zeros((1, 32, 32, 5)), jnp.zeros((1, 16)),
                    jnp.ones((1, net.n_actions), bool), jnp.ones((1, net.n_actions)))
    params = flax.serialization.from_bytes(tmpl, (path / "params.msgpack").read_bytes())
    return net, params, meta


# ----------------------------------------------------------------------- env
def make_env(grid, kinds=(0, 1, 2), n_agents=3, **kw):
    lo, hi = STAGES_V2[grid]
    cfg = EnvConfigV2(grid=grid, n_agents=n_agents, extent_min=lo, extent_max=hi,
                      train_prior_kinds=tuple(kinds), **kw).with_grid(grid)
    return SearchEnvV2(cfg)


def decides(state):
    """[N] bool: the drone's action changes anything this step."""
    return (state.status == ACTIVE) & (state.ground == AIRBORNE) & (state.freeze == 0)


# ------------------------------------------------------------------------ gen
def make_episode_fn(env, act_fn, p_random):
    """Episode rollout recording obs, target, decide flag and reward per step."""
    T, A = env.cfg.max_steps, env.n_actions

    def ep(params, k):
        k1, k2 = jax.random.split(k)
        s, _ = env.reset(k1)

        def body(c, _):
            s, done, k = c
            k, ka, kr, ku = jax.random.split(k, 4)
            obs = env.observe(s)
            gain = P.action_gain(env, s)
            a, target = act_fn(params, s, ka)
            # exploration: a few random legal actions (target unchanged)
            legal = obs["action_mask"]
            rnd = jax.random.categorical(kr, jnp.where(legal, 0.0, -1e9))
            flip = jax.random.bernoulli(ku, p_random, a.shape)
            a = jnp.where(flip, rnd, a)
            s2, _, r, d = env.step(s, a)
            rec = (obs["local"].astype(jnp.float16), obs["global"].astype(jnp.float16),
                   obs["scalars"].astype(jnp.float16), legal, gain.astype(jnp.float16),
                   target.astype(jnp.float16), decides(s) & ~done,
                   jnp.where(done, 0.0, r))
            return (s2, done | d, k), rec

        _, recs = jax.lax.scan(body, (s, jnp.array(False), k2), jnp.arange(T))
        return recs

    return ep


def returns_to_go(rew, gamma=GAMMA):
    """[B, T] rewards -> discounted return from each step on."""
    out = np.zeros_like(rew)
    acc = np.zeros(rew.shape[0], rew.dtype)
    for t in range(rew.shape[1] - 1, -1, -1):
        acc = rew[:, t] + gamma * acc
        out[:, t] = acc
    return out


def cmd_gen(args):
    env = make_env(args.grid)
    A = env.n_actions
    tcfg = TrainConfigV2(search_depth=args.search_depth, max_considered=args.considered)
    params = None
    if args.policy == "planner":
        def act_fn(params, s, k):
            a = P.lookahead(env, s, k)
            return a, jax.nn.one_hot(a, A)
    elif args.policy == "greedy":
        def act_fn(params, s, k):
            a = P.greedy(env, s, k)
            return a, jax.nn.one_hot(a, A)
    elif args.policy == "search":
        net, params, _ = load_ckpt(args.ckpt)
        run = mcts.make_fleet_search_fn(env, tcfg, net=net)

        def act_fn(params, s, k):
            res = run(params, s, k, args.sims)
            return res.action, res.weights
    else:
        raise SystemExit(f"unknown policy {args.policy}")

    ep = jax.jit(jax.vmap(make_episode_fn(env, act_fn, args.p_random), (None, 0)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total, t0, ep_returns = 0, time.time(), []
    for i in range(0, args.episodes, args.batch):
        b = min(args.batch, args.episodes - i)
        keys = jax.random.split(jax.random.PRNGKey(args.seed * 100003 + i), b)
        loc, glb, sca, msk, gan, tgt, dec, rew = jax.device_get(ep(params, keys))
        rtg = returns_to_go(np.asarray(rew, np.float32))
        ep_returns += rtg[:, 0].tolist()
        T = rew.shape[1]
        # keep a random subset of steps per episode, then only deciding rows
        steps = np.sort(rng.choice(T, min(args.keep_steps, T), replace=False))
        sel = dec[:, steps]                                   # [B,S,N]
        e_i, s_i, n_i = np.nonzero(sel)
        t_i = steps[s_i]
        shard = {
            "local": loc[e_i, t_i, n_i], "global_view": glb[e_i, t_i, n_i],
            "scalars": sca[e_i, t_i, n_i], "action_mask": msk[e_i, t_i, n_i],
            "gain": gan[e_i, t_i, n_i],
            "target": tgt[e_i, t_i, n_i], "ret": rtg[e_i, t_i].astype(np.float32),
            "episode": (i + e_i).astype(np.int32), "t": t_i.astype(np.int32),
        }
        np.savez(out / f"shard_{i:06d}.npz", **shard)
        total += len(e_i)
        print(f"[{i + b}/{args.episodes}] {len(e_i)} samples, mean return "
              f"{np.mean(rtg[:, 0]):.4f}, {time.time() - t0:.0f}s", flush=True)
    meta = {**vars(args), "samples": total, "mean_return": float(np.mean(ep_returns)),
            "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("gen done", json.dumps(meta), flush=True)


# ---------------------------------------------------------------------- train
def load_data(dirs):
    cols = {}
    for d in dirs:
        for p in sorted(Path(d).glob("shard_*.npz")):
            with np.load(p) as z:
                for k in OBS_KEYS + ("target", "ret", "episode"):
                    if k not in z:
                        continue
                    v = z[k]
                    if k == "episode":
                        v = v + hash(str(d)) % 1000 * 1_000_000
                    cols.setdefault(k, []).append(v)
    if not cols:
        raise SystemExit(f"no shards under {dirs}")
    return {k: np.concatenate(v, 0) for k, v in cols.items()}


def cmd_train(args):
    d = load_data(args.data.split(","))
    n = len(d["ret"])
    # validation split by episode, so it is not neighbouring steps of a train episode
    eps = np.unique(d["episode"])
    rng = np.random.default_rng(args.seed)
    val_eps = rng.choice(eps, max(1, int(len(eps) * args.val_frac)), replace=False)
    is_val = np.isin(d["episode"], val_eps)
    tr_idx, va_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    va_idx = va_idx[: args.val_max]
    print(f"{n} samples from {len(eps)} episodes: train {len(tr_idx)} val {len(va_idx)}",
          flush=True)

    net = NN.PolicyValue(width=args.width, embed=args.embed, use_gain=args.gain)
    if args.init:
        _, params, _ = load_ckpt(args.init)
    else:
        params = net.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 31, 31, 7)),
                          jnp.zeros((1, 32, 32, 5)), jnp.zeros((1, 16)),
                          jnp.ones((1, 66), bool), jnp.ones((1, 66)))
    sched = optax.cosine_decay_schedule(args.lr, args.iters, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(sched, weight_decay=args.wd))
    opt_state = opt.init(params)

    def batch(idx):
        tgt = d["target"][idx].astype(np.float32) * d["action_mask"][idx]
        tgt = tgt / np.maximum(tgt.sum(-1, keepdims=True), 1e-6)
        return {"local": d["local"][idx], "glob": d["global_view"][idx],
                "scalars": d["scalars"][idx], "mask": d["action_mask"][idx],
                "gain": d["gain"][idx] if args.gain else None, "target": tgt, "ret": d["ret"][idx] * NN.VALUE_SCALE}

    def loss_fn(p, b):
        logits, value = net.apply(p, b["local"], b["glob"], b["scalars"], b["mask"],
                                  b["gain"])
        lp = jax.nn.log_softmax(logits)
        pol = -jnp.sum(jnp.where(b["mask"], b["target"] * lp, 0.0), -1).mean()
        ent = -jnp.sum(jnp.where(b["target"] > 0, b["target"] * jnp.log(
            jnp.maximum(b["target"], 1e-9)), 0.0), -1).mean()
        val = ((value - b["ret"]) ** 2).mean()
        acc = (jnp.argmax(logits, -1) == jnp.argmax(b["target"], -1)).mean()
        return pol + args.value_weight * val, {"kl": pol - ent, "val": val, "acc": acc}

    @jax.jit
    def step(p, o, b):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, b)
        u, o = opt.update(g, o, p)
        return optax.apply_updates(p, u), o, aux

    eval_fn = jax.jit(loss_fn)

    def evaluate(p):
        accs = []
        for j in range(0, len(va_idx), 1024):
            _, aux = eval_fn(p, batch(va_idx[j:j + 1024]))
            accs.append({k: float(v) for k, v in aux.items()})
        return {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}

    best, best_score, hist, t0 = params, np.inf, [], time.time()
    for it in range(1, args.iters + 1):
        idx = rng.choice(tr_idx, args.batch, replace=len(tr_idx) < args.batch)
        params, opt_state, aux = step(params, opt_state, batch(idx))
        if it % args.eval_every == 0 or it == args.iters:
            va = evaluate(params)
            score = va["kl"] + args.value_weight * va["val"]
            hist.append({"iter": it, **{f"train_{k}": float(v) for k, v in aux.items()},
                         **{f"val_{k}": v for k, v in va.items()}})
            print(f"it {it} train kl {float(aux['kl']):.3f} val {float(aux['val']):.4f} "
                  f"acc {float(aux['acc']):.3f} | val kl {va['kl']:.3f} "
                  f"val {va['val']:.4f} acc {va['acc']:.3f} | {time.time() - t0:.0f}s",
                  flush=True)
            if score < best_score:
                best_score, best = score, params
    meta = {"data": args.data, "init": args.init, "samples": n, "iters": args.iters,
            "batch": args.batch, "lr": args.lr, "width": args.width, "embed": args.embed,
            "value_scale": NN.VALUE_SCALE, "value_weight": args.value_weight,
            "n_actions": 66, "crop": 31, "use_gain": args.gain, "best_val_score": best_score,
            "history": hist, "seconds": time.time() - t0}
    save_ckpt(args.out, best, meta)
    print(f"saved {args.out} best {best_score:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--policy", default="planner")
    g.add_argument("--ckpt", default=None)
    g.add_argument("--grid", type=int, default=128)
    g.add_argument("--episodes", type=int, default=256)
    g.add_argument("--batch", type=int, default=32)
    g.add_argument("--keep-steps", type=int, default=64)
    g.add_argument("--sims", type=int, default=32)
    g.add_argument("--search-depth", type=int, default=8)
    g.add_argument("--considered", type=int, default=16)
    g.add_argument("--p-random", type=float, default=0.1)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="comma separated gen directories")
    t.add_argument("--out", required=True)
    t.add_argument("--init", default=None)
    t.add_argument("--iters", type=int, default=4000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=1e-4)
    t.add_argument("--width", type=int, default=64)
    t.add_argument("--embed", type=int, default=256)
    t.add_argument("--gain", action="store_true", help="action_gain network input")
    t.add_argument("--value-weight", type=float, default=1.0)
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--val-max", type=int, default=8192)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"gen": cmd_gen, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
