"""Behavior cloning from the look-ahead planner, then MAPPO (plan section 5.7).

Several seeds train in one process with jax.vmap (PureJaxRL style).
The map-size curriculum runs one jit per grid size; network shapes do not change.
Run one process per GPU with CUDA_VISIBLE_DEVICES for the fastest wall clock.
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from tqdm import tqdm

from sar.baselines.policies import lookahead
from sar.config import EnvConfig, TrainConfig
from sar.env import maps
from sar.env.grid_env import SearchEnv
from sar.train.networks import Actor, Critic

# grid side -> polygon extent range (cells)
STAGES = {32: (16, 30), 64: (30, 60), 96: (50, 94)}
OBS_LOWP = ("local", "global", "world")


def make_env(grid: int, ring: int = 4, real_paths=(), **kw) -> SearchEnv:
    lo, hi = STAGES[grid]
    bank = maps.load_real_bank(real_paths, grid) if real_paths else None
    return SearchEnv(EnvConfig(grid=grid, ring=ring, extent_min=lo, extent_max=hi, **kw), real_bank=bank)


def tmap(f, *t):
    return jax.tree_util.tree_map(f, *t)


def where_tree(c, a, b):
    """Select per env (leading axis) between two pytrees."""
    return tmap(lambda x, y: jnp.where(c.reshape(c.shape + (1,) * (x.ndim - 1)), x, y), a, b)


class Trainer:
    def __init__(self, env: SearchEnv, tc: TrainConfig):
        self.env, self.tc = env, tc
        self.actor = Actor(env.n_actions)
        self.critic = Critic()
        self.atx = optax.chain(optax.clip_by_global_norm(tc.max_grad_norm), optax.adam(tc.lr, eps=1e-5))
        self.ctx = optax.chain(optax.clip_by_global_norm(tc.max_grad_norm), optax.adam(tc.lr, eps=1e-5))
        self.vreset = jax.vmap(partial(env.reset))
        self.vstep = jax.vmap(env.step)
        self.vteach = jax.vmap(partial(lookahead, env))

    # ------------------------------------------------------------------ init
    def init(self, key):
        env, tc = self.env, self.tc
        k1, k2, k3, k4 = jax.random.split(key, 4)
        _, obs = env.reset(k1)
        ap = self.actor.init(k2, obs["local"], obs["global"], obs["scalars"], obs["action_mask"])
        cp = self.critic.init(k3, obs["local"], obs["global"], obs["scalars"],
                              jnp.broadcast_to(obs["world"], (env.n,) + obs["world"].shape))
        return self.fresh_runner(k4, ap, cp)

    def fresh_runner(self, key, ap, cp):
        k1, k2 = jax.random.split(key)
        es, obs = self.vreset(jax.random.split(k1, self.tc.n_envs))
        E = self.tc.n_envs
        return {
            "ap": ap, "cp": cp, "bc": ap,
            "aopt": self.atx.init(ap), "copt": self.ctx.init(cp),
            "es": es, "obs": obs,
            "ep_ret": jnp.zeros(E), "ep_disc": jnp.zeros(E), "ep_t": jnp.zeros(E, jnp.int32),
            "rng": k2,
        }

    # ------------------------------------------------------------- forward
    def logits(self, ap, obs):
        return self.actor.apply(ap, obs["local"], obs["global"], obs["scalars"], obs["action_mask"])

    def value(self, cp, obs):
        n = self.env.n
        world = jnp.broadcast_to(obs["world"][..., None, :, :, :],
                                 obs["world"].shape[:-3] + (n,) + obs["world"].shape[-3:])
        return self.critic.apply(cp, obs["local"], obs["global"], obs["scalars"], world)

    # ------------------------------------------------------------- rollout
    def rollout(self, rs, beta, teacher: bool):
        """Collect rollout_len steps. If teacher, label with the planner and
        execute the student with prob beta per env (DAgger)."""
        tc, E = self.tc, self.tc.n_envs
        rng, k_pool, k_mix = jax.random.split(rs["rng"], 3)
        pool_es, pool_obs = self.vreset(jax.random.split(k_pool, E))
        use_student = jax.random.bernoulli(k_mix, beta, (E,))

        def step(c, _):
            es, obs, ep_ret, ep_disc, ep_t, rng = c
            rng, ka, kt = jax.random.split(rng, 3)
            lg = self.logits(rs["ap"], obs)
            v = self.value(rs["cp"], obs)
            a = jax.random.categorical(ka, lg)
            if teacher:
                ta = self.vteach(es, jax.random.split(kt, E))
                a = jnp.where(use_student[:, None], a, ta)
            else:
                ta = a
            logp = jnp.take_along_axis(jax.nn.log_softmax(lg), a[..., None], -1)[..., 0]
            es2, obs2, r, d = self.vstep(es, a)
            ep_ret = ep_ret + r
            ep_disc = ep_disc + r * tc.gamma ** ep_t
            fin = (d, ep_ret, ep_disc)
            es2 = where_tree(d, pool_es, es2)
            obs2 = where_tree(d, pool_obs, obs2)
            ep_ret = jnp.where(d, 0.0, ep_ret)
            ep_disc = jnp.where(d, 0.0, ep_disc)
            ep_t = jnp.where(d, 0, ep_t + 1)
            o = {k: (x.astype(jnp.bfloat16) if k in OBS_LOWP else x) for k, x in obs.items()}
            tr = {"obs": o, "a": a, "ta": ta, "logp": logp, "v": v, "r": r, "d": d}
            return (es2, obs2, ep_ret, ep_disc, ep_t, rng), (tr, fin)

        c0 = (rs["es"], rs["obs"], rs["ep_ret"], rs["ep_disc"], rs["ep_t"], rng)
        (es, obs, ep_ret, ep_disc, ep_t, rng), (traj, fin) = jax.lax.scan(step, c0, None, tc.rollout_len)
        rs = {**rs, "es": es, "obs": obs, "ep_ret": ep_ret, "ep_disc": ep_disc, "ep_t": ep_t, "rng": rng}
        done, ret, disc = fin
        n_fin = jnp.maximum(done.sum(), 1)
        stats = {"ep_return": (ret * done).sum() / n_fin, "ep_disc": (disc * done).sum() / n_fin,
                 "episodes": done.sum()}
        return rs, traj, stats

    def gae(self, traj, last_v):
        tc = self.tc

        def f(c, x):
            g, nv = c
            r, d, v = x
            nd = 1.0 - d[:, None].astype(jnp.float32)
            delta = r[:, None] + tc.gamma * nv * nd - v
            g = delta + tc.gamma * tc.gae_lambda * nd * g
            return (g, v), g

        _, adv = jax.lax.scan(f, (jnp.zeros_like(last_v), last_v), (traj["r"], traj["d"], traj["v"]), reverse=True)
        return adv, adv + traj["v"]

    # -------------------------------------------------------------- update
    def _minibatches(self, rng, batch):
        tc = self.tc
        n = tc.rollout_len * tc.n_envs
        flat = tmap(lambda x: x.reshape((n,) + x.shape[2:]), batch)
        perm = jax.random.permutation(rng, n)
        return tmap(lambda x: x[perm].reshape((tc.ppo_minibatches, -1) + x.shape[1:]), flat)

    def update(self, rs, traj, it, mode: str):
        tc = self.tc
        last_v = self.value(rs["cp"], rs["obs"])
        adv, ret = self.gae(traj, last_v)
        batch = {**traj, "adv": adv, "ret": ret}
        kl_w = tc.kl_coef * jnp.maximum(0.0, 1.0 - it / tc.kl_fade)
        actor_on = it >= tc.critic_warmup

        def loss_fn(params, mb):
            ap, cp = params
            obs = mb["obs"]
            obs = {**obs, "action_mask": obs["action_mask"]}
            lg = self.logits(ap, obs)
            logp_all = jax.nn.log_softmax(lg)
            w = (obs["active"] & (obs["action_mask"].sum(-1) > 1)).astype(jnp.float32)
            wsum = jnp.maximum(w.sum(), 1.0)
            v = self.value(cp, obs)
            vloss = 0.5 * jnp.mean((v - mb["ret"]) ** 2)
            if mode == "bc":
                ce = -jnp.take_along_axis(logp_all, mb["ta"][..., None], -1)[..., 0]
                aloss = (ce * w).sum() / wsum
                acc = ((jnp.argmax(lg, -1) == mb["ta"]) * w).sum() / wsum
                return aloss + tc.vf_coef * vloss, {"bc_loss": aloss, "bc_acc": acc, "v_loss": vloss}
            logp = jnp.take_along_axis(logp_all, mb["a"][..., None], -1)[..., 0]
            a = mb["adv"]
            am = (a * w).sum() / wsum
            asd = jnp.sqrt((((a - am) ** 2) * w).sum() / wsum + 1e-8)
            a = (a - am) / asd
            ratio = jnp.exp(logp - mb["logp"])
            pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps) * a)
            pg = (pg * w).sum() / wsum
            p = jnp.exp(logp_all)
            ent = (-(p * jnp.where(obs["action_mask"], logp_all, 0.0)).sum(-1) * w).sum() / wsum
            bl = jax.lax.stop_gradient(jax.nn.log_softmax(self.logits(rs["bc"], obs)))
            kl = ((jnp.exp(bl) * jnp.where(obs["action_mask"], bl - logp_all, 0.0)).sum(-1) * w).sum() / wsum
            total = pg - tc.ent_coef * ent + kl_w * kl + tc.vf_coef * vloss
            return total, {"pg": pg, "entropy": ent, "kl_bc": kl, "v_loss": vloss}

        def epoch(c, _):
            ap, cp, aopt, copt, rng = c
            rng, k = jax.random.split(rng)
            mbs = self._minibatches(k, batch)

            def mb_step(c2, mb):
                ap, cp, aopt, copt = c2
                (_, info), (ga, gc) = jax.value_and_grad(loss_fn, has_aux=True)((ap, cp), mb)
                ua, aopt2 = self.atx.update(ga, aopt, ap)
                uc, copt = self.ctx.update(gc, copt, cp)
                on = actor_on if mode == "ppo" else True
                ap = tmap(lambda p, u: jnp.where(on, p + u, p), ap, ua)
                aopt = tmap(lambda n_, o: jnp.where(on, n_, o), aopt2, aopt)
                cp = optax.apply_updates(cp, uc)
                return (ap, cp, aopt, copt), info

            (ap, cp, aopt, copt), info = jax.lax.scan(mb_step, (ap, cp, aopt, copt), mbs)
            return (ap, cp, aopt, copt, rng), info

        n_ep = tc.bc_epochs if mode == "bc" else tc.ppo_epochs
        rng, k = jax.random.split(rs["rng"])
        (ap, cp, aopt, copt, _), info = jax.lax.scan(
            epoch, (rs["ap"], rs["cp"], rs["aopt"], rs["copt"], k), None, n_ep)
        rs = {**rs, "ap": ap, "cp": cp, "aopt": aopt, "copt": copt, "rng": rng}
        return rs, tmap(jnp.mean, info)

    def chunk(self, rs, its, betas, mode: str):
        """Run len(its) updates in one jit call."""
        def body(rs, x):
            it, beta = x
            rs, traj, stats = self.rollout(rs, beta, teacher=(mode == "bc"))
            rs, info = self.update(rs, traj, it, mode)
            return rs, {**stats, **info}
        return jax.lax.scan(body, rs, (its, betas))


def load_init(rs, src: Path, seed: int, grid: int):
    """Replace actor and critic params with a saved PPO checkpoint; optimizer state stays fresh."""
    new = dict(rs)
    for key, name in (("ap", "actor"), ("cp", "critic")):
        f = src / f"seed{seed}" / f"g{grid}_ppo_{name}.msgpack"
        tmpl = tmap(lambda x: x[0], rs[key])
        loaded = serialization.from_bytes(tmpl, f.read_bytes())
        new[key] = tmap(lambda x: jnp.asarray(x)[None], loaded)
    return new


def save(path: Path, tree):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(jax.device_get(tree)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1, help="vmapped seeds; 1 is fastest on GPU")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--ring", type=int, default=4)
    ap.add_argument("--grids", default="32,64,96")
    ap.add_argument("--bc-iters", default="100,60,60", help="BC updates per grid stage (0 = no cloning)")
    ap.add_argument("--ppo-updates", default="200,200,400", help="PPO updates per grid stage")
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--bc-epochs", type=int, default=2)
    ap.add_argument("--kl-fade", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--init-dir", default="", help="run dir with seed*/g*_ppo_{actor,critic}.msgpack to continue from")
    ap.add_argument("--init-grid", type=int, default=96)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--t-turn", type=float, default=EnvConfig.t_turn, help="seconds per 45 degree turn (PX4 SIH: 1.02)")
    ap.add_argument("--prior-kinds", default="0,1,2", help="training prior types; 4 = real bank, repeat to weight")
    ap.add_argument("--real-bank", default="", help="comma list of region npz files for prior type 4")
    ap.add_argument("--save-every", type=float, default=0.0, help="minutes between mid-stage checkpoints (0 = stage end only)")
    ap.add_argument("--max-minutes", type=float, default=1e9, help="wall clock cap; stops cleanly")
    args = ap.parse_args()

    grids = [int(g) for g in args.grids.split(",")]
    bc_iters = [int(x) for x in args.bc_iters.split(",")]
    ppo_updates = [int(x) for x in args.ppo_updates.split(",")]
    use_bc = sum(bc_iters) > 0
    tc = TrainConfig(seed=args.seed, n_seeds=args.n_seeds, n_envs=args.n_envs, rollout_len=args.rollout_len,
                     ppo_epochs=args.ppo_epochs, ppo_minibatches=args.minibatches,
                     bc_epochs=args.bc_epochs, kl_fade=args.kl_fade, lr=args.lr)
    if not use_bc:
        tc = TrainConfig(**{**tc.__dict__, "kl_coef": 0.0, "critic_warmup": 0})
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"args_seed{args.seed}.json").write_text(json.dumps({**vars(args), "train_config": {k: str(v) for k, v in tc.__dict__.items()}}, indent=1))
    log = open(out / f"metrics_seed{args.seed}.jsonl", "a")
    print("devices:", jax.devices(), flush=True)

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_seeds)
    rs = None
    t_start = time.time()
    ppo_it = 0
    bc_total = sum(bc_iters)
    bc_it = 0
    for si, g in enumerate(grids):
        real = [x for x in args.real_bank.split(",") if x]
        env = make_env(g, args.ring, real_paths=real, t_turn=args.t_turn,
                       train_prior_kinds=tuple(int(x) for x in args.prior_kinds.split(",")))
        tr = Trainer(env, tc)
        if rs is None:
            rs = jax.jit(jax.vmap(tr.init))(keys)
            if args.init_dir:
                rs = load_init(rs, Path(args.init_dir), args.seed, args.init_grid)
        else:  # keep params and optimizer state, new envs for the new grid size
            rs = jax.jit(jax.vmap(lambda k, r: {**tr.fresh_runner(k, r["ap"], r["cp"]),
                                                "aopt": r["aopt"], "copt": r["copt"], "bc": r["bc"]}))(
                jax.random.split(jax.random.PRNGKey(args.seed + 100 + si), args.n_seeds), rs)
        for mode, n_up in (("bc", bc_iters[si]), ("ppo", ppo_updates[si])):
            if n_up == 0:
                continue
            fn = jax.jit(jax.vmap(partial(tr.chunk, mode=mode), in_axes=(0, None, None)))
            bar = tqdm(total=n_up, desc=f"g{g} {mode}", mininterval=5)
            done_up = 0
            t_save = time.time()
            while done_up < n_up:
                k = min(args.chunk, n_up - done_up)
                if mode == "bc":
                    its = jnp.arange(bc_it, bc_it + k)
                    betas = jnp.minimum(its / max(bc_total, 1), 1.0) * tc.bc_beta_end
                else:
                    its = jnp.arange(ppo_it, ppo_it + k)
                    betas = jnp.zeros(k)
                t0 = time.time()
                rs, info = fn(rs, its, betas)
                info = jax.device_get(info)
                dt = time.time() - t0
                steps = k * tc.rollout_len * tc.n_envs * args.n_seeds
                rec = {"grid": g, "mode": mode, "it": int(its[-1]), "sps": steps / dt,
                       "minutes": (time.time() - t_start) / 60}
                rec.update({f"{kk}": [float(x) for x in np.asarray(v)[:, -1]] for kk, v in info.items()})
                log.write(json.dumps(rec) + "\n")
                log.flush()
                bar.update(k)
                bar.set_postfix(disc=np.mean(rec["ep_disc"]), sps=f"{rec['sps']:,.0f}")
                done_up += k
                if mode == "bc":
                    bc_it += k
                else:
                    ppo_it += k
                if args.save_every and (time.time() - t_save) / 60 > args.save_every:
                    t_save = time.time()
                    for s in range(args.n_seeds):
                        for key, nm in (("ap", "actor"), ("cp", "critic")):
                            save(out / f"seed{args.seed + s}" / f"g{g}_{mode}_{nm}.msgpack", tmap(lambda x: x[s], rs[key]))
                if (time.time() - t_start) / 60 > args.max_minutes:
                    break
            bar.close()
            if mode == "bc":
                rs = {**rs, "bc": rs["ap"]}
            for s in range(args.n_seeds):
                save(out / f"seed{args.seed + s}" / f"g{g}_{mode}_actor.msgpack", tmap(lambda x: x[s], rs["ap"]))
                save(out / f"seed{args.seed + s}" / f"g{g}_{mode}_critic.msgpack", tmap(lambda x: x[s], rs["cp"]))
            if (time.time() - t_start) / 60 > args.max_minutes:
                print("time cap reached", flush=True)
                return
    print(f"done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
