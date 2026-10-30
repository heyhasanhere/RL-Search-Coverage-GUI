"""Trained search agents the bridge can fly the search drone with, all on Unity's 936-value observation.

Drop-in: any of these anywhere under models/ is found on the next scan (Catalog):
  *.onnx                    ML-Agents export (PPO); deterministic continuous actions
  <dir>/model.zip           Stable-Baselines3 checkpoint (DQN or an actor-critic such as TRPO) with
                            <dir>/vecnormalize.pkl; DQN also needs <dir>/action_mapping.json
Files that do not load or take another observation are skipped. The SB3 checkpoints run in numpy:
their weights and normalization are read without torch or SB3.
"""

import io
import json
import pickle
import re
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
          "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


class OnnxPilot:
    """ML-Agents ONNX export."""

    def __init__(self, path, n_obs):
        import onnxruntime as ort          # only needed when the agent flies
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        if inp.shape[-1] != n_obs:
            raise ValueError(f"{Path(path).name} takes {inp.shape[-1]} observations, the build gives {n_obs}")
        if "deterministic_continuous_actions" not in {o.name for o in self.sess.get_outputs()}:
            raise ValueError(f"{Path(path).name} is not an ML-Agents export")
        self.inp = inp.name

    def act(self, obs):
        a = self.sess.run(["deterministic_continuous_actions"], {self.inp: np.asarray(obs, np.float32)[None]})[0][0]
        return np.clip(a, -1, 1).tolist()


class _Weights(pickle.Unpickler):
    """A torch state dict (torch.save zip) as numpy arrays."""

    def __init__(self, z: zipfile.ZipFile, prefix: str):
        super().__init__(io.BytesIO(z.read(f"{prefix}/data.pkl")))
        self.z, self.prefix = z, prefix

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *_: data
        if module == "torch" and name in DTYPES:
            return name
        raise pickle.UnpicklingError(f"unexpected {module}.{name} in a state dict")

    def persistent_load(self, pid):
        _, kind, key, _, _ = pid
        return np.frombuffer(self.z.read(f"{self.prefix}/data/{key}"), DTYPES[kind])


def _tensor(storage, offset, size, stride, *_):
    a = storage[offset:]
    return np.lib.stride_tricks.as_strided(a, tuple(size), tuple(s * a.itemsize for s in stride)).copy()


def _state_dict(model_zip: Path) -> dict:
    with zipfile.ZipFile(model_zip) as outer:
        inner = zipfile.ZipFile(io.BytesIO(outer.read("policy.pth")))
        prefix = inner.namelist()[0].split("/")[0]
        return _Weights(inner, prefix).load()


class _Stub:
    """Stands in for any SB3 or gymnasium class in vecnormalize.pkl; keeps its attributes."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


class _Normalize(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("numpy", "builtins", "collections", "copyreg", "_codecs"):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


def _layers(sd: dict, prefix: str) -> list:
    """Weights and biases of the Linear layers under `prefix`, in order."""
    idx = sorted({int(m.group(1)) for k in sd if (m := re.fullmatch(re.escape(prefix) + r"(\d+)\.weight", k))})
    return [(sd[f"{prefix}{i}.weight"], sd[f"{prefix}{i}.bias"]) for i in idx]


def _mlp(x, layers, act, last_act=False):
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1 or last_act:
            x = act(x)
    return x


class Sb3Pilot:
    """Stable-Baselines3 DQN or actor-critic checkpoint with its observation normalization."""

    def __init__(self, folder, n_obs):
        folder = Path(folder)
        data = json.loads(zipfile.ZipFile(folder / "model.zip").read("data"))
        sd = _state_dict(folder / "model.zip")
        with open(folder / "vecnormalize.pkl", "rb") as f:
            vn = _Normalize(f).load()
        rms = vn.obs_rms
        self.mean, self.std = np.asarray(rms.mean, np.float64), np.sqrt(np.asarray(rms.var, np.float64) + vn.epsilon)
        self.clip, self.norm = float(vn.clip_obs), bool(getattr(vn, "norm_obs", True))
        if self.mean.shape[-1] != n_obs:
            raise ValueError(f"{folder.name} takes {self.mean.shape[-1]} observations, the build gives {n_obs}")
        dqn = "q_net.q_net.0.weight" in sd
        # SB3's default activation: ReLU for DQN, tanh for actor-critic policies
        act = re.search(r"activation\.(\w+)", str(data.get("policy_kwargs", "")))
        relu = act.group(1) == "ReLU" if act else dqn
        self.hidden = (lambda x: np.maximum(x, 0)) if relu else np.tanh
        if dqn:                                   # argmax of the Q values picks one of the mapped actions
            self.q = _layers(sd, "q_net.q_net.")
            self.actions = np.array(json.loads((folder / "action_mapping.json").read_text())["actions"], float)
        else:                                     # the action mean of the policy network
            self.q = None
            self.pi = _layers(sd, "mlp_extractor.policy_net.")
            self.head = (sd["action_net.weight"], sd["action_net.bias"])

    def act(self, obs):
        x = np.asarray(obs, np.float64)
        if self.norm:
            x = np.clip((x - self.mean) / self.std, -self.clip, self.clip)
        if self.q is not None:
            return self.actions[int(np.argmax(_mlp(x, self.q, self.hidden)))].tolist()
        h = _mlp(x, self.pi, self.hidden, last_act=True)
        return np.clip(h @ self.head[0].T + self.head[1], -1, 1).tolist()


def label(path: Path) -> str:
    """Short name for the UI: algorithm and training steps, e.g. "PPO 13M"; the name when it has neither."""
    if path.suffix == ".onnx":
        algo = next((a for a in ("sac", "poca", "ppo") if a in path.stem.lower()), "ppo")
        steps = [s for s in re.findall(r"\d+", path.stem) if int(s) >= 100_000]
        if not steps and not re.search(r"\d+m\b", path.stem.lower()):
            return path.stem
        n = int(steps[-1]) if steps else int(re.findall(r"(\d+)m\b", path.stem.lower())[-1]) * 1_000_000
    else:
        meta = path / "metadata.json"
        m = json.loads(meta.read_text()) if meta.is_file() else {}
        found = re.search(r"(dqn|trpo|ppo|a2c|sac|td3|qrdqn|tqc|ars)", path.name.lower())
        algo = m.get("contract", {}).get("algorithm") or (found.group(1) if found else "sb3")
        steps = [str(m["steps"])] if "steps" in m else re.findall(r"step_(\d+)", path.name)
        n = int(steps[-1]) if steps else 0
    return f"{algo.upper()} {round(n / 1e6)}M" if n else algo.upper()


def load(path: Path, n_obs: int):
    return OnnxPilot(path, n_obs) if path.suffix == ".onnx" else Sb3Pilot(path, n_obs)


class Catalog:
    """The agents under `root`, rescanned on each call; a file is loaded again only when it changes."""

    def __init__(self, root):
        self.root = Path(root)
        self.loaded: dict[Path, tuple] = {}    # path -> (mtime, pilot or None, error)

    def _candidates(self):
        if not self.root.is_dir():
            return []
        onnx = [p for p in self.root.rglob("*.onnx") if p.is_file()]
        sb3 = [p.parent for p in self.root.rglob("model.zip") if (p.parent / "vecnormalize.pkl").is_file()]
        return sorted(onnx + sb3)

    def scan(self, n_obs: int) -> dict[str, dict]:
        """Agent id (path under root, no suffix) -> {"path", "label", "pilot"} for every agent that loads."""
        found = {}
        for p in self._candidates():
            files = [p] if p.is_file() else [f for f in p.iterdir() if f.is_file()]
            mtime = max(f.stat().st_mtime for f in files)
            if self.loaded.get(p, (None,))[0] != mtime:
                try:
                    self.loaded[p] = (mtime, load(p, n_obs), "")
                except Exception as exc:          # a file still being copied, or another kind of model
                    self.loaded[p] = (mtime, None, str(exc))
            _, pilot, _ = self.loaded[p]
            if pilot is not None:
                rel = p.relative_to(self.root)
                found[str(rel.with_suffix("") if p.is_file() else rel)] = {"path": p, "label": label(p), "pilot": pilot}
        # two agents with the same label: add their folders
        labels = [a["label"] for a in found.values()]
        for a in found.values():
            if labels.count(a["label"]) > 1:
                a["label"] += f" ({a['path'].parent.name})"
        return found
