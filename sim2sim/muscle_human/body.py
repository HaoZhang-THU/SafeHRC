from __future__ import annotations

import functools
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from muscle_supermimic.assets.robots import get_myofullbody_cfg
from muscle_supermimic.assets.robots.myofullbody.myofullbody_cfg import _disable_fingers, get_spec
from muscle_supermimic.tasks.tracking.config.myofullbody.env_cfgs import myofullbody_flat_tracking_env_cfg

BOARD_L, BOARD_W, BOARD_T = 2.0, 1.0, 0.10
FINGERS = ("2", "3", "4", "5")
FALL_HEIGHT, FALL_UP = 0.55, 0.5


def finger_angles(curl: dict) -> dict[str, float]:
    out = {}
    for side in "rl":
        c1, c2 = curl[side]
        for k in FINGERS:
            out[f"mcp{k}_flexion_{side}"] = c1
            out[f"pm{k}_flexion_{side}"] = c2
            out[f"md{k}_flexion_{side}"] = 0.6 * c2
        out[f"mp_flexion_{side}"] = 0.3 * c2
        out[f"ip_flexion_{side}"] = -0.3 * c2
    return out


def _qmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def bake_joints(spec: mujoco.MjSpec, angles: dict[str, float]) -> None:
    for name, th in angles.items():
        j = spec.joint(name)
        body = j.parent
        a = np.asarray(j.axis, float)
        a = a / np.linalg.norm(a)
        qa = np.r_[np.cos(th / 2), np.sin(th / 2) * a]
        p = np.asarray(j.pos, float)
        q0 = np.asarray(body.quat, float)
        body.pos = np.asarray(body.pos, float) + _qmat(q0) @ (p - _qmat(qa) @ p)
        w1, x1, y1, z1 = q0
        w2, x2, y2, z2 = qa
        body.quat = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def hooked_spec(curl: dict, self_collisions: bool = True) -> mujoco.MjSpec:
    spec = get_spec(self_collisions=self_collisions, disable_fingers=False)
    bake_joints(spec, finger_angles(curl))
    _disable_fingers(spec)
    return spec


def board_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name="board")
    b.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOARD_L / 2, BOARD_W / 2, BOARD_T / 2],
               contype=0, conaffinity=1, friction=[1.0, 0.005, 0.0001], rgba=[0.95, 0.65, 0.15, 1.0])
    return spec


def fallen(env, command_name: str) -> torch.Tensor:
    S = env.command_manager.get_term(command_name).transition_ctx().S
    return torch.maximum(FALL_HEIGHT - S[0][..., 2] > 0.0, FALL_UP - S[1][..., 2, 2] > 0.0)


def env_cfg(motion_file: Path, curl: dict) -> ManagerBasedRlEnvCfg:
    cfg = myofullbody_flat_tracking_env_cfg()
    for k in ("body_pos_dev", "root_pos_dev", "root_ori_dev", "time_out"):
        cfg.terminations.pop(k)
    cfg.terminations["fall"] = TerminationTermCfg(func=fallen, params={"command_name": "motion"})
    robot = get_myofullbody_cfg(self_collisions=True, disable_fingers=True)
    robot.spec_fn = functools.partial(hooked_spec, curl, True)
    cfg.scene.entities = {"robot": robot, "board": EntityCfg(spec_fn=board_spec)}
    cmd = cfg.commands["motion"]
    cmd.motion_file, cmd.sampling_mode = str(motion_file), "start"
    cfg.scene.num_envs = 1
    cfg.sim.nconmax = 160
    cfg.episode_length_s, cfg.auto_reset = 1e9, False
    return cfg
