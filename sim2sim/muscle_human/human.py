from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from multiprocessing.connection import Client
from pathlib import Path
import numpy as np
import torch
from body import env_cfg, hooked_spec
from policy import GaitTable, Policy

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FPS = 100
RESYNC = 10
KP, KYAW, W_MAX, V_MAX = 1.0, 1.0, 0.3, 0.3
FF_TAU = 0.3
GRIP_POINT_BOX = (np.array([-0.07, -0.07, -0.06]), np.array([0.14, 0.07, 0.09]))
HANDS = ("thirdmc_r", "thirdmc_l")
WRISTS = ("lunate_r", "lunate_l")
K_POS, D_POS, F_MAX = 3000.0, 100.0, 100.0
K_ROT, D_ROT, T_MAX = 20.0, 1.0, 5.0
RAMP_S = 1.0


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotvec(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R) - 1) / 2
    th = math.acos(max(-1.0, min(1.0, c)))
    if th < 1e-8:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2 * math.sin(th)) * v if th < math.pi - 1e-6 else th * np.sqrt(np.maximum(np.diag(R) + 1, 0) / 2)


class GripSpring:
    def __init__(self, dt: float) -> None:
        self.dt, self.t, self.prev = dt, 0.0, None

    def __call__(self, p: np.ndarray, R: np.ndarray, p_t: np.ndarray, R_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.zeros_like(p) if self.prev is None else (p - self.prev[0]) / self.dt
        w = np.zeros_like(p) if self.prev is None else np.array([rotvec(R[i] @ self.prev[1][i].T) for i in range(2)]) / self.dt
        self.prev = (p.copy(), R.copy())
        ramp = min(1.0, self.t / RAMP_S)
        self.t += self.dt
        F, T = np.zeros_like(p), np.zeros_like(p)
        for i in range(2):
            f = ramp * K_POS * (p_t[i] - p[i]) - D_POS * v[i]
            n = np.linalg.norm(f)
            F[i] = f * (F_MAX / n) if n > F_MAX else f
            tq = ramp * K_ROT * rotvec(R_t[i] @ R[i].T) - D_ROT * w[i]
            n = np.linalg.norm(tq)
            T[i] = tq * (T_MAX / n) if n > T_MAX else tq
        return F, T


def render_model(grip: dict, folder: Path) -> Path:
    import musclemimic_models

    spec = hooked_spec({s: grip[s]["curl"] for s in "rl"})
    base = musclemimic_models.get_xml_path("myofullbody").parent
    spec.meshdir = str((base / spec.meshdir).resolve())
    spec.texturedir = str((base / spec.texturedir).resolve())
    path = folder / "human.xml"
    path.write_text(spec.to_xml())
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--seconds", type=float, default=300.0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    conn = Client(("127.0.0.1", int(os.environ["MUSCLE_HUMAN_PORT"])),
                  authkey=bytes.fromhex(os.environ["MUSCLE_HUMAN_AUTHKEY"]))

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from tensordict import TensorDict

    from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx
    from muscle_supermimic.utils.rotations import yaw_mat

    gait = np.load(DATA / "gait_table.npz")
    grip = json.loads((DATA / "grip_pose.json").read_text())
    table = GaitTable(gait)
    policy = Policy(args.checkpoint)
    tmp = Path(tempfile.mkdtemp(prefix="muscle_human_"))
    n = int(args.seconds * FPS) + 180
    np.savez(tmp / "motion.npz", qpos=np.tile(gait["stand_qpos"], (n, 1)), joint_names=gait["joint_names"],
             jnt_type=gait["jnt_type"], frequency=float(FPS))

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg(tmp / "motion.npz", {s: grip[s]["curl"] for s in "rl"}),
                                               device="cpu"))
    u = env.unwrapped
    cmd = u.command_manager.get_term("motion")
    d, sm = u.sim.data, u.sim.mj_model
    joints = [j for j in range(sm.njnt) if sm.joint(j).name.startswith("robot/")]
    q_adr = torch.tensor(np.concatenate([np.arange(sm.jnt_qposadr[j], sm.jnt_qposadr[j] + (7 if sm.jnt_type[j] == 0 else 1))
                                         for j in joints]))
    v_adr = torch.arange(sm.jnt_dofadr[joints[0]], sm.jnt_dofadr[joints[0]] + 3)
    hand_ids = [sm.body(f"robot/{b}").id for b in HANDS]
    wrist_ids = [sm.body(f"robot/{b}").id for b in WRISTS]
    board = u.scene["board"]
    spring = GripSpring(1.0 / FPS)
    pts_b = np.array(grip["hand_points_board"]).reshape(2, 3, 3)
    base = np.array(grip["hand_points_rel"])
    memory = {"hands": None, "v_ff": None, "w_ff": 0.0}

    def state() -> dict:
        hands = d.xipos[0, hand_ids].numpy().astype(np.float64)
        hv = np.zeros_like(hands) if memory["hands"] is None else (hands - memory["hands"]) * FPS
        memory["hands"] = hands
        return {"qpos": d.qpos[0, q_adr].numpy().astype(np.float64), "vel": d.qvel[0, v_adr].numpy().astype(np.float64),
                "hands": hands, "hands_v": hv}

    def goal(msg: dict) -> torch.Tensor:
        v_t, w_t = np.asarray(msg["v"]), float(msg["w"])
        if memory["v_ff"] is None:
            memory["v_ff"], memory["w_ff"] = v_t.copy(), w_t
        a = min(RESYNC / FPS / FF_TAU, 1.0)
        memory["v_ff"] = memory["v_ff"] + a * (v_t - memory["v_ff"])
        memory["w_ff"] = memory["w_ff"] + a * (w_t - memory["w_ff"])
        S = cmd.read_state()
        Y = yaw_mat(S[1])[0].numpy().astype(np.float64)
        body_h = math.atan2(Y[1, 0], Y[0, 0])
        v_b = Y[:2, :2].T @ (memory["v_ff"] + KP * (np.asarray(msg["tgt"]) - S[0][0, :2].numpy()))
        nb = np.linalg.norm(v_b)
        if nb > V_MAX:
            v_b *= V_MAX / nb
        w = float(np.clip(memory["w_ff"] + KYAW * wrap(float(msg["h"]) - body_h), -W_MAX, W_MAX))
        P0 = cmd.heads.fk(StepCtx(S, cmd.f))[0][0, 0].numpy().astype(np.float64)
        Rb = quat_mat(np.asarray(msg["board"][3:7]))
        pos = np.asarray(msg["grip"]["pos"])
        pts = np.array([[pos[k] + Rb @ (pts_b[k, j] - pts_b[k, 0]) for j in range(3)] for k in range(2)])
        rel = np.clip((pts.reshape(6, 3) - P0) @ Y, base + GRIP_POINT_BOX[0], base + GRIP_POINT_BOX[1])
        return torch.tensor(np.r_[v_b, w, rel.flatten()], dtype=torch.float32).unsqueeze(0)

    obs = TensorDict(u.reset(env_ids=torch.arange(1))[0], batch_size=[1])
    phase, vs = torch.zeros(1), torch.zeros(1, 3)
    g = None
    conn.send({**state(), "model": str(render_model(grip, tmp))})
    k = 0
    try:
        while True:
            msg = conn.recv()
            if msg.get("done"):
                break
            d.xfrc_applied[0] = 0.0
            d.xfrc_applied[0, hand_ids, :3] = torch.as_tensor(np.asarray(msg["F"]), dtype=d.xfrc_applied.dtype)
            bp, bq = np.asarray(msg["board"][:3]), np.asarray(msg["board"][3:7])
            board.write_mocap_pose_to_sim(torch.as_tensor(np.r_[bp, bq], dtype=torch.float32).unsqueeze(0))
            p_w = d.xpos[0, wrist_ids].numpy().astype(np.float64)
            R_w = d.xmat[0, wrist_ids].numpy().astype(np.float64).reshape(2, 3, 3)
            F_g, T_g = spring(p_w, R_w, np.asarray(msg["grip"]["pos"]), np.asarray(msg["grip"]["R"]))
            d.xfrc_applied[0, wrist_ids] = torch.as_tensor(np.c_[F_g, T_g], dtype=d.xfrc_applied.dtype)
            if k % RESYNC == 0:
                g = goal(msg)
            g = torch.cat((g[:, :21], table.goal(phase, vs)), -1)
            phase, vs = table.advance(phase, vs, g[:, :3], 1.0 / FPS)
            obs, _, dones, _ = env.step(policy(obs["policy"], g))
            k += 1
            if bool(dones[0]):
                conn.send({"fell": True})
                break
            conn.send(state())
    except EOFError:
        pass


if __name__ == "__main__":
    main()
