from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import mujoco
import musclemimic_models
import numpy as np
from scipy.optimize import minimize
from muscle_supermimic.assets.robots.myofullbody.myofullbody_cfg import get_spec

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = Path(os.environ.get("MUSCLEMIMIC_HOME", "~/.musclemimic")).expanduser() / "caches/AMASS/MyoFullBody/gmr"
FPS = 100.0
N_PHASE = 128
F_MIN, F_MAX, TURN_AMP = 0.55, 1.25, 0.45
CYCLES = {
    "forward": ("KIT/314/walking_medium07_poses", 342, 471),
    "backward": ("KIT/4/WalkingStraightBackwards08_poses", 166, 310),
    "left": ("CMU/143/143_41_poses", 1705, 1795),
}
STAND = ("KIT/314/walking_medium07_poses", 20)
LEG = ["hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle"]
SPINE_DEG = {"flex_extension": -4.8, "lat_bending": 0.0, "axial_rotation": 0.0}
ARM_POSE = {"elv_angle": 1.57, "shoulder_elv": 0.428, "shoulder_rot": 0.0, "elbow_flex": 0.0,
            "pro_sup": -1.57, "deviation": 0.0, "flexion": -0.008}

BOARD_L, BOARD_W, BOARD_T, BOARD_Z = 2.0, 1.0, 0.10, 0.93
HUMAN_OFFSET = 0.12
HAND_DX = 0.22
HOLD_IN, TIP_GAP = (0.015, 0.06), (0.0005, 0.003)
CLEAR_HAND, CLEAR_ARM = 0.0005, 0.01
ARM = tuple(ARM_POSE)
FINGERS = ("2", "3", "4", "5")
HAND_POINTS = ("lunate_r", "thirdmc_r", "fifthmc_r", "lunate_l", "thirdmc_l", "fifthmc_l")


def load_clip(cache: Path, name: str) -> dict:
    d = np.load(cache / f"{name}.npz", allow_pickle=True)
    return {"qpos": np.asarray(d["qpos"], dtype=np.float64), "joint_names": [str(n) for n in d["joint_names"]],
            "jnt_type": np.asarray(d["jnt_type"])}


def qpos_layout(names: list[str], types: np.ndarray) -> dict[str, int]:
    adr, a = {}, 0
    for n, t in zip(names, types):
        adr[n] = a
        a += {0: 7, 1: 4, 2: 1, 3: 1}[int(t)]
    return adr


def clip_model(names: list[str]) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(musclemimic_models.get_xml_path("myofullbody").as_posix())
    keep = set(names)
    for j in list(spec.joints):
        if j.name not in keep:
            spec.delete(j)
    for e in list(spec.equalities):
        if e.name1 not in keep or (e.name2 and e.name2 not in keep):
            spec.delete(e)
    return spec.compile()


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_z(a: float) -> np.ndarray:
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def heading(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(-(1 - 2 * (x * x + z * z)), -(2 * (x * y - w * z))))


class Couplings:
    def __init__(self, model: mujoco.MjModel, adr: dict[str, int]):
        self.rows = []
        for e in range(model.neq):
            if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or model.eq_obj2id[e] < 0:
                continue
            j1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.eq_obj1id[e])
            j2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.eq_obj2id[e])
            if j1 in adr and j2 in adr:
                self.rows.append((adr[j1], adr[j2], model.eq_data[e][:5].copy()))

    def __call__(self, q: np.ndarray) -> np.ndarray:
        for a1, a2, c in self.rows:
            x = q[..., a2]
            q[..., a1] = c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3 + c[4] * x ** 4
        return q


def mirror(model: mujoco.MjModel, names: list[str], adr: dict[str, int], qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    M = np.diag([-1.0, 1.0, 1.0])
    out = qpos.copy()
    out[:, 0] = -qpos[:, 0]
    out[:, 3:7] = qpos[:, 3:7] * np.array([1.0, 1.0, -1.0, -1.0])
    for n in names[1:]:
        p = n[:-2] + ("_l" if n.endswith("_r") else "_r") if n[-2:] in ("_r", "_l") else n
        j, k = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, x) for x in (n, p))
        dot = float(data.xaxis[k] @ (M @ data.xaxis[j]))
        hinge = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        out[:, adr[p]] = qpos[:, adr[n]] * (-np.sign(dot) if hinge else np.sign(dot))
    return out


def contacts(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{b}_{s}") for s in "rl" for b in ("calcn", "toes")]
    pts = np.zeros((len(q), 4, 3))
    for i, f in enumerate(q):
        data.qpos[:] = f
        mujoco.mj_kinematics(model, data)
        pts[i] = data.xpos[feet]
    ground = np.percentile(pts[:, :, 2], 3)
    return np.stack([pts[:, 2 * k:2 * k + 2, 2].min(1) < ground + 0.02 for k in range(2)], 1)


def cycle(qpos: np.ndarray, contact: np.ndarray, cols: list[int]) -> dict:
    T = len(qpos)
    h = np.unwrap([heading(x) for x in qpos[:, 3:7]])
    c, s = np.cos(-h[0]), np.sin(-h[0])
    d = qpos[-1, :2] - qpos[0, :2]
    disp = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    period = (T - 1) / FPS
    u = np.linspace(0, 1, T)
    ph = np.linspace(0, 1, N_PHASE, endpoint=False)
    legs = qpos[:, cols]
    rest = np.stack([np.interp(ph, u, legs[:, j]) for j in range(legs.shape[1])], 1)
    rest = rest - np.outer(ph, legs[-1] - rest[0])
    down = np.stack([np.interp(ph, u, contact[:, j].astype(float)) for j in range(2)], 1) > 0.5
    return {"rest": rest, "contact": down, "speed": float(np.linalg.norm(disp) / period), "period": period}


def gait_table(cache: Path) -> tuple[dict, np.ndarray, list[str], np.ndarray]:
    first = load_clip(cache, CYCLES["forward"][0])
    names, types = first["joint_names"], first["jnt_type"]
    adr = qpos_layout(names, types)
    model = clip_model(names)
    joints = [f"{j}_{s}" for s in "rl" for j in LEG]
    cols = [adr[j] for j in joints]
    segs = {n: load_clip(cache, c)["qpos"][a:b + 1] for n, (c, a, b) in CYCLES.items()}
    segs["right"] = mirror(model, names, adr, segs["left"])
    cyc = {n: cycle(q, contacts(model, q), cols) for n, q in segs.items()}
    con = cyc["right"]["contact"][:, 0]
    td = [i for i in range(1, N_PHASE) if con[i] and not con[i - 1]]
    k = td[0] if td else 0
    cyc["right"]["rest"] = np.roll(cyc["right"]["rest"], -k, axis=0)
    cyc["right"]["contact"] = np.roll(cyc["right"]["contact"], -k, axis=0)

    stand = load_clip(cache, STAND[0])["qpos"][STAND[1]].copy()
    couple = Couplings(model, adr)
    for j, deg in SPINE_DEG.items():
        stand[adr[j]] = np.radians(deg)
    couple(stand)
    h0 = heading(stand[3:7])
    tilt = quat_mul(quat_conj(quat_z(h0 + np.pi / 2)), stand[3:7])
    tilt = tilt if tilt[0] >= 0 else -tilt
    q = stand.copy()
    q[:2] = 0.0
    q[3:7] = quat_mul(quat_z(np.pi / 2), tilt / np.linalg.norm(tilt))
    for s in "rl":
        for j, v in ARM_POSE.items():
            q[adr[f"{j}_{s}"]] = v
    couple(q)

    order = ["forward", "left", "backward", "right"]
    table = {"rest": np.stack([cyc[n]["rest"] for n in order]).astype(np.float32),
             "contact": np.stack([cyc[n]["contact"] for n in order]).astype(np.float32),
             "stand": stand[cols].astype(np.float32),
             "speed": np.array([cyc[n]["speed"] for n in order], np.float32),
             "period": np.array([cyc[n]["period"] for n in order], np.float32),
             "joints": np.array(joints), "f_min": F_MIN, "f_max": F_MAX, "turn_amp": TURN_AMP}
    return table, q, names, types


def full_qpos(m: mujoco.MjModel, qg: np.ndarray, names: list[str], types: np.ndarray, adr: dict[str, int]) -> np.ndarray:
    q = m.qpos0.copy()
    for n, t in zip(names, types):
        k = 7 if int(t) == 0 else 1
        a = m.jnt_qposadr[m.joint(n).id]
        q[a:a + k] = qg[adr[n]:adr[n] + k]
    return q


def board_frame(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = heading(q[3:7])
    centre = np.r_[q[:2], 0.0] + (BOARD_W / 2 + HUMAN_OFFSET) * np.array([np.cos(h), np.sin(h), 0.0])
    centre[2] = BOARD_Z
    yaw = h + np.pi / 2
    return centre, np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])


def grip_design(q_stand: np.ndarray, names: list[str], types: np.ndarray, seed: int) -> dict:
    spec = get_spec(self_collisions=False, disable_fingers=False)
    b = spec.worldbody.add_body(name="board")
    b.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOARD_L / 2, BOARD_W / 2, BOARD_T / 2],
               contype=0, conaffinity=0)
    m = spec.compile()
    d = mujoco.MjData(m)
    gm = clip_model(names)
    adr = qpos_layout(names, types)
    couple = Couplings(gm, adr)

    centre, Rb = board_frame(q_stand)
    bid = m.body("board").id
    m.body_pos[bid], m.body_quat[bid] = centre, 0.0
    mujoco.mju_mat2Quat(m.body_quat[bid], Rb.flatten())
    bgeom = m.geom("board").id
    fr = np.zeros(6)

    def dist(g: int) -> float:
        return mujoco.mj_geomDistance(m, d, g, bgeom, 0.1, fr)

    out = {}
    for side, sx in (("r", -1.0), ("l", 1.0)):
        lun = m.body(f"lunate_{side}").id
        sub = set()
        for bb in range(m.nbody):
            a = bb
            while a > 0 and a != lun:
                a = m.body_parentid[a]
            if a == lun:
                sub.add(bb)
        arm_bodies = {m.body(f"{n}_{side}").id for n in ("humerus", "ulna", "radius")}
        hand_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub]
        arm_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_bodies]
        tips = [m.body(f"{k}distph_{side}").id for k in FINGERS]
        tip_geoms = [g for g in hand_geoms if m.geom_bodyid[g] in tips]
        jarm = [m.jnt_qposadr[m.joint(f"{n}_{side}").id] for n in ARM]
        garm = [adr[f"{n}_{side}"] for n in ARM]
        rng = np.array([m.jnt_range[m.joint(f"{n}_{side}").id] for n in ARM])
        jf = [m.jnt_qposadr[m.joint(f"{p}{k}_flexion_{side}").id] for k in FINGERS for p in ("mcp", "pm")]
        jd = [m.jnt_qposadr[m.joint(f"md{k}_flexion_{side}").id] for k in FINGERS]
        jt = [m.jnt_qposadr[m.joint(f"{n}_{side}").id] for n in ("mp_flexion", "ip_flexion")]
        base = full_qpos(m, q_stand, names, types, adr)

        def pose(x: np.ndarray) -> None:
            qg = q_stand.copy()
            qg[garm] = np.clip(x[:7], rng[:, 0], rng[:, 1])
            couple(qg)
            q = full_qpos(m, qg, names, types, adr)
            c1, c2 = float(np.clip(x[7], 0.0, 1.6)), float(np.clip(x[8], 0.0, 1.6))
            q[jf[0::2]], q[jf[1::2]], q[jd], q[jt] = c1, c2, 0.6 * c2, (0.3 * c2, -0.3 * c2)
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)

        def cost(x: np.ndarray) -> float:
            pose(x)
            e = 0.0
            for g in hand_geoms:
                e += 1e6 * max(0.0, CLEAR_HAND - dist(g)) ** 2
            for g in arm_geoms:
                e += 1e4 * max(0.0, CLEAR_ARM - dist(g)) ** 2
            for t in tips:
                p = Rb.T @ (d.xpos[t] - centre)
                inside = BOARD_W / 2 - p[1]
                e += 1e5 * (max(0.0, HOLD_IN[0] - inside) ** 2 + max(0.0, inside - HOLD_IN[1]) ** 2)
                e += 1e5 * max(0.0, p[2] + BOARD_T / 2) ** 2
            for t in tips:
                gap = min(dist(g) for g in tip_geoms if m.geom_bodyid[g] == t)
                e += 1e5 * max(0.0, gap - TIP_GAP[1]) ** 2
            pl = Rb.T @ (d.xpos[lun] - centre)
            e += 10.0 * max(0.0, abs(pl[0] - sx * HAND_DX) - 0.08) ** 2
            e += 0.01 * float(np.sum((x[:7] - base[jarm]) ** 2))
            return e

        rs = np.random.default_rng(seed)
        lo = np.r_[rng[:, 0], 0.0, 0.0]
        hi = np.r_[rng[:, 1], 1.6, 1.6]
        X = lo + (hi - lo) * rs.random((40000, 9))
        C = np.array([cost(x) for x in X])
        best = None
        for i in np.argsort(C)[:20]:
            r = minimize(cost, X[i], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-8})
            if best is None or r.fun < best.fun:
                best = r
        pose(best.x)
        out[side] = {"arm": dict(zip(ARM, np.clip(best.x[:7], rng[:, 0], rng[:, 1]).round(4).tolist())),
                     "curl": [round(float(np.clip(best.x[7], 0, 1.6)), 4), round(float(np.clip(best.x[8], 0, 1.6)), 4)],
                     "lunate_pos": (Rb.T @ (d.xpos[lun] - centre)).round(4).tolist(),
                     "lunate_R": (Rb.T @ d.xmat[lun].reshape(3, 3)).round(5).tolist(),
                     "third_mc_pos": (Rb.T @ (d.xipos[m.body(f"thirdmc_{side}").id] - centre)).round(4).tolist()}
        print(side, "cost", f"{best.fun:.3g}", flush=True)
    return out


def hand_points(grip: dict, q_stand: np.ndarray, names: list[str], types: np.ndarray) -> None:
    m = get_spec(self_collisions=False, disable_fingers=False).compile()
    d = mujoco.MjData(m)
    adr = qpos_layout(names, types)
    q = q_stand.copy()
    for side in "rl":
        for n, v in grip[side]["arm"].items():
            q[adr[f"{n}_{side}"]] = v
    Couplings(clip_model(names), adr)(q)
    d.qpos[:] = full_qpos(m, q, names, types, adr)
    mujoco.mj_kinematics(m, d)
    h = heading(q[3:7])
    Y = np.array([[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])
    P = np.array([d.xpos[m.body(b).id] for b in HAND_POINTS])
    centre, Rb = board_frame(q)
    grip["hand_points_rel"] = ((P - d.xpos[m.body("pelvis").id]) @ Y).round(4).tolist()
    grip["hand_points_board"] = ((P - centre) @ Rb).round(4).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=CACHE, help="MuscleMimic's MyoFullBody GMR cache")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    table, q_stand, names, types = gait_table(args.cache)
    np.savez(DATA / "gait_table.npz", **table, stand_qpos=q_stand, joint_names=np.array(names), jnt_type=types)
    grip = grip_design(q_stand, names, types, args.seed)
    hand_points(grip, q_stand, names, types)
    (DATA / "grip_pose.json").write_text(json.dumps(grip, indent=1))
    print("wrote", DATA / "gait_table.npz", DATA / "grip_pose.json")


if __name__ == "__main__":
    main()
