from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
G1_URDF = HERE / "assets" / "g1" / "g1_23dof_palmup.urdf"
G1_MJCF = HERE / "assets" / "g1" / "g1_23dof_palmup.xml"
LOW_LEVEL_POLICY = REPO / "isaac_sage" / "assets" / "policy_waist_tracking_g1_23dof.pt"

JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
DEFAULT_POS = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 11)
EFFORT = np.array([88, 88, 88, 139, 50, 50] * 2 + [88, 50, 50] + [25] * 8, dtype=np.float64)
ARMATURE = np.array([0.03] * 19 + [0.04, 0.03, 0.03, 0.03])
FRICTIONLOSS = 0.03
KP = np.array([100, 100, 100, 200, 20, 20] * 2 + [400] * 3 + [90, 60, 20, 60] * 2, dtype=np.float64)
KD = np.array([2.5, 2.5, 2.5, 5.0, 0.2, 0.1] * 2 + [5.0] * 3 + [2.0, 1.0, 0.4, 1.0] * 2)
LOCKED = np.arange(12, 23)

ROBOTS = ["robot_0", "robot_1", "robot_2"]
AGENTS = ["robot_0", "robot_1"]
NUM_WAYPOINTS = 5
OBS_CORE = 49
BOARD_L, BOARD_W, BOARD_T, BOARD_MASS = 2.0, 1.0, 0.10, 6.0
BOARD_Z0 = 0.97
GOAL_RADIUS = 0.15

FLOOR, BOARD, WALL = 1, 2, 4
ROBOT_BITS = (8, 16, 32)
FEET = 64
ALL_ROBOTS = sum(ROBOT_BITS)


class TaskCfg:
    dt = 0.005
    decimation = 200
    ll_decimation = 4
    ll_steps_per_hl = 50
    action_scale = 0.25
    filter_alpha = 0.3
    gait_frequency = 0.77
    max_target_dist = 0.2
    max_target_yaw = 0.3
    robot_offset = 0.20
    termination_height = 0.5
    board_drop_height = 0.70
    robot_collision_radius = 0.2
    lidar_rays = 36
    lidar_range = 5.0
    episode_length_s = 150.0
    board_friction = 3.0
    robot_friction = 1.0
    cone = "elliptic"
    impratio = 10.0
    noslip_iterations = 0
    board_condim = 3


def yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[0], q[1:]
    return v * (2.0 * w * w - 1.0) - np.cross(u, v) * w * 2.0 + u * (2.0 * np.dot(u, v))


def rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[0], q[1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def wrap(a: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _g1_mjcf() -> Path:
    if G1_MJCF.is_file():
        return G1_MJCF
    text = G1_URDF.read_text().replace(
        '<compiler meshdir="meshes" discardvisual="false"/>',
        '<compiler meshdir="meshes" discardvisual="false" fusestatic="false"/>')
    tmp = G1_URDF.with_name(f"_fused_off_{os.getpid()}.urdf")
    tmp.write_text(text)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    out = G1_MJCF.with_name(f"_{os.getpid()}.xml")
    mujoco.mj_saveLastXML(str(out), model)
    os.replace(out, G1_MJCF)
    return G1_MJCF


def build_model(obstacle_specs, cfg: TaskCfg = TaskCfg, n_robots: int = 3) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = cfg.dt
    spec.option.iterations = 50
    spec.option.ls_iterations = 20
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    spec.option.cone = (mujoco.mjtCone.mjCONE_ELLIPTIC if cfg.cone == "elliptic"
                        else mujoco.mjtCone.mjCONE_PYRAMIDAL)
    spec.option.impratio = cfg.impratio
    spec.option.noslip_iterations = cfg.noslip_iterations
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.stat.extent = 6.0
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]

    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                           rgb1=[0.32, 0.34, 0.36], rgb2=[0.26, 0.28, 0.30])
    spec.add_material(name="grid", texrepeat=[8, 8], reflectance=0.05).textures[
        mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=False)
    spec.worldbody.add_light(pos=[3, -3, 5], dir=[-0.4, 0.4, -1], diffuse=[0.4, 0.4, 0.4])
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            material="grid", contype=FLOOR, conaffinity=FEET, group=2,
                            friction=[1.0, 0.005, 0.0001])
    for i, (size, pos) in enumerate(obstacle_specs):
        spec.worldbody.add_geom(name=f"wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[s / 2 for s in size], pos=list(pos), rgba=[0.55, 0.45, 0.38, 1],
                                contype=WALL, conaffinity=BOARD | ALL_ROBOTS, group=2,
                                friction=[1.0, 0.005, 0.0001])

    board = spec.worldbody.add_body(name="board", pos=[0, 0, BOARD_Z0])
    board.add_freejoint(name="board_root")
    board.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[BOARD_L / 2, BOARD_W / 2, BOARD_T / 2], mass=BOARD_MASS,
                   rgba=[0.85, 0.62, 0.30, 1], contype=BOARD, conaffinity=FLOOR | WALL | ALL_ROBOTS,
                   group=2, friction=[cfg.board_friction, 0.005, 0.0001], condim=cfg.board_condim)

    mjcf = _g1_mjcf()
    for r in range(n_robots):
        child = mujoco.MjSpec.from_file(str(mjcf))
        child.meshdir = str(mjcf.parent / "meshes")
        child.body("pelvis").add_freejoint(name="root")
        frame = spec.worldbody.add_frame()
        spec.attach(child, prefix=f"r{r}_", frame=frame)

    model_spec_joints = {j.name: j for j in spec.joints}
    for r in range(n_robots):
        for i, name in enumerate(JOINTS):
            joint = model_spec_joints[f"r{r}_{name}"]
            joint.armature = float(ARMATURE[i])
            joint.frictionloss = FRICTIONLOSS
            joint.damping = 0.0
            act = spec.add_actuator(name=f"r{r}_{name}", target=f"r{r}_{name}",
                                    trntype=mujoco.mjtTrn.mjTRN_JOINT)
            act.gainprm[0] = 1.0
            act.forcelimited = 1
            act.forcerange = [-EFFORT[i], EFFORT[i]]

    model = spec.compile()
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        if not (len(body) > 3 and body[0] == "r" and body[1].isdigit() and body[2] == "_"):
            continue
        r = int(body[1])
        if model.geom_group[g] == 1:
            model.geom_contype[g] = model.geom_conaffinity[g] = 0
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_size[g, 0] <= 0.006:
            model.geom_contype[g], model.geom_conaffinity[g] = FEET, FLOOR
            model.geom_friction[g] = [1.0, 0.005, 0.0001]
            model.geom_solref[g] = [0.004, 1.0]
            model.geom_solimp[g] = [0.95, 0.99, 0.001, 0.5, 2.0]
            continue
        model.geom_contype[g] = ROBOT_BITS[r]
        model.geom_conaffinity[g] = BOARD | WALL | (ALL_ROBOTS & ~ROBOT_BITS[r])
        model.geom_friction[g] = [cfg.robot_friction, 0.005, 0.0001]
        model.geom_group[g] = 3
    return model


class TeamCarryEnv:
    def __init__(self, scene: dict, cfg: TaskCfg = TaskCfg, ll_policy: str | Path = LOW_LEVEL_POLICY,
                 seed: int = 0) -> None:
        self.cfg = cfg
        self.obstacles = [(tuple(s), tuple(p)) for s, p in scene["obstacle_specs"]]
        self.model = build_model(self.obstacles, cfg)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.ll_policy = torch.jit.load(str(ll_policy), map_location="cpu").eval()
        m = self.model

        self.qadr = np.array([[m.jnt_qposadr[m.joint(f"r{r}_{n}").id] for n in JOINTS] for r in range(3)])
        self.vadr = np.array([[m.jnt_dofadr[m.joint(f"r{r}_{n}").id] for n in JOINTS] for r in range(3)])
        self.act = np.array([[m.actuator(f"r{r}_{n}").id for n in JOINTS] for r in range(3)])
        self.root_q = np.array([m.jnt_qposadr[m.joint(f"r{r}_root").id] for r in range(3)])
        self.root_v = np.array([m.jnt_dofadr[m.joint(f"r{r}_root").id] for r in range(3)])
        self.board_q = m.jnt_qposadr[m.joint("board_root").id]
        self.board_v = m.jnt_dofadr[m.joint("board_root").id]
        self.feet = [[m.body(f"r{r}_{s}_ankle_roll_link").id for s in ("left", "right")] for r in range(3)]

        aabb = [[p[0] - s[0] / 2, p[0] + s[0] / 2, p[1] - s[1] / 2, p[1] + s[1] / 2] for s, p in self.obstacles]
        self.aabb = np.array(aabb, dtype=np.float64).reshape(-1, 4)
        self.ray_offsets = np.linspace(0.0, 2.0 * math.pi, cfg.lidar_rays + 1)[:-1]

        self.path_default = np.array(scene["path_waypoints"], dtype=np.float64)
        self.start_xy = np.array(scene.get("start_xy", scene["path_waypoints"][0]), dtype=np.float64)
        self.max_episode_length = math.ceil(cfg.episode_length_s / (cfg.dt * cfg.decimation))
        half_l, half_w, off = BOARD_L / 2, BOARD_W / 2, cfg.robot_offset
        self.default_robot = [((half_l + off, 0.0), math.pi), ((-(half_l + off), 0.0), 0.0),
                              ((0.0, half_w + off), -math.pi / 2)]
        self.anchors_local = np.array([[half_l + off, 0.0], [-(half_l + off), 0.0], [0.0, half_w + off]])
        self.corners_local = np.array([[sx * half_l, sy * half_w, BOARD_T / 2] for sx in (1, -1) for sy in (1, -1)])
        self.frame_hook = None

    def robot_pos(self, r: int) -> np.ndarray:
        return self.data.qpos[self.root_q[r]:self.root_q[r] + 3]

    def robot_quat(self, r: int) -> np.ndarray:
        return self.data.qpos[self.root_q[r] + 3:self.root_q[r] + 7]

    def robot_lin_vel_w(self, r: int) -> np.ndarray:
        return self.data.qvel[self.root_v[r]:self.root_v[r] + 3]

    def robot_ang_vel_b(self, r: int) -> np.ndarray:
        return self.data.qvel[self.root_v[r] + 3:self.root_v[r] + 6]

    def robot_yaw(self, r: int) -> float:
        return yaw_of(self.robot_quat(r))

    @property
    def board_pos(self) -> np.ndarray:
        return self.data.qpos[self.board_q:self.board_q + 3]

    @property
    def board_quat(self) -> np.ndarray:
        return self.data.qpos[self.board_q + 3:self.board_q + 7]

    @property
    def board_lin_vel(self) -> np.ndarray:
        return self.data.qvel[self.board_v:self.board_v + 3]

    @property
    def board_yaw(self) -> float:
        return yaw_of(self.board_quat)

    def board_corners(self) -> np.ndarray:
        q, p = self.board_quat, self.board_pos
        return np.array([rot(q, c) for c in self.corners_local]) + p

    def reset(self, pose: list | None = None, path: list | None = None) -> dict:
        cfg, d, m = self.cfg, self.data, self.model
        mujoco.mj_resetData(m, d)
        xy0 = self.start_xy if pose is None else np.array(pose[:2], dtype=np.float64)
        yaw0 = 0.0 if pose is None else float(pose[2])
        c, s = math.cos(yaw0), math.sin(yaw0)
        R = np.array([[c, -s], [s, c]])
        self.path = self.path_default.copy() if path is None else np.array(path, dtype=np.float64)

        for r, ((lx, ly), ryaw) in enumerate(self.default_robot):
            xy = xy0 + R @ np.array([lx, ly])
            yaw = ryaw + yaw0
            d.qpos[self.root_q[r]:self.root_q[r] + 3] = [xy[0], xy[1], 0.8]
            d.qpos[self.root_q[r] + 3:self.root_q[r] + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            d.qpos[self.qadr[r]] = DEFAULT_POS
        d.qpos[self.board_q:self.board_q + 3] = [xy0[0], xy0[1], BOARD_Z0]
        d.qpos[self.board_q + 3:self.board_q + 7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        for r in range(3):
            low = min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for f in self.feet[r]
                      for g in range(m.ngeom) if m.geom_bodyid[g] == f and m.geom_contype[g] == FEET)
            d.qpos[self.root_q[r] + 2] -= low + 0.001
        mujoco.mj_forward(m, d)

        self.ll_actions = np.zeros((3, 23))
        self.filtered = np.zeros((3, 23))
        self.gait_phase = self.rng.random(3) * 2.0 * math.pi
        self.tgt_pos = np.array([self.robot_pos(r)[:2] for r in range(3)])
        self.tgt_yaw = np.array([self.robot_yaw(r) for r in range(3)])
        self.tgt_start_pos, self.tgt_final_pos = self.tgt_pos.copy(), self.tgt_pos.copy()
        self.tgt_start_yaw, self.tgt_final_yaw = self.tgt_yaw.copy(), self.tgt_yaw.copy()
        self.hl_ll_count = 0
        self.ll_counter = 0
        self.episode_length = 0
        self.goal_reached = False
        self.reached = False
        self.history = {a: np.zeros((3, OBS_CORE)) for a in AGENTS}
        self._sync_path()
        return self.observations()

    def _sync_path(self) -> None:
        P = self.path.shape[0]
        seg0, seg = self.path[:-1], self.path[1:] - self.path[:-1]
        len_sq = np.maximum((seg * seg).sum(-1), 1e-12)
        b = self.board_pos[:2]
        ratio = np.clip(((b - seg0) * seg).sum(-1) / len_sq, 0.0, 1.0)
        dist_sq = ((seg0 + ratio[:, None] * seg - b) ** 2).sum(-1)
        k = int(dist_sq.argmin())
        seg_len = np.sqrt(len_sq)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        arc = cum[k] + ratio[k] * seg_len[k]
        idx = int(np.clip(np.searchsorted(cum, arc + 1e-3, side="left"), 1, P - 1))
        self.wp_reached = idx - 1
        self.waypoints = self.path[np.minimum(idx + np.arange(NUM_WAYPOINTS), P - 1)]

    def step(self, commands: dict) -> tuple[dict, bool]:
        cfg = self.cfg
        for agent, cmd in commands.items():
            r = ROBOTS.index(agent)
            dx = float(np.clip(cmd[0], -cfg.max_target_dist, cfg.max_target_dist))
            dy = float(np.clip(cmd[1], -cfg.max_target_dist, cfg.max_target_dist))
            dyaw = float(np.clip(cmd[2], -cfg.max_target_yaw, cfg.max_target_yaw))
            yaw = self.robot_yaw(r)
            c, s = math.cos(yaw), math.sin(yaw)
            self.tgt_start_pos[r] = self.tgt_pos[r]
            self.tgt_start_yaw[r] = self.tgt_yaw[r]
            self.tgt_final_pos[r] = self.robot_pos(r)[:2] + [c * dx - s * dy, s * dx + c * dy]
            self.tgt_final_yaw[r] = yaw + dyaw
        self.hl_ll_count = 0
        for _ in range(cfg.decimation):
            self._apply_action()
            mujoco.mj_step(self.model, self.data)
            self.gait_phase = (self.gait_phase + 2.0 * math.pi * cfg.gait_frequency * cfg.dt) % (2.0 * math.pi)
            self.ll_counter += 1
            if self.frame_hook is not None:
                self.frame_hook(self)
        self.episode_length += 1
        done, info = self.dones()
        self._sync_path()
        at_goal = np.linalg.norm(self.path[-1] - self.board_pos[:2]) < GOAL_RADIUS
        self.reached |= bool(at_goal)
        self.goal_reached |= bool(at_goal)
        return self.observations(), done, info

    def dones(self) -> tuple[bool, dict]:
        fallen = self.any_fallen()
        drop = self.board_pos[2] < self.cfg.board_drop_height
        timeout = self.episode_length >= self.max_episode_length
        terminated = fallen or drop or self.goal_reached
        return terminated or timeout, {"fallen": fallen, "drop": drop, "timeout": timeout}

    def any_fallen(self) -> bool:
        for r in range(3):
            g = rot_inv(self.robot_quat(r), np.array([0.0, 0.0, -1.0]))
            if self.robot_pos(r)[2] < self.cfg.termination_height or abs(g[0]) > 0.7 or abs(g[1]) > 0.7:
                return True
        return False

    def _apply_action(self) -> None:
        cfg, d = self.cfg, self.data
        if self.ll_counter % cfg.ll_decimation == 0:
            self.hl_ll_count += 1
            frac = min(self.hl_ll_count / float(cfg.ll_steps_per_hl), 1.0)
            for r in (0, 1):
                self.tgt_pos[r] = self.tgt_start_pos[r] + frac * (self.tgt_final_pos[r] - self.tgt_start_pos[r])
                sy, fy = self.tgt_start_yaw[r], self.tgt_final_yaw[r]
                self.tgt_yaw[r] = sy + frac * wrap(fy - sy)
            obs = np.stack([self._ll_obs(r) for r in range(3)]).astype(np.float32)
            with torch.no_grad():
                act = self.ll_policy(torch.from_numpy(obs)).numpy().astype(np.float64)
            act[:, LOCKED] = 0.0
            self.ll_actions = act
            self.filtered = (1.0 - cfg.filter_alpha) * self.filtered + cfg.filter_alpha * act
        q = d.qpos[self.qadr]
        dq = d.qvel[self.vadr]
        tau = KP * (self.filtered * cfg.action_scale + DEFAULT_POS - q) - KD * dq
        d.ctrl[self.act] = np.clip(tau, -EFFORT, EFFORT)

    def _ll_obs(self, r: int) -> np.ndarray:
        cfg, d = self.cfg, self.data
        quat = self.robot_quat(r)
        pos = self.robot_pos(r)
        yaw = yaw_of(quat)
        if r == 2:
            bxy, bvel, byaw = self.board_pos[:2], self.board_lin_vel[:2], self.board_yaw
            half_w = BOARD_W / 2 + cfg.robot_offset
            ox, oy = -math.sin(byaw), math.cos(byaw)
            look = max((bvel[0] * ox + bvel[1] * oy) * 0.5, 0.0)
            tx, ty = bxy[0] + (half_w + look) * ox, bxy[1] + (half_w + look) * oy
            tyaw = byaw - math.pi / 2
        else:
            tx, ty = self.tgt_pos[r]
            tyaw = self.tgt_yaw[r]
        dxw, dyw = tx - pos[0], ty - pos[1]
        c, s = math.cos(-yaw), math.sin(-yaw)
        cmd = np.array([c * dxw - s * dyw, s * dxw + c * dyw, wrap(tyaw - yaw)])
        obs = np.concatenate([
            self.ll_actions[r],
            self.robot_ang_vel_b(r) * 0.25,
            d.qpos[self.qadr[r]] - DEFAULT_POS,
            d.qvel[self.vadr[r]] * 0.05,
            [math.sin(self.gait_phase[r]), math.cos(self.gait_phase[r])],
            rot_inv(quat, np.array([0.0, 0.0, -1.0])),
            cmd, cmd,
        ])
        return np.clip(obs, -100.0, 100.0)

    def lidar(self, xy: np.ndarray, yaw: float) -> np.ndarray:
        if self.aabb.shape[0] == 0:
            return np.zeros(self.cfg.lidar_rays)
        dmax = self.cfg.lidar_range
        ang = yaw + self.ray_offsets
        dx, dy = np.cos(ang), np.sin(ang)
        eps = 1e-6
        dx = np.where(np.abs(dx) < eps, eps, dx)[:, None]
        dy = np.where(np.abs(dy) < eps, eps, dy)[:, None]
        x0, x1, y0, y1 = (self.aabb[:, i][None] for i in range(4))
        t1x, t2x = (x0 - xy[0]) / dx, (x1 - xy[0]) / dx
        t1y, t2y = (y0 - xy[1]) / dy, (y1 - xy[1]) / dy
        t_enter = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
        t_exit = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
        hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_enter <= dmax)
        dist = np.where(hit, np.maximum(t_enter, 0.0), dmax).min(-1)
        return 1.0 - np.minimum(dist, dmax) / dmax

    def observations(self) -> dict:
        bpos, bquat, bvel = self.board_pos.copy(), self.board_quat.copy(), self.board_lin_vel.copy()
        corners = self.board_corners()
        byaw = yaw_of(bquat)
        out = {}
        for agent in AGENTS:
            r = ROBOTS.index(agent)
            pos, quat = self.robot_pos(r), self.robot_quat(r)
            yaw = yaw_of(quat)
            c, s = math.cos(yaw), math.sin(yaw)

            def body(v: np.ndarray) -> np.ndarray:
                return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])

            psi_task = np.concatenate([body(w - pos[:2]) for w in self.waypoints])
            lin_b = rot_inv(quat, self.robot_lin_vel_w(r))
            psi_ego = np.concatenate([body(bpos[:2] - pos[:2]), lin_b[:2], [wrap(byaw - yaw)], [pos[2]],
                                      [rot_inv(quat, np.array([0.0, 0.0, -1.0]))[0]]])
            psi_ptn = []
            for p in ROBOTS:
                if p == agent:
                    continue
                q = ROBOTS.index(p)
                ppos, pquat = self.robot_pos(q), self.robot_quat(q)
                psi_ptn += [body(ppos[:2] - pos[:2]), body(self.robot_lin_vel_w(q)[:2]),
                            [wrap(yaw_of(pquat) - yaw)], [ppos[2]],
                            [rot_inv(pquat, np.array([0.0, 0.0, -1.0]))[0]]]
            rel = corners - pos
            corners_b = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], -1)
            com = bpos - pos
            psi_obj = np.concatenate([corners_b.reshape(-1), body(com), [com[2]], body(bvel), [bvel[2]]])
            core = np.concatenate([psi_task, psi_ego, np.concatenate(psi_ptn), psi_obj])
            lidar = self.lidar(pos[:2], yaw)
            h = np.roll(self.history[agent], 1, axis=0)
            h[0] = core
            self.history[agent] = h
            out[agent] = np.concatenate([core, lidar, h[1:].reshape(-1)]).astype(np.float32)
        return out

    def obstacle_collision(self) -> bool:
        if self.aabb.shape[0] == 0:
            return False
        a = self.aabb
        mx, my = (a[:, 0] + a[:, 1]) / 2, (a[:, 2] + a[:, 3]) / 2
        hx, hy = (a[:, 1] - a[:, 0]) / 2, (a[:, 3] - a[:, 2]) / 2
        cxy, yaw = self.board_pos[:2], self.board_yaw
        co, si = math.cos(yaw), math.sin(yaw)
        aco, asi = abs(co), abs(si)
        eu, ev = BOARD_L / 2, BOARD_W / 2
        dx, dy = cxy[0] - mx, cxy[1] - my
        hit = ((np.abs(dx) <= hx + eu * aco + ev * asi) & (np.abs(dy) <= hy + eu * asi + ev * aco)
               & (np.abs(dx * co + dy * si) <= eu + hx * aco + hy * asi)
               & (np.abs(-dx * si + dy * co) <= ev + hx * asi + hy * aco))
        if hit.any():
            return True
        rad = self.cfg.robot_collision_radius
        for r in range(3):
            p = self.robot_pos(r)[:2]
            cx, cy = np.clip(p[0], a[:, 0], a[:, 1]), np.clip(p[1], a[:, 2], a[:, 3])
            if (((p[0] - cx) ** 2 + (p[1] - cy) ** 2) <= rad * rad).any():
                return True
        return False

    def board_tilt(self) -> float:
        z = self.board_corners()[:, 2]
        return float(np.abs(z - z.mean()).sum())

    def board_roll_pitch_deg(self) -> tuple[float, float]:
        g = rot_inv(self.board_quat, np.array([0.0, 0.0, -1.0]))
        return (math.degrees(math.asin(np.clip(g[1], -1, 1))), math.degrees(math.asin(np.clip(g[0], -1, 1))))

    def carrier_2_anchor(self) -> np.ndarray:
        byaw = self.board_yaw
        reach = BOARD_W / 2 + self.cfg.robot_offset
        b = self.board_pos[:2]
        return np.array([b[0] - reach * math.sin(byaw), b[1] + reach * math.cos(byaw)])

    def carrier_2_geometry(self) -> tuple[float, float, float]:
        f = self.robot_pos(2)[:2]
        a = self.aabb
        if a.shape[0]:
            dx = f[0] - np.clip(f[0], a[:, 0], a[:, 1])
            dy = f[1] - np.clip(f[1], a[:, 2], a[:, 3])
            clear = float(np.sqrt(dx * dx + dy * dy).min() - self.cfg.robot_collision_radius)
        else:
            clear = 5.0
        partner = min(float(np.linalg.norm(self.robot_pos(r)[:2] - f)) for r in (0, 1))
        place = float(np.linalg.norm(f - self.carrier_2_anchor()))
        return clear, partner, place

    def safety_state(self, SafetyState: type):
        def t(x) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x), dtype=torch.float32)

        return SafetyState(
            board_xy=t(self.board_pos[:2])[None], board_yaw=t([self.board_yaw]),
            robot_xy=t(np.stack([self.robot_pos(r)[:2] for r in range(3)]))[None],
            robot_yaw=t([[self.robot_yaw(r) for r in range(3)]]),
            controlled=torch.tensor([0, 1], dtype=torch.long), anchors_local=t(self.anchors_local),
            aabbs=t(self.aabb), action_scale=t([self.cfg.max_target_dist, self.cfg.max_target_dist,
                                                 self.cfg.max_target_yaw]))
