from __future__ import annotations

import json
import math
import os
import secrets
import subprocess
import tempfile
from multiprocessing.connection import Listener
from pathlib import Path
import mujoco
import numpy as np
import torch
from team_env import (AGENTS, BOARD_L, BOARD_T, BOARD_W, BOARD_Z0, DEFAULT_POS, EFFORT, FEET, JOINTS, KD, KP, LOCKED,
                      LOW_LEVEL_POLICY, OBS_CORE, TaskCfg, TeamCarryEnv, build_model, rot, rot_inv, wrap, yaw_of)

HERE = Path(__file__).resolve().parent
HUMAN_OFFSET = 0.12
HUMAN_RADIUS = 0.25
ROOT_DROP = 0.175
EXCHANGE = 2
K_XY, D_XY, F_Z_MAX = 300.0, 60.0, 40.0
SUP_W, SUP_Z, SUP_K = 3.5, 0.945, 3000.0
SUP_D = 2.0 * math.sqrt(SUP_K * 3.0)
MU = 1.5
EDGE_BAND = 0.25
LOOKAHEAD_S = 0.5
Q_FWD = np.array([math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qz(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def human_heading(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(-(1 - 2 * (x * x + z * z)), -(2 * (x * y - w * z)))


class HumanTeamEnv(TeamCarryEnv):
    def __init__(self, scene: dict, checkpoint: str, cfg: TaskCfg = TaskCfg, ll_policy: str | Path = LOW_LEVEL_POLICY,
                 seed: int = 0) -> None:
        self.cfg = cfg
        self.obstacles = [(tuple(s), tuple(p)) for s, p in scene["obstacle_specs"]]
        self.model = build_model(self.obstacles, cfg, n_robots=2)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.ll_policy = torch.jit.load(str(ll_policy), map_location="cpu").eval()
        m = self.model
        self.qadr = np.array([[m.jnt_qposadr[m.joint(f"r{r}_{n}").id] for n in JOINTS] for r in range(2)])
        self.vadr = np.array([[m.jnt_dofadr[m.joint(f"r{r}_{n}").id] for n in JOINTS] for r in range(2)])
        self.act = np.array([[m.actuator(f"r{r}_{n}").id for n in JOINTS] for r in range(2)])
        self.root_q = np.array([m.jnt_qposadr[m.joint(f"r{r}_root").id] for r in range(2)])
        self.root_v = np.array([m.jnt_dofadr[m.joint(f"r{r}_root").id] for r in range(2)])
        self.board_q = m.jnt_qposadr[m.joint("board_root").id]
        self.board_v = m.jnt_dofadr[m.joint("board_root").id]
        self.board_body = m.body("board").id
        self.feet = [[m.body(f"r{r}_{s}_ankle_roll_link").id for s in ("left", "right")] for r in range(2)]
        aabb = [[p[0] - s[0] / 2, p[0] + s[0] / 2, p[1] - s[1] / 2, p[1] + s[1] / 2] for s, p in self.obstacles]
        self.aabb = np.array(aabb, dtype=np.float64).reshape(-1, 4)
        self.ray_offsets = np.linspace(0.0, 2.0 * math.pi, cfg.lidar_rays + 1)[:-1]
        self.path_default = np.array(scene["path_waypoints"], dtype=np.float64)
        self.start_xy = np.array(scene.get("start_xy", scene["path_waypoints"][0]), dtype=np.float64)
        self.max_episode_length = math.ceil(cfg.episode_length_s / (cfg.dt * cfg.decimation))
        half_l, half_w, off = BOARD_L / 2, BOARD_W / 2, cfg.robot_offset
        self.default_robot = [((half_l + off, 0.0), math.pi), ((-(half_l + off), 0.0), 0.0),
                              ((0.0, half_w + HUMAN_OFFSET), -math.pi / 2)]
        self.anchors_local = np.array([[half_l + off, 0.0], [-(half_l + off), 0.0], [0.0, half_w + HUMAN_OFFSET]])
        self.corners_local = np.array([[sx * half_l, sy * half_w, BOARD_T / 2] for sx in (1, -1) for sy in (1, -1)])
        self.frame_hook = None
        self.grip = json.loads((HERE / "data" / "grip_pose.json").read_text())
        self.h_model = None
        self._start(checkpoint)

    def _start(self, checkpoint: str) -> None:
        key = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=key)
        self.log = tempfile.NamedTemporaryFile("w", prefix="muscle_human_", suffix=".log", delete=False)
        env = dict(os.environ, MUSCLE_HUMAN_AUTHKEY=key.hex(), MUSCLE_HUMAN_PORT=str(self.listener.address[1]))
        self.proc = subprocess.Popen([os.environ["MUSCLE_HUMAN_PYTHON"], str(HERE / "human.py"), checkpoint,
                                      "--seconds", str(self.cfg.episode_length_s + 3.0)],
                                     cwd=HERE, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.conn = self.listener.accept()

    def _recv(self, timeout: float = 900.0) -> dict:
        if not self.conn.poll(timeout):
            raise RuntimeError(f"muscle human silent for {timeout} s (log {self.log.name})")
        return self.conn.recv()

    def _ingest(self, st: dict) -> None:
        q = np.asarray(st["qpos"], dtype=np.float64)
        self.h_qpos = q
        xy = self.Rt @ q[:2] + self.t
        self.h_pos = np.array([xy[0], xy[1], q[2] - ROOT_DROP])
        self.h_quat = qmul(qz(self.th), qmul(q[3:7], Q_FWD))
        v = np.asarray(st["vel"], dtype=np.float64)
        self.h_vel = np.array([*(self.Rt @ v[:2]), v[2]])
        hands, hv = np.asarray(st["hands"], dtype=np.float64), np.asarray(st["hands_v"], dtype=np.float64)
        self.hand_w = np.concatenate([(self.Rt @ hands[:, :2].T).T + self.t, hands[:, 2:]], 1)
        self.hand_v = np.concatenate([(self.Rt @ hv[:, :2].T).T, hv[:, 2:]], 1)

    def robot_pos(self, r: int) -> np.ndarray:
        return self.h_pos if r == 2 else super().robot_pos(r)

    def robot_quat(self, r: int) -> np.ndarray:
        return self.h_quat if r == 2 else super().robot_quat(r)

    def robot_lin_vel_w(self, r: int) -> np.ndarray:
        return self.h_vel if r == 2 else super().robot_lin_vel_w(r)

    def robot_ang_vel_b(self, r: int) -> np.ndarray:
        return np.zeros(3) if r == 2 else super().robot_ang_vel_b(r)

    def reset(self, pose: list | None = None, path: list | None = None) -> dict:
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        xy0 = self.start_xy if pose is None else np.array(pose[:2], dtype=np.float64)
        yaw0 = 0.0 if pose is None else float(pose[2])
        c, s = math.cos(yaw0), math.sin(yaw0)
        R = np.array([[c, -s], [s, c]])
        self.path = self.path_default.copy() if path is None else np.array(path, dtype=np.float64)
        for r in range(2):
            (lx, ly), ryaw = self.default_robot[r]
            xy = xy0 + R @ np.array([lx, ly])
            yaw = ryaw + yaw0
            d.qpos[self.root_q[r]:self.root_q[r] + 3] = [xy[0], xy[1], 0.8]
            d.qpos[self.root_q[r] + 3:self.root_q[r] + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            d.qpos[self.qadr[r]] = DEFAULT_POS
        d.qpos[self.board_q:self.board_q + 3] = [xy0[0], xy0[1], BOARD_Z0]
        d.qpos[self.board_q + 3:self.board_q + 7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        for r in range(2):
            low = min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for f in self.feet[r]
                      for g in range(m.ngeom) if m.geom_bodyid[g] == f and m.geom_contype[g] == FEET)
            d.qpos[self.root_q[r] + 2] -= low + 0.001
        mujoco.mj_forward(m, d)

        (lx, ly), ryaw = self.default_robot[2]
        spawn_xy, spawn_yaw = xy0 + R @ np.array([lx, ly]), ryaw + yaw0
        hello = self._recv()
        q0 = np.asarray(hello["qpos"])
        self.th = spawn_yaw - human_heading(q0[3:7])
        self.Rt = np.array([[math.cos(self.th), -math.sin(self.th)], [math.sin(self.th), math.cos(self.th)]])
        self.t = spawn_xy - self.Rt @ q0[:2]
        self._ingest(hello)
        self.h_model_path = hello["model"]
        self.h_fell = False
        bq, bp = self.board_quat.copy(), self.board_pos.copy()
        hands_b = np.array([rot_inv(bq, h - bp) for h in self.hand_w])
        self.grip_edge = hands_b[:, 1].copy()
        self.grip_dx = hands_b[:, 0] - rot_inv(bq, np.r_[self.h_pos[:2], 0.0] - np.r_[bp[:2], 0.0])[0]
        self.board_prev = None

        self.ll_actions = np.zeros((2, 23))
        self.filtered = np.zeros((2, 23))
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

    def _exchange(self) -> None:
        d = self.data
        bp, bq, bv = self.board_pos.copy(), self.board_quat.copy(), self.board_lin_vel.copy()
        w = rot(bq, d.qvel[self.board_v + 3:self.board_v + 6])
        com = d.xipos[self.board_body]
        lim = BOARD_L / 2 - 0.05
        px = rot_inv(bq, np.r_[self.h_pos[:2] - bp[:2], 0.0])[0]
        f_b, tau_b, f_h = np.zeros(3), np.zeros(3), np.zeros((2, 3))
        for i, side in enumerate("rl"):
            gs = self.grip[side]
            gl = np.array(gs["third_mc_pos"], float)
            gl[0] = float(np.clip(px + self.grip_dx[i] + gs["third_mc_pos"][0] - gs["lunate_pos"][0], -lim, lim))
            g = bp + rot(bq, gl)
            vg = bv + np.cross(w, g - bp)
            e, ev = g - self.hand_w[i], vg - self.hand_v[i]
            hl = rot_inv(bq, self.hand_w[i] - bp)
            fz = float(np.clip(SUP_W + SUP_K * (SUP_Z + gl[2] - g[2]) - SUP_D * vg[2], 0.0, F_Z_MAX))
            fxy = K_XY * e[:2] + D_XY * ev[:2]
            fn, cap = float(np.linalg.norm(fxy)), MU * fz
            if fn > cap:
                k = cap / max(fn, 1e-9)
                fxy *= k
                gl[0] = float(np.clip(hl[0] + (gl[0] - hl[0]) * k, -lim, lim))
                gl[1] = float(np.clip(hl[1] + (gl[1] - hl[1]) * k, self.grip_edge[i] - EDGE_BAND, self.grip_edge[i]))
                g = bp + rot(bq, gl)
            fb = np.array([-fxy[0], -fxy[1], fz])
            f_b += fb
            tau_b += np.cross(g - com, fb)
            f_h[i] = -fb
        d.xfrc_applied[self.board_body] = np.concatenate([f_b, tau_b])

        cur = (bp[:2].copy(), yaw_of(bq), bv[:2].copy(), float(w[2]))
        p, byaw, v, wz = self.board_prev or cur
        self.board_prev = cur
        out = np.array([-math.sin(byaw), math.cos(byaw)])
        r = (BOARD_W / 2 + HUMAN_OFFSET + max(float(v @ out) * LOOKAHEAD_S, 0.0)) * out
        tgt, vt = p + r, v + wz * np.array([-r[1], r[0]])
        RtT = self.Rt.T
        Rb = np.stack([rot(bq, e) for e in np.eye(3)], 1)
        Rh = np.array([[math.cos(-self.th), -math.sin(-self.th), 0.0], [math.sin(-self.th), math.cos(-self.th), 0.0],
                       [0.0, 0.0, 1.0]])
        pos, Rs = [], []
        for k, side in enumerate("rl"):
            gp = np.array(self.grip[side]["lunate_pos"], float)
            gp[0] = float(np.clip(px + self.grip_dx[k], -lim, lim))
            gw = bp + Rb @ gp
            pos.append(np.r_[RtT @ (gw[:2] - self.t), gw[2]])
            Rs.append(Rh @ Rb @ np.array(self.grip[side]["lunate_R"], float))
        msg = {"F": np.concatenate([(RtT @ f_h[:, :2].T).T, f_h[:, 2:]], 1), "tgt": RtT @ (tgt - self.t),
               "v": RtT @ vt, "h": byaw - math.pi / 2 - self.th, "w": wz,
               "grip": {"pos": np.array(pos), "R": np.array(Rs)},
               "board": np.r_[RtT @ (bp[:2] - self.t), bp[2], qmul(qz(-self.th), bq)]}
        try:
            self.conn.send(msg)
            st = self._recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            st = {"fell": True}
        if st.get("fell"):
            self.h_fell = True
        else:
            self._ingest(st)

    def _apply_action(self) -> None:
        cfg, d = self.cfg, self.data
        if self.ll_counter % EXCHANGE == 0 and not self.h_fell:
            self._exchange()
        if self.ll_counter % cfg.ll_decimation == 0:
            self.hl_ll_count += 1
            frac = min(self.hl_ll_count / float(cfg.ll_steps_per_hl), 1.0)
            for r in (0, 1):
                self.tgt_pos[r] = self.tgt_start_pos[r] + frac * (self.tgt_final_pos[r] - self.tgt_start_pos[r])
                sy, fy = self.tgt_start_yaw[r], self.tgt_final_yaw[r]
                self.tgt_yaw[r] = sy + frac * wrap(fy - sy)
            obs = np.stack([self._ll_obs(r) for r in range(2)]).astype(np.float32)
            with torch.no_grad():
                act = self.ll_policy(torch.from_numpy(obs)).numpy().astype(np.float64)
            act[:, LOCKED] = 0.0
            self.ll_actions = act
            self.filtered = (1.0 - cfg.filter_alpha) * self.filtered + cfg.filter_alpha * act
        q = d.qpos[self.qadr]
        dq = d.qvel[self.vadr]
        tau = KP * (self.filtered * cfg.action_scale + DEFAULT_POS - q) - KD * dq
        d.ctrl[self.act] = np.clip(tau, -EFFORT, EFFORT)

    def any_fallen(self) -> bool:
        for r in range(2):
            g = rot_inv(self.robot_quat(r), np.array([0.0, 0.0, -1.0]))
            if self.robot_pos(r)[2] < self.cfg.termination_height or abs(g[0]) > 0.7 or abs(g[1]) > 0.7:
                return True
        return self.h_fell

    def carrier_2_anchor(self) -> np.ndarray:
        byaw = self.board_yaw
        reach = BOARD_W / 2 + HUMAN_OFFSET
        b = self.board_pos[:2]
        return np.array([b[0] - reach * math.sin(byaw), b[1] + reach * math.cos(byaw)])

    def carrier_2_geometry(self) -> tuple[float, float, float]:
        f = self.robot_pos(2)[:2]
        a = self.aabb
        if a.shape[0]:
            dx = f[0] - np.clip(f[0], a[:, 0], a[:, 1])
            dy = f[1] - np.clip(f[1], a[:, 2], a[:, 3])
            clear = float(np.sqrt(dx * dx + dy * dy).min() - HUMAN_RADIUS)
        else:
            clear = 5.0
        partner = min(float(np.linalg.norm(self.robot_pos(r)[:2] - f)) for r in (0, 1))
        return clear, partner, float(np.linalg.norm(f - self.carrier_2_anchor()))

    def render_overlay(self, renderer: mujoco.Renderer, cam: mujoco.MjvCamera, opt: mujoco.MjvOption,
                       rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        if self.h_model is None:
            self.h_model = mujoco.MjModel.from_xml_path(self.h_model_path)
            self.h_model.vis.global_.fovy = self.model.vis.global_.fovy
            self.h_model.vis.global_.offwidth = max(w, self.h_model.vis.global_.offwidth)
            self.h_model.vis.global_.offheight = max(h, self.h_model.vis.global_.offheight)
            self.h_data = mujoco.MjData(self.h_model)
            self.h_renderer = mujoco.Renderer(self.h_model, height=h, width=w)
            self.h_opt = mujoco.MjvOption()
            self.h_opt.geomgroup[:] = [1, 0, 0, 0, 0, 0]
            self.h_opt.sitegroup[:] = 0
        q = self.h_qpos.copy()
        q[:2] = self.Rt @ q[:2] + self.t
        q[3:7] = qmul(qz(self.th), q[3:7])
        self.h_data.qpos[:] = q
        mujoco.mj_fwdPosition(self.h_model, self.h_data)
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, camera=cam, scene_option=opt)
        depth = renderer.render()
        renderer.disable_depth_rendering()
        hr = self.h_renderer
        hr.update_scene(self.h_data, camera=cam, scene_option=self.h_opt)
        h_rgb = hr.render()
        hr.enable_depth_rendering()
        hr.update_scene(self.h_data, camera=cam, scene_option=self.h_opt)
        h_depth = hr.render()
        hr.disable_depth_rendering()
        front = (h_depth < depth - 1e-3) & (h_depth < 0.999 * self.h_model.stat.extent * self.h_model.vis.map.zfar)
        out = rgb.copy()
        out[front] = h_rgb[front]
        return out

    def close(self) -> None:
        try:
            if self.proc.poll() is None and not self.h_fell:
                self.conn.send({"done": True})
            self.proc.wait(timeout=120)
        except Exception:
            self.proc.kill()
        finally:
            self.conn.close()
            self.listener.close()
            self.log.close()
