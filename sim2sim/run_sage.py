#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
H_CAP = 5.0
TASK_TOGGLES = ("projection_progress", "partner_noise", "partner_noise_lin",
                "partner_noise_yaw_deg", "partner_gain_lin", "partner_gain_yaw")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="a train.sh checkpoint, e.g. results/<run>/models/iter_200.pt")
    p.add_argument("--scene_json", default=None, help="scene JSON (default: the run's scene from run_args.json)")
    p.add_argument("--num_episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shield", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--fixed_spawn", action="store_true")
    p.add_argument("--episode_length_s", type=float, default=None)
    p.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) // 3)))
    p.add_argument("--out", default=None)
    p.add_argument("--video", default=None, help="mp4 of episode 0")
    p.add_argument("--violation_tol", type=float, default=1e-4)
    p.add_argument("--muscle_human", default=None, help="muscle-driven human checkpoint as carrier robot_2")
    return p.parse_args()


def run_args(checkpoint: str) -> dict:
    for path in (Path(checkpoint).resolve().parent / "run_args.json",
                 Path(checkpoint).resolve().parent.parent / "run_args.json"):
        if path.is_file():
            return json.loads(path.read_text())
    return {}


def run_args_toggles(run: dict) -> dict:
    return {k: run.get(k) for k in TASK_TOGGLES if run.get(k) not in (None, False)}


def scene_path(args, run: dict) -> Path:
    name = args.scene_json or run.get("scene_json")
    if not name:
        raise SystemExit("no --scene_json given and no scene_json in the run's run_args.json")
    path = Path(name)
    return path if path.is_absolute() or path.is_file() else REPO / path


def _setup_paths() -> None:
    for path in (str(HERE), str(REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_policy(checkpoint: str) -> tuple:
    import torch
    from isaac_sage.safety import FormationSafetyModel, SafetyConfig
    from isaac_sage.trainer import RunningMeanStd
    from isaac_sage.tree_policy import TeamActor

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    n, od, ad = int(ck["n_agents"]), int(ck["obs_dim_per"]), int(ck["act_dim_per"])
    if (n, od, ad) != (2, 183, 3):
        raise ValueError(f"checkpoint dims {(n, od, ad)} != MuJoCo env (2, 183, 3)")
    cfg = ck["cfg"]
    actor = TeamActor(od, ad, n, int(cfg["tree_depth"]), float(cfg.get("log_std_init", -0.5)))
    actor.load_state_dict(ck["actor"])
    actor.eval()
    obs_norm = RunningMeanStd(od * n)
    obs_norm.load_state_dict(ck["obs_norm"])
    known = {f.name for f in fields(SafetyConfig)}
    safety = FormationSafetyModel(SafetyConfig(**{k: v for k, v in dict(cfg.get("safety", {})).items() if k in known}))
    return actor, obs_norm, safety, cfg


class Video:
    def __init__(self, env, path: str, fps: int = 25, eye: tuple = (-2.6, 3.4, 3.2), lookat: tuple = (0.0, 0.0, 0.6),
                 size: tuple = (960, 540)) -> None:
        import imageio
        import mujoco

        self.every = max(int(round(1.0 / (fps * env.cfg.dt))), 1)
        env.model.vis.global_.offwidth = max(size[0], env.model.vis.global_.offwidth)
        env.model.vis.global_.offheight = max(size[1], env.model.vis.global_.offheight)
        self.renderer = mujoco.Renderer(env.model, height=size[1], width=size[0])
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [0, 1, 1, 0, 0, 0]
        self.cam = mujoco.MjvCamera()
        off = np.array(eye) - np.array(lookat)
        self.cam.distance = float(np.linalg.norm(off))
        self.cam.azimuth = math.degrees(math.atan2(-off[1], -off[0]))
        self.cam.elevation = -math.degrees(math.asin(off[2] / self.cam.distance))
        self.lookat = np.array(lookat)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(path, fps=fps, quality=7, macro_block_size=8)

    def __call__(self, env) -> None:
        if env.ll_counter % self.every:
            return
        self.cam.lookat[:] = env.board_pos + self.lookat
        self.renderer.update_scene(env.data, camera=self.cam, scene_option=self.opt)
        rgb = self.renderer.render()
        if hasattr(env, "render_overlay"):
            rgb = env.render_overlay(self.renderer, self.cam, self.opt, rgb)
        self.writer.append_data(rgb)

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


def run_episode(job: dict) -> dict:
    _setup_paths()
    import torch

    torch.set_num_threads(1)
    from isaac_sage.safety import FAMILY_NAMES, SafetyState
    from isaac_sage.trainer import deploy_action
    from team_env import TaskCfg, TeamCarryEnv

    a = job["args"]
    scene = json.loads(Path(job["scene_json"]).read_text())
    cfg = type("Cfg", (TaskCfg,), {})
    cfg.episode_length_s = float(a["episode_length_s"] or scene.get("episode_length_s", 300.0))
    if a["muscle_human"]:
        from muscle_human.team import HUMAN_RADIUS, HumanTeamEnv

        env = HumanTeamEnv(scene, str(Path(a["muscle_human"]).resolve()), cfg, seed=job["seed"])
    else:
        env = TeamCarryEnv(scene, cfg, seed=job["seed"])
    actor, obs_norm, safety, ck_cfg = load_policy(a["checkpoint"])
    if a["muscle_human"]:
        from dataclasses import replace

        from isaac_sage.safety import FormationSafetyModel

        r = safety.cfg.robot_radius
        safety = FormationSafetyModel(replace(safety.cfg, robot_radii=(r, r, HUMAN_RADIUS)))
    variant = str(ck_cfg.get("variant", "?"))
    shield = (variant != "happo_tree") if a["shield"] == "auto" else (a["shield"] == "on")
    toggles = job["toggles"]
    rng = np.random.default_rng(job["seed"] + 7919)
    scale = np.array([cfg.max_target_dist, cfg.max_target_dist, cfg.max_target_yaw])

    obs = env.reset(job["pose"], job["path"])
    video = None
    if job.get("video"):
        video = Video(env, job["video"])
        env.frame_hook = video
    num_waypoints = max(env.path.shape[0] - 1, 1)
    leaf_counts = np.zeros((2, actor.actors[0].tree.num_leaves))
    st = {k: 0.0 for k in ("n_safe", "pviol", "interv", "corr", "nomcorr", "infeas", "active", "n_hl",
                           "col_steps", "tilt_sum", "tilt_max", "roll_max", "pitch_max", "place_sum", "place_max")}
    for f in FAMILY_NAMES:
        st[f"pviol_{f}"] = 0.0
    st.update(z_min=np.inf, min_h=H_CAP, f_clear=H_CAP, f_partner=H_CAP, dropped=False, fallen=False)
    t0 = time.time()
    while True:
        raw = torch.from_numpy(np.concatenate([obs["robot_0"], obs["robot_1"]]))[None]
        state = env.safety_state(SafetyState)
        with torch.no_grad():
            shielded, info = deploy_action(actor, safety, obs_norm, raw, state)
            nominal = info["a_nom"]
            actions = shielded if shield else nominal
            rows = info["rows"]
            violated = rows.margins(nominal) < -a["violation_tol"]
            nomcorr = float((shielded - nominal).norm(dim=-1))
            corr = float((actions - nominal).norm(dim=-1))
            min_b = float(rows.barrier.masked_fill(rows.mask <= 0, H_CAP).min(dim=1).values.clamp(max=H_CAP))
        for i in range(2):
            leaf_counts[i, int(info["leaves"][0, i])] += 1
        st["n_safe"] += 1
        st["pviol"] += float(violated.any(1))
        for fid, f in enumerate(FAMILY_NAMES):
            st[f"pviol_{f}"] += float((violated & (rows.family == fid)).any(1))
        st["interv"] += float(corr > 1e-4)
        st["corr"] += corr
        st["nomcorr"] += nomcorr
        st["infeas"] += float((~info["feasible"]).any())
        st["active"] += float(info["active"].float().sum(1))
        st["min_h"] = min(st["min_h"], min_b)

        act = actions[0].numpy().astype(np.float64)
        cmds = {}
        for i, agent in enumerate(("robot_0", "robot_1")):
            c = np.clip(act[3 * i:3 * i + 3], -1.0, 1.0) * scale
            if toggles.get("partner_noise") and i == 1:
                lin = float(toggles.get("partner_noise_lin", 0.01))
                yaw_sd = math.radians(float(toggles.get("partner_noise_yaw_deg", 1.5)))
                xy = c[:2] * float(toggles.get("partner_gain_lin", 1.0)) + (rng.random(2) * 2.0 - 1.0) * lin
                ang = rng.standard_normal() * yaw_sd
                ca, sa = math.cos(ang), math.sin(ang)
                c = np.array([ca * xy[0] - sa * xy[1], sa * xy[0] + ca * xy[1],
                              c[2] * float(toggles.get("partner_gain_yaw", 1.0))])
            cmds[agent] = c
        obs, done, flags = env.step(cmds)

        tilt = env.board_tilt()
        roll, pitch = env.board_roll_pitch_deg()
        st["tilt_sum"] += tilt
        st["tilt_max"] = max(st["tilt_max"], tilt)
        st["roll_max"] = max(st["roll_max"], abs(roll))
        st["pitch_max"] = max(st["pitch_max"], abs(pitch))
        st["z_min"] = min(st["z_min"], float(env.board_pos[2]))
        st["n_hl"] += 1
        st["dropped"] |= bool(env.board_pos[2] < cfg.board_drop_height)
        st["fallen"] |= env.any_fallen()
        st["col_steps"] += float(env.obstacle_collision())
        clear, partner, place = env.carrier_2_geometry()
        st["f_clear"] = min(st["f_clear"], clear)
        st["f_partner"] = min(st["f_partner"], partner)
        st["place_sum"] += place
        st["place_max"] = max(st["place_max"], place)
        if done:
            break
    if video is not None:
        video.close()
    if hasattr(env, "close"):
        env.close()
    n_hl, n_safe = max(st["n_hl"], 1.0), max(st["n_safe"], 1.0)
    row = {
        "episode": job["episode"], "pool_index": job["pool_index"], "wp": int(env.wp_reached),
        "success": bool(env.reached), "steps": int(env.episode_length),
        "timeout": env.episode_length >= env.max_episode_length,
        "collision_hl_steps": int(st["col_steps"]), "drop": bool(st["dropped"]), "fallen": bool(st["fallen"]),
        "tilt_mean_m": st["tilt_sum"] / n_hl, "tilt_max_m": st["tilt_max"],
        "roll_abs_max_deg": st["roll_max"], "pitch_abs_max_deg": st["pitch_max"], "board_z_min": st["z_min"],
        "pviol_nominal": st["pviol"] / n_safe, "intervention_rate": st["interv"] / n_safe,
        "correction": st["corr"] / n_safe, "nominal_correction": st["nomcorr"] / n_safe,
        "qp_infeasible": st["infeas"] / n_safe, "rows_active": st["active"] / n_safe, "min_h": st["min_h"],
        "carrier_2_wall_clearance_min_m": st["f_clear"], "carrier_2_partner_dist_min_m": st["f_partner"],
        "carrier_2_place_err_mean_m": st["place_sum"] / n_hl, "carrier_2_place_err_max_m": st["place_max"],
        "wall_s": time.time() - t0, "leaf_counts": leaf_counts.tolist(), "variant": variant, "shield": shield,
    }
    for f in FAMILY_NAMES:
        row[f"pviol_{f}"] = st[f"pviol_{f}"] / n_safe
    row["completion"] = 1.0 if row["success"] else min(row["wp"] / num_waypoints, 1.0)
    row["reason"] = ("success" if row["success"] else "drop" if row["drop"] else "fallen" if row["fallen"]
                     else "timeout" if row["timeout"] else "other")
    print(f"[ep {job['episode']:2d}] pool {job['pool_index']:3d} {row['reason']:8s} steps {row['steps']:3d} "
          f"wp {row['wp']:2d}/{num_waypoints} coll {row['collision_hl_steps']} tilt {row['tilt_mean_m']:.3f} "
          f"interv {row['intervention_rate']:.2f} ({row['wall_s']:.0f} s)", flush=True)
    return row


def main() -> None:
    args = parse_args()
    _setup_paths()
    from isaac_sage.safety import FAMILY_NAMES

    run = run_args(args.checkpoint)
    scene_json = scene_path(args, run)
    scene = json.loads(scene_json.read_text())
    toggles = run_args_toggles(run)
    if not toggles.get("projection_progress"):
        raise NotImplementedError(f"only runs with projection_progress are supported (got {toggles})")
    pool = None if args.fixed_spawn else scene.get("spawn_pool")
    rng = np.random.default_rng(args.seed)
    jobs = []
    for ep in range(args.num_episodes):
        k = int(rng.integers(len(pool))) if pool else -1
        jobs.append({"episode": ep, "pool_index": k, "seed": args.seed * 1000 + ep,
                     "pose": pool[k]["pose"] if pool else None, "path": pool[k]["path"] if pool else None,
                     "toggles": toggles, "args": vars(args), "scene_json": str(scene_json),
                     "video": args.video if ep == 0 else None})
    print(f"[mujoco-eval] checkpoint {args.checkpoint}\n[mujoco-eval] scene {scene.get('name')} episodes "
          f"{args.num_episodes} spawn {'randomised' if pool else 'fixed'} toggles {toggles} workers {args.workers}",
          flush=True)
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as workers:
            rows = list(workers.map(run_episode, jobs))
    else:
        rows = [run_episode(job) for job in jobs]
    rows.sort(key=lambda r: r["episode"])

    def arr(k: str) -> np.ndarray:
        return np.array([float(r[k]) for r in rows])

    def ms(k: str) -> dict:
        v = arr(k)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    leaf = np.sum([np.array(r.pop("leaf_counts")) for r in rows], 0)
    occ = leaf / np.maximum(leaf.sum(1, keepdims=True), 1.0)
    summary = {
        "simulator": "mujoco", "checkpoint": args.checkpoint, "variant": rows[0]["variant"],
        "shield": rows[0]["shield"], "scene": scene.get("name"), "episodes": len(rows),
        "n_path_wp": len(scene["path_waypoints"]) - 1, "seed": args.seed, "env_toggles": toggles,
        "spawn": "randomised" if pool else "fixed", "success_rate": float(arr("success").mean()),
        "completion": ms("completion"),
        "collision_episode_rate": float((arr("collision_hl_steps") > 0).mean()),
        "collision_hl_steps": ms("collision_hl_steps"),
        "drop_episode_rate": float(arr("drop").mean()), "fallen_episode_rate": float(arr("fallen").mean()),
        "timeout_episode_rate": float(arr("timeout").mean()),
        "fail_split": {k: sum(1 for r in rows if r["reason"] == k)
                       for k in ("success", "drop", "fallen", "timeout", "other")},
        "carrier_2_wall_clearance_min_m": ms("carrier_2_wall_clearance_min_m"),
        "carrier_2_wall_contact_episode_rate": float((arr("carrier_2_wall_clearance_min_m") <= 0.0).mean()),
        "carrier_2_partner_dist_min_m": ms("carrier_2_partner_dist_min_m"),
        "carrier_2_place_err_mean_m": ms("carrier_2_place_err_mean_m"),
        "carrier_2_place_err_max_m": ms("carrier_2_place_err_max_m"),
        "tilt_mean_m": ms("tilt_mean_m"), "tilt_max_m": ms("tilt_max_m"), "steps": ms("steps"),
        "wall_time_s": time.time() - t0,
    }
    for k in ("pviol_nominal", "intervention_rate", "correction", "nominal_correction", "qp_infeasible",
              "rows_active", "min_h") + tuple(f"pviol_{f}" for f in FAMILY_NAMES):
        summary[k] = ms(k)
    summary["leaf_occupancy"] = {f"carrier_{i}": [round(float(v), 4) for v in occ[i]] for i in range(2)}
    summary["leaves_used"] = {f"carrier_{i}": int((occ[i] > 0).sum()) for i in range(2)}

    s = summary
    print("\n===== SAGE deployment evaluation (MuJoCo) =====")
    print(f"checkpoint: {args.checkpoint}")
    print(f"variant: {s['variant']} | shield: {'on' if s['shield'] else 'off'} | scene: {s['scene']} | "
          f"episodes: {s['episodes']} | seed: {args.seed}")
    print(f"success rate: {s['success_rate']:.3f}   completion: {s['completion']['mean']:.3f}"
          f" ± {s['completion']['std']:.3f}")
    print(f"collision ep rate: {s['collision_episode_rate']:.3f}   drop: {s['drop_episode_rate']:.3f}"
          f"   fallen: {s['fallen_episode_rate']:.3f}   timeout: {s['timeout_episode_rate']:.3f}")
    print(f"fail split: {s['fail_split']}")
    print(f"PViol(a_nom): {s['pviol_nominal']['mean']:.3f}   "
          + "  ".join(f"{f}={s[f'pviol_{f}']['mean']:.3f}" for f in FAMILY_NAMES))
    print(f"||Q(a_nom)-a_nom||: {s['nominal_correction']['mean']:.4f}   ||a_exec-a_nom||: "
          f"{s['correction']['mean']:.4f}   intervention rate: {s['intervention_rate']['mean']:.3f}")
    print(f"QP infeasible: {s['qp_infeasible']['mean']:.3f}   active rows: {s['rows_active']['mean']:.2f}"
          f"   min h: {s['min_h']['mean']:.3f}")
    print(f"tilt mean/max (m): {s['tilt_mean_m']['mean']:.3f} / {s['tilt_max_m']['mean']:.3f}")
    print(f"carrier_2: wall clearance min {s['carrier_2_wall_clearance_min_m']['mean']:.3f} m (contact ep rate "
          f"{s['carrier_2_wall_contact_episode_rate']:.3f}), partner dist min {s['carrier_2_partner_dist_min_m']['mean']:.3f}"
          f" m, place err mean/max {s['carrier_2_place_err_mean_m']['mean']:.3f} / "
          f"{s['carrier_2_place_err_max_m']['mean']:.3f} m")
    print(f"leaves used: {s['leaves_used']}   wall time {s['wall_time_s']:.0f} s")
    print("===============================================")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2))
        print(f"[mujoco-eval] wrote {out}")


if __name__ == "__main__":
    main()
