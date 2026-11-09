from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn as nn
from muscle_supermimic.rl.supermimic.networks import RunningNorm, mlp

VEL_TAU = 0.25
LOOKAHEAD_S = 0.2
QD_DT = 0.02


def _mlp_from(sd: dict) -> nn.Sequential:
    w = [sd[k] for k in sorted((k for k in sd if k.endswith(".weight")), key=lambda k: int(k.split(".")[0]))]
    net = mlp(w[0].shape[1], [x.shape[0] for x in w[:-1]], w[-1].shape[0])
    net.load_state_dict(sd)
    return net


class Policy(nn.Module):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.dec, self.enc = _mlp_from(ck["dec"]), _mlp_from(ck["enc"])
        self.obs_norm = RunningNorm(ck["obs_norm"]["mean"].shape[0])
        self.goal_norm = RunningNorm(ck["goal_norm"]["mean"].shape[0])
        self.obs_norm.load_state_dict(ck["obs_norm"])
        self.goal_norm.load_state_dict(ck["goal_norm"])
        self.eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        o = self.obs_norm(obs)
        z = torch.tanh(self.enc(torch.cat((self.goal_norm(goal), o), -1)))
        return torch.sigmoid(self.dec(torch.cat((z, o), -1)))


class GaitTable:
    def __init__(self, d: dict) -> None:
        self.rest, self.contact, self.stand = (torch.as_tensor(d[k]) for k in ("rest", "contact", "stand"))
        self.speed, self.period = torch.as_tensor(d["speed"]), torch.as_tensor(d["period"])
        self.f_min, self.f_max, self.turn_amp = float(d["f_min"]), float(d["f_max"]), float(d["turn_amp"])
        self.n = self.rest.shape[1]

    def blend(self, v: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        speed = v.norm(dim=-1)
        ang = torch.atan2(v[..., 1], v[..., 0]) % (2 * math.pi)
        x = ang / (math.pi / 2)
        i0 = x.floor().long() % 4
        t = x - x.floor()
        wts = torch.zeros(v.shape[:-1] + (4,), dtype=v.dtype)
        wts.scatter_(-1, i0.unsqueeze(-1), (1 - t).unsqueeze(-1))
        wts.scatter_add_(-1, ((i0 + 1) % 4).unsqueeze(-1), t.unsqueeze(-1))
        nat_speed, nat_period = (wts * self.speed).sum(-1), (wts * self.period).sum(-1)
        freq = ((speed.clamp(min=1e-3) / nat_speed).sqrt() / nat_period).clamp(self.f_min, self.f_max)
        amp = (speed / (freq * nat_speed * nat_period)).clamp(0.0, 1.2)
        turn = (speed < 0.02) & (w.abs() > 0.05)
        side = torch.where(w > 0, 1, 3)
        wt = torch.zeros_like(wts).scatter_(-1, side.unsqueeze(-1), 1.0)
        wts = torch.where(turn.unsqueeze(-1), wt, wts)
        amp = torch.where(turn, self.turn_amp * (w.abs() / 0.3).clamp(max=1.0), amp)
        freq = torch.where(turn, torch.full_like(freq, self.f_min), freq)
        return wts, freq, amp

    def pose(self, phase: torch.Tensor, wts: torch.Tensor, amp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (phase % 1.0) * self.n
        i = x.floor().long() % self.n
        j = (i + 1) % self.n
        t = (x - x.floor()).unsqueeze(-1).unsqueeze(-1)
        rest = self.rest[:, i].movedim(0, -2) * (1 - t) + self.rest[:, j].movedim(0, -2) * t
        rest = (wts.unsqueeze(-1) * rest).sum(-2)
        q = self.stand + amp.clamp(max=1.0).unsqueeze(-1) * (rest - self.stand)
        ct = (wts.unsqueeze(-1) * self.contact[:, i].movedim(0, -2)).sum(-2) > 0.5
        return q, torch.where((amp > 0.02).unsqueeze(-1), ct, torch.ones_like(ct))

    def advance(self, phase: torch.Tensor, vs: torch.Tensor, cmd: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        _, freq, _ = self.blend(vs[..., :2], vs[..., 2])
        return (phase + freq * dt) % 1.0, vs + (cmd - vs) * (1 - math.exp(-dt / VEL_TAU))

    def goal(self, phase: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
        wts, freq, amp = self.blend(vs[..., :2], vs[..., 2])
        q0, down = self.pose(phase, wts, amp)
        q1, _ = self.pose(phase + LOOKAHEAD_S * freq, wts, amp)
        qd = (self.pose(phase + QD_DT * freq, wts, amp)[0] - self.pose(phase - QD_DT * freq, wts, amp)[0]) / (2 * QD_DT)
        return torch.cat((q0, q1, down.to(q0.dtype), qd), -1)
