"""
Trajectory tracking task.

The vessel must follow a continuously-moving reference point that traces
one of several parametric curves:
    - Sinusoidal  — classic oscillating path
    - Lemniscate  — Bernoulli figure-8 (good for heading-change stress testing)
    - Circle      — constant-curvature path
    - Straight    — constant-heading baseline
    - Spline      — Catmull-Rom spline through random control points (Phase 2)

Reference:
    Catmull-Rom: Barry & Goldman (1988) "A recursive evaluation algorithm for
    a class of Catmull-Rom splines".
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from configs.blueboat_cfg import EnvironmentConfig


class TrajectoryType(str, Enum):
    SINUSOIDAL = "sinusoidal"
    LEMNISCATE = "lemniscate"
    CIRCLE = "circle"
    STRAIGHT = "straight"
    SPLINE = "spline"


# Number of Catmull-Rom control points for spline trajectories
_SPLINE_NUM_CTRL = 8


class TrajectoryTrackingTask:
    """Manages per-environment reference trajectories."""

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        num_envs: int,
        device: str = "cuda",
        traj_type: TrajectoryType = TrajectoryType.SINUSOIDAL,
    ) -> None:
        self.env_cfg = env_cfg
        self.num_envs = num_envs
        self.device = device
        self.traj_type = traj_type

        # Common trajectory parameters (randomised per env at reset)
        self._amplitude = torch.ones(num_envs, device=device) * 10.0
        self._frequency = torch.ones(num_envs, device=device) * 0.1
        self._phase = torch.zeros(num_envs, device=device)
        self._speed = torch.ones(num_envs, device=device) * 0.5
        self._origin = torch.zeros(num_envs, 2, device=device)

        # Spline control points: (num_envs, num_ctrl, 2)
        self._ctrl_pts = torch.zeros(
            num_envs, _SPLINE_NUM_CTRL, 2, device=device
        )

        # Parametric time per environment
        self._t = torch.zeros(num_envs, device=device)

        # Current target
        self._target_pos = torch.zeros(num_envs, 2, device=device)
        self._target_heading = torch.zeros(num_envs, device=device)

        all_ids = torch.arange(num_envs, device=device)
        self.reset(all_ids)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Tensor) -> None:
        """Re-initialise trajectory parameters for specified environments."""
        n = env_ids.numel()
        if n == 0:
            return

        self._amplitude[env_ids] = 5.0 + 10.0 * torch.rand(n, device=self.device)
        self._frequency[env_ids] = 0.05 + 0.15 * torch.rand(n, device=self.device)
        self._phase[env_ids] = torch.rand(n, device=self.device) * 2.0 * torch.pi
        self._speed[env_ids] = 0.3 + 0.7 * torch.rand(n, device=self.device)
        self._origin[env_ids] = (
            torch.rand(n, 2, device=self.device) - 0.5
        ) * 20.0

        self._t[env_ids] = torch.rand(n, device=self.device) * 2.0 * torch.pi

        # Sample fresh Catmull-Rom control points for spline trajectories
        if self.traj_type == TrajectoryType.SPLINE:
            self._sample_ctrl_points(env_ids)

        self._update_target(env_ids)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance trajectory time and recompute targets for all envs."""
        self._t += self._speed * dt

        # Wrap spline parameter to loop cleanly
        if self.traj_type == TrajectoryType.SPLINE:
            self._t = self._t % float(_SPLINE_NUM_CTRL - 1)

        all_ids = torch.arange(self.num_envs, device=self.device)
        self._update_target(all_ids)

    # ------------------------------------------------------------------
    # Target accessors
    # ------------------------------------------------------------------

    @property
    def target_pos(self) -> Tensor:
        return self._target_pos

    @property
    def target_heading(self) -> Tensor:
        return self._target_heading

    def get_distance_to_target(self, pos: Tensor) -> Tensor:
        return ((pos - self._target_pos) ** 2).sum(dim=-1).sqrt()

    # ------------------------------------------------------------------
    # Parametric curve evaluation
    # ------------------------------------------------------------------

    def _update_target(self, env_ids: Tensor) -> None:
        t = self._t[env_ids]
        A = self._amplitude[env_ids]
        f = self._frequency[env_ids]
        phi = self._phase[env_ids]
        origin = self._origin[env_ids]

        pos, heading = self._evaluate_trajectory(t, A, f, phi, origin, env_ids)
        self._target_pos[env_ids] = pos
        self._target_heading[env_ids] = heading

    def _evaluate_trajectory(
        self,
        t: Tensor,
        A: Tensor,
        f: Tensor,
        phi: Tensor,
        origin: Tensor,
        env_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.traj_type == TrajectoryType.SINUSOIDAL:
            return self._sinusoidal(t, A, f, phi, origin)
        elif self.traj_type == TrajectoryType.LEMNISCATE:
            return self._lemniscate(t, A, f, phi, origin)
        elif self.traj_type == TrajectoryType.CIRCLE:
            return self._circle(t, A, f, phi, origin)
        elif self.traj_type == TrajectoryType.SPLINE:
            return self._catmull_rom(t, env_ids, origin)
        else:  # STRAIGHT
            return self._straight(t, A, f, phi, origin)

    # ------------------------------------------------------------------
    # Individual curve implementations
    # ------------------------------------------------------------------

    def _sinusoidal(self, t, A, f, phi, origin):
        tt = t * f + phi
        x = A * tt / (2.0 * torch.pi)
        y = A * 0.5 * torch.sin(tt)
        dx_dt = A * f / (2.0 * torch.pi)
        dy_dt = A * 0.5 * torch.cos(tt) * f
        pos = torch.stack([x, y], dim=-1) + origin
        return pos, torch.atan2(dy_dt, dx_dt)

    def _lemniscate(self, t, A, f, phi, origin):
        """Bernoulli lemniscate — figure-8 with heading-reversal stress."""
        tt = t * f + phi
        denom = 1.0 + torch.sin(tt) ** 2
        x = A * torch.cos(tt) / denom
        y = A * torch.sin(tt) * torch.cos(tt) / denom
        # Numerical tangent (analytic is messy, eps is sufficient)
        eps = 1e-4
        tt2 = tt + eps
        denom2 = 1.0 + torch.sin(tt2) ** 2
        x2 = A * torch.cos(tt2) / denom2
        y2 = A * torch.sin(tt2) * torch.cos(tt2) / denom2
        dx_dt = (x2 - x) / eps
        dy_dt = (y2 - y) / eps
        pos = torch.stack([x, y], dim=-1) + origin
        return pos, torch.atan2(dy_dt, dx_dt)

    def _circle(self, t, A, f, phi, origin):
        tt = t * f + phi
        x = A * torch.cos(tt)
        y = A * torch.sin(tt)
        dx_dt = -A * torch.sin(tt) * f
        dy_dt = A * torch.cos(tt) * f
        pos = torch.stack([x, y], dim=-1) + origin
        return pos, torch.atan2(dy_dt, dx_dt)

    def _straight(self, t, A, f, phi, origin):
        x = A * t
        y = torch.zeros_like(t)
        pos = torch.stack([x, y], dim=-1) + origin
        heading = torch.zeros_like(t)
        return pos, heading

    def _catmull_rom(
        self,
        t_global: Tensor,
        env_ids: Tensor,
        origin: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate Catmull-Rom spline at global parameter t_global.

        t_global ∈ [0, N-1) where N = _SPLINE_NUM_CTRL.
        Segment index = floor(t_global), clamped to [0, N-3].
        Local t ∈ [0, 1] within the segment.

        Catmull-Rom formula for segment P1→P2 with neighbours P0, P3:
            q(t) = 0.5 * ((2P1)
                        + (-P0 + P2)*t
                        + (2P0 - 5P1 + 4P2 - P3)*t²
                        + (-P0 + 3P1 - 3P2 + P3)*t³)
        """
        N = _SPLINE_NUM_CTRL
        ctrl = self._ctrl_pts[env_ids]   # (n_ids, N, 2)
        n_ids = env_ids.numel()
        env_range = torch.arange(n_ids, device=self.device)

        # Segment index and local t
        seg = t_global.long().clamp(0, N - 3)          # (n_ids,)
        t_loc = (t_global - seg.float()).unsqueeze(-1)  # (n_ids, 1)
        t2 = t_loc ** 2
        t3 = t_loc ** 3

        # Gather the four control points with clamped boundary indices
        def _gather(idx_offset: int) -> Tensor:
            idx = (seg + idx_offset).clamp(0, N - 1)
            return ctrl[env_range, idx]  # (n_ids, 2)

        P0 = _gather(-1)
        P1 = _gather(0)
        P2 = _gather(1)
        P3 = _gather(2)

        # Position
        pos = 0.5 * (
            2.0 * P1
            + (-P0 + P2) * t_loc
            + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t2
            + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t3
        ) + origin  # (n_ids, 2)

        # Tangent (derivative w.r.t. t_loc)
        tangent = 0.5 * (
            (-P0 + P2)
            + 2.0 * (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * t_loc
            + 3.0 * (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * t2
        )  # (n_ids, 2)

        heading = torch.atan2(tangent[:, 1], tangent[:, 0])

        return pos, heading

    def _sample_ctrl_points(self, env_ids: Tensor) -> None:
        """Sample random Catmull-Rom control points on a reasonable scale."""
        n = env_ids.numel()
        A = self._amplitude[env_ids]  # (n,)
        # Distribute control points roughly in a region of radius A
        # Use a progressive walk so the spline stays continuous and smooth
        pts = torch.zeros(n, _SPLINE_NUM_CTRL, 2, device=self.device)
        for k in range(_SPLINE_NUM_CTRL):
            angle = k * (2.0 * torch.pi / _SPLINE_NUM_CTRL)
            r = A * (0.5 + 0.5 * torch.rand(n, device=self.device))
            pts[:, k, 0] = r * torch.cos(torch.tensor(angle, device=self.device))
            pts[:, k, 1] = r * torch.sin(torch.tensor(angle, device=self.device))
        # Add small random perturbations so the spline is not a perfect polygon
        pts += (torch.rand_like(pts) - 0.5) * A.unsqueeze(-1).unsqueeze(-1) * 0.3
        self._ctrl_pts[env_ids] = pts
