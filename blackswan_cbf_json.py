import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # headless

import pygame
import math
import random
import json
import datetime
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List, Optional

pygame.init()

#######################################################################
# RESULT CONTAINER (MATCHES SWITCHING JSON)
#######################################################################

class SimulationResult:
    def __init__(self):
        self.goals_reached: int = 0
        self.caught: bool = False
        self.time_elapsed: float = 0.0
        self.distance_trace: List[float] = []
        self.goal_times: List[float] = []

        self.black_swan_goal: Optional[int] = None
        self.black_swan_time: Optional[float] = None
        self.termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goals_reached": self.goals_reached,
            "caught": self.caught,
            "time_elapsed": self.time_elapsed,
            "distance_trace": self.distance_trace,
            "goal_times": self.goal_times,
            "black_swan_goal": self.black_swan_goal,
            "black_swan_time": self.black_swan_time,
            "termination_reason": self.termination_reason,
        }

#######################################################################
# SINGLE TRIAL (CBF, PURE-PURSUIT PURSUERS)
#######################################################################

def run_black_swan_cbf_trial(
    seed: int,
    max_goals: int = 10,
    max_time: float = 30.0,
    FPS: int = 120,
) -> Dict[str, Any]:

    random.seed(seed)
    np.random.seed(seed)

    result = SimulationResult()

    # -------------------------------------------------
    # PARAMETERS (MATCH SWITCHING)
    # -------------------------------------------------
    w, h = 800, 600

    evader_speed = 5.0
    pursuer_speed = 3.0

    CBF_ALPHA = 0.9
    CBF_SAFETY_RADIUS = 75.0

    evader_radius = 15
    pursuer_radius = 15
    goal_radius = 20
    COLLISION_RADIUS = evader_radius + pursuer_radius

    MIN_GOAL_DIST_FROM_PURSUER = 100

    # -------------------------------------------------
    # BLACK SWAN (FIXED AT GOAL 4)
    # -------------------------------------------------
    BLACK_SWAN_TRIGGER_GOAL = 4
    if max_goals <= BLACK_SWAN_TRIGGER_GOAL:
        raise ValueError("max_goals must be > 4 for Black Swan test")

    result.black_swan_goal = BLACK_SWAN_TRIGGER_GOAL

    # -------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------
    evader_x, evader_y = w // 4, h // 2

    pursuer1_x, pursuer1_y = 3 * w // 4, h // 2
    pursuer1_vx, pursuer1_vy = 0.0, 0.0

    pursuer2_active = False
    pursuer2_x, pursuer2_y = None, None
    pursuer2_vx, pursuer2_vy = 0.0, 0.0

    goal_counter = 0

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def spawn_goal_away_from_pursuer(pux, puy):
        for _ in range(100):
            gx = random.randint(0, w)
            gy = random.randint(0, h)
            if math.hypot(gx - pux, gy - puy) >= MIN_GOAL_DIST_FROM_PURSUER:
                return gx, gy
        return random.randint(0, w), random.randint(0, h)

    def spawn_pursuer_away_from_evader(evx, evy, min_dist=200):
        for _ in range(100):
            x = random.randint(0, w)
            y = random.randint(0, h)
            if math.hypot(x - evx, y - evy) >= min_dist:
                return x, y
        return random.randint(0, w), random.randint(0, h)

    def cbf_qp(evx, evy, pux, puy, pvx, pvy, desired_vx, desired_vy):
        rx = evx - pux
        ry = evy - puy
        dist_sq = rx * rx + ry * ry
        dist = math.sqrt(dist_sq) if dist_sq > 1e-6 else 1e-6

        h_val = dist_sq - CBF_SAFETY_RADIUS ** 2
        grad_h_x = 2.0 * rx
        grad_h_y = 2.0 * ry

        rel_vx_des = desired_vx - pvx
        rel_vy_des = desired_vy - pvy
        h_dot_des = grad_h_x * rel_vx_des + grad_h_y * rel_vy_des

        if h_dot_des >= -CBF_ALPHA * h_val:
            return desired_vx, desired_vy

        rhs = -CBF_ALPHA * h_val + grad_h_x * pvx + grad_h_y * pvy

        def objective(v):
            return (v[0] - desired_vx) ** 2 + (v[1] - desired_vy) ** 2

        constraints = [
            {"type": "ineq", "fun": lambda v: grad_h_x * v[0] + grad_h_y * v[1] - rhs},
            {"type": "ineq", "fun": lambda v: evader_speed ** 2 - (v[0] ** 2 + v[1] ** 2)},
        ]

        res = minimize(objective, np.array([desired_vx, desired_vy]),
                       method="SLSQP", constraints=constraints)

        if res.success:
            return float(res.x[0]), float(res.x[1])

        nx, ny = rx / dist, ry / dist
        return evader_speed * nx, evader_speed * ny

    # -------------------------------------------------
    # INITIAL GOAL
    # -------------------------------------------------
    goal_x, goal_y = spawn_goal_away_from_pursuer(pursuer1_x, pursuer1_y)

    start_time = pygame.time.get_ticks() / 1000.0
    clock = pygame.time.Clock()

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------
    while True:
        clock.tick(FPS)
        t = pygame.time.get_ticks() / 1000.0 - start_time

        # distance logging
        dists = [math.hypot(evader_x - pursuer1_x, evader_y - pursuer1_y)]
        if pursuer2_active:
            dists.append(math.hypot(evader_x - pursuer2_x, evader_y - pursuer2_y))
        result.distance_trace.append(min(dists))

        if t >= max_time:
            result.termination_reason = "time_limit"
            break

        # desired velocity (goal-seeking)
        gdx, gdy = goal_x - evader_x, goal_y - evader_y
        gdist = math.hypot(gdx, gdy)
        if gdist > 1e-6:
            des_vx = (gdx / gdist) * evader_speed
            des_vy = (gdy / gdist) * evader_speed
        else:
            des_vx = des_vy = 0.0

        # CBF constraints (both pursuers)
        des_vx, des_vy = cbf_qp(
            evader_x, evader_y,
            pursuer1_x, pursuer1_y,
            pursuer1_vx, pursuer1_vy,
            des_vx, des_vy
        )

        if pursuer2_active:
            des_vx, des_vy = cbf_qp(
                evader_x, evader_y,
                pursuer2_x, pursuer2_y,
                pursuer2_vx, pursuer2_vy,
                des_vx, des_vy
            )

        # renormalize speed
        sp = math.hypot(des_vx, des_vy)
        if sp > 1e-6:
            des_vx = des_vx * evader_speed / sp
            des_vy = des_vy * evader_speed / sp

        # update evader
        evader_x += des_vx
        evader_y += des_vy

        # pursuer 1 (pure pursuit)
        dx1, dy1 = evader_x - pursuer1_x, evader_y - pursuer1_y
        d1 = math.hypot(dx1, dy1)
        if d1 > 1e-6:
            pursuer1_vx = (dx1 / d1) * pursuer_speed
            pursuer1_vy = (dy1 / d1) * pursuer_speed
            pursuer1_x += pursuer1_vx
            pursuer1_y += pursuer1_vy

        # pursuer 2 (pure pursuit – SAME AS SWITCHING)
        if pursuer2_active:
            dx2, dy2 = evader_x - pursuer2_x, evader_y - pursuer2_y
            d2 = math.hypot(dx2, dy2)
            if d2 > 1e-6:
                pursuer2_vx = (dx2 / d2) * pursuer_speed
                pursuer2_vy = (dy2 / d2) * pursuer_speed
                pursuer2_x += pursuer2_vx
                pursuer2_y += pursuer2_vy

        # bounds
        evader_x = max(evader_radius, min(w - evader_radius, evader_x))
        evader_y = max(evader_radius, min(h - evader_radius, evader_y))
        pursuer1_x = max(pursuer_radius, min(w - pursuer_radius, pursuer1_x))
        pursuer1_y = max(pursuer_radius, min(h - pursuer_radius, pursuer1_y))
        if pursuer2_active:
            pursuer2_x = max(pursuer_radius, min(w - pursuer_radius, pursuer2_x))
            pursuer2_y = max(pursuer_radius, min(h - pursuer_radius, pursuer2_y))

        # collision
        if (evader_x - pursuer1_x) ** 2 + (evader_y - pursuer1_y) ** 2 < COLLISION_RADIUS ** 2:
            result.caught = True
            result.termination_reason = "caught"
            break
        if pursuer2_active and (evader_x - pursuer2_x) ** 2 + (evader_y - pursuer2_y) ** 2 < COLLISION_RADIUS ** 2:
            result.caught = True
            result.termination_reason = "caught"
            break

        # goal reached
        if math.hypot(evader_x - goal_x, evader_y - goal_y) <= evader_radius + goal_radius:
            goal_counter += 1
            result.goal_times.append(t)

            # Black Swan trigger (goal 4)
            if (not pursuer2_active) and goal_counter == BLACK_SWAN_TRIGGER_GOAL:
                pursuer2_x, pursuer2_y = spawn_pursuer_away_from_evader(evader_x, evader_y)
                pursuer2_active = True
                result.black_swan_time = t

            if goal_counter >= max_goals:
                result.termination_reason = "max_goals"
                break

            goal_x, goal_y = spawn_goal_away_from_pursuer(pursuer1_x, pursuer1_y)

    result.goals_reached = goal_counter
    result.time_elapsed = t
    return result.to_dict()

#######################################################################
# MULTI-TRIAL RUNNER
#######################################################################

def run_trials(num_trials: int = 15):
    results = []
    for i in range(num_trials):
        res = run_black_swan_cbf_trial(seed=5*i)
        results.append(res)
        print(
            f"Trial {i}: goals={res['goals_reached']} "
            f"caught={res['caught']} "
            f"reason={res['termination_reason']}"
        )

    fname = f"black_swan_cbf_goal4_fair_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    with open(fname, "w") as f:
        json.dump({"trials": results}, f, indent=2)

    print(f"\nSaved to {fname}")

#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":
    run_trials()
