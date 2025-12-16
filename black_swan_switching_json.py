import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # headless

import pygame
import math
import random
import json
import datetime
from typing import List, Dict, Any, Optional

pygame.init()

#######################################################################
# RESULT CONTAINER
#######################################################################

class SimulationResult:
    def __init__(self):
        self.goals_reached: int = 0
        self.caught: bool = False
        self.time_elapsed: float = 0.0
        self.goal_times: List[float] = []
        self.distance_trace: List[float] = []

        # Black Swan
        self.black_swan_goal: Optional[int] = None
        self.black_swan_time: Optional[float] = None
        self.termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goals_reached": self.goals_reached,
            "caught": self.caught,
            "time_elapsed": self.time_elapsed,
            "goal_times": self.goal_times,
            "distance_trace": self.distance_trace,
            "black_swan_goal": self.black_swan_goal,
            "black_swan_time": self.black_swan_time,
            "termination_reason": self.termination_reason,
        }

#######################################################################
# SINGLE TRIAL
#######################################################################

def run_black_swan_switching_trial(
    seed: int,
    max_goals: int = 10,
    max_time: float = 30.0,
    FPS: int = 120,
):

    random.seed(seed)

    # -------------------------------------------------
    # PARAMETERS (UNCHANGED FROM ORIGINAL SWITCHING)
    # -------------------------------------------------
    w, h = 800, 600

    evader_speed = 6.0
    pursuer_speed = 3.0

    evader_radius = 15
    pursuer_radius = 15
    goal_radius = 20
    COLLISION_RADIUS = evader_radius + pursuer_radius

    SAFE_CHECK_HORIZON = 12
    BUFFER_DIST = 20
    alpha_avoidance = 0.9

    MIN_GOAL_DIST_FROM_PURSUER = 100

    # -------------------------------------------------
    # BLACK SWAN STANDARDIZATION
    # -------------------------------------------------
    BLACK_SWAN_TRIGGER_GOAL = 4
    if max_goals <= BLACK_SWAN_TRIGGER_GOAL:
        raise ValueError(
            f"max_goals ({max_goals}) must be > BLACK_SWAN_TRIGGER_GOAL ({BLACK_SWAN_TRIGGER_GOAL})"
        )

    result = SimulationResult()
    result.black_swan_goal = BLACK_SWAN_TRIGGER_GOAL

    # -------------------------------------------------
    # INITIAL STATE
    # -------------------------------------------------
    evader_x, evader_y = w // 4, h // 2

    pursuer1_x, pursuer1_y = 3 * w // 4, h // 2

    pursuer2_active = False
    pursuer2_x, pursuer2_y = None, None

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

    def will_be_captured(evx, evy, pux, puy, horizon):
        ex, ey = evx, evy
        px, py = pux, puy
        for _ in range(horizon):
            dx, dy = ex - px, ey - py
            d = math.hypot(dx, dy)
            if d <= COLLISION_RADIUS:
                return True
            if d > 1e-6:
                px += (dx / d) * pursuer_speed
                py += (dy / d) * pursuer_speed
            gx, gy = goal_x, goal_y
            gdx, gdy = gx - ex, gy - ey
            gd = math.hypot(gdx, gdy)
            if gd > 1e-6:
                ex += (gdx / gd) * evader_speed
                ey += (gdy / gd) * evader_speed
        return False

    def will_be_captured_multi(evx, evy, pursuers):
        return any(
            will_be_captured(evx, evy, px, py, SAFE_CHECK_HORIZON)
            for px, py in pursuers
        )

    def early_warning(evx, evy, pursuers):
        for px, py in pursuers:
            if math.hypot(evx - px, evy - py) <= COLLISION_RADIUS + BUFFER_DIST:
                return True
        return False

    def choose_evader_action(evx, evy, goalx, goaly, pursuers):
        # Candidate directions
        angles = [i * math.pi / 8 for i in range(16)]
        best = None
        best_score = -1e9

        for ang in angles:
            nx = evx + evader_speed * math.cos(ang)
            ny = evy + evader_speed * math.sin(ang)

            # bounds
            if not (0 <= nx <= w and 0 <= ny <= h):
                continue

            # safety check
            if will_be_captured_multi(nx, ny, pursuers):
                continue

            # distance improvement
            score = -math.hypot(goalx - nx, goaly - ny)
            if score > best_score:
                best_score = score
                best = (nx, ny)

        if best is not None:
            return best

        # fallback: goal direction
        dx, dy = goalx - evx, goaly - evy
        d = math.hypot(dx, dy)
        if d > 1e-6:
            return (
                evx + (dx / d) * evader_speed,
                evy + (dy / d) * evader_speed,
            )
        return evx, evy

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

        if t >= max_time:
            result.termination_reason = "time_limit"
            break

        pursuers = [(pursuer1_x, pursuer1_y)]
        if pursuer2_active:
            pursuers.append((pursuer2_x, pursuer2_y))

        # log distance
        result.distance_trace.append(
            min(math.hypot(evader_x - px, evader_y - py) for px, py in pursuers)
        )

        # evader move
        evader_x, evader_y = choose_evader_action(
            evader_x, evader_y, goal_x, goal_y, pursuers
        )

        # pursuer 1
        dx, dy = evader_x - pursuer1_x, evader_y - pursuer1_y
        d = math.hypot(dx, dy)
        if d > 1e-6:
            pursuer1_x += (dx / d) * pursuer_speed
            pursuer1_y += (dy / d) * pursuer_speed

        # pursuer 2
        if pursuer2_active:
            dx2, dy2 = evader_x - pursuer2_x, evader_y - pursuer2_y
            d2 = math.hypot(dx2, dy2)
            if d2 > 1e-6:
                pursuer2_x += (dx2 / d2) * pursuer_speed
                pursuer2_y += (dy2 / d2) * pursuer_speed

        # collision
        for px, py in pursuers:
            if math.hypot(evader_x - px, evader_y - py) <= COLLISION_RADIUS:
                result.caught = True
                result.termination_reason = "caught"
                break
        if result.caught:
            break

        # goal reached
        if math.hypot(evader_x - goal_x, evader_y - goal_y) <= evader_radius + goal_radius:
            goal_counter += 1
            result.goal_times.append(t)

            # Black Swan trigger (fixed at goal 6)
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
        res = run_black_swan_switching_trial(seed=i)
        results.append(res)
        print(
            f"Trial {i}: goals={res['goals_reached']} "
            f"caught={res['caught']} "
            f"reason={res['termination_reason']}"
        )

    fname = f"black_swan_switching_goal6_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    with open(fname, "w") as f:
        json.dump({"trials": results}, f, indent=2)

    print(f"\nSaved to {fname}")

#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":
    run_trials()
