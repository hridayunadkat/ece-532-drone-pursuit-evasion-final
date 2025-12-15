import pygame
import math
import random
import time
import json
import datetime
from typing import Dict, Any, List

class SimulationResult:
    def __init__(self):
        self.goals_reached: int = 0
        self.caught: bool = False
        self.time_elapsed: float = 0.0
        self.distance_trace: List[float] = []
        self.goal_times: List[float] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goals_reached': self.goals_reached,
            'caught': self.caught,
            'time_elapsed': self.time_elapsed,
            'distance_trace': self.distance_trace,
            'goal_times': self.goal_times
        }

#######################################################################
# RUN SWITCHING FILTER SIMULATION
#######################################################################

def run_switching_filter(seed: int = None, max_goals: int = 10, max_time: float = 30.0) -> Dict[str, Any]:
    
    if seed is not None:
        random.seed(seed)
    
    result = SimulationResult()

    PREDICTION_HORIZON = 60
    SAFE_CHECK_HORIZON = PREDICTION_HORIZON // 2
    VIS_PRED_STEPS = 40
    
    GRID_SPACING = 40
    GRID_MARGIN = 0
    GRID_EVAL_EVERY_N = 6
    RECOMPUTE_MOVE_THRESH = 2.0
    
    w, h = 800, 600
    
    evader_speed = 6.0
    pursuer_speed = 2.0
    rel_speed = evader_speed / pursuer_speed
    if rel_speed <=3.0:
        alpha_avoidance = 0.99
    else:
        alpha_avoidance = 1
    tangent_offset = math.pi / 6 
    
    evader_radius, pursuer_radius, goal_radius = 15, 15, 20
    
    NUM_CANDIDATES = 12
    SAFE_MARGIN = 2.0
    BUFFER_DIST = 10  
    
    FPS = 120
    BASIN_ALPHA = 120
    
    WHITE = (255,255,255)
    BLUE = (50,150,255)
    RED = (255,80,80)
    GREEN = (50,200,50)
    BLACK = (0,0,0)
    PRED_E_COLOR = (0,120,255)
    PRED_P_COLOR = (220,30,30)
    BASIN_COLOR = (255,80,80, BASIN_ALPHA)
    
    
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Fixed Switching Filter with Buffer & Tangent Escape")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    
    evader_x = w//4
    evader_y = h//2
    pursuer_x = 3*w//4
    pursuer_y = h//2
    goal_counter = 0
    goal_times = []
    
    goal_x, goal_y = random.randint(0, w), random.randint(0, h)
    goal_spawn_time = pygame.time.get_ticks() / 1000.0
    running = True
    game_over = False
    
    frame_count = 0
    _last_pursuer_pos = (pursuer_x, pursuer_y)
    choice_idx = 0
    unsafe_frames = 0
    
    start_time = pygame.time.get_ticks() / 1000.0
    
    def will_be_captured(evx, evy, pux, puy, horizon=PREDICTION_HORIZON):
        e_x, e_y = float(evx), float(evy)
        p_x, p_y = float(pux), float(puy)
        for _ in range(horizon):
            dx = e_x - p_x
            dy = e_y - p_y
            dist = math.hypot(dx, dy)
            if dist <= (evader_radius + pursuer_radius):
                return True
            if dist > 1e-6:
                ev_vx = (dx / dist) * evader_speed
                ev_vy = (dy / dist) * evader_speed
            else:
                ev_vx, ev_vy = 0.0, 0.0
            e_x += ev_vx
            e_y += ev_vy
            e_x = max(evader_radius, min(w - evader_radius, e_x))
            e_y = max(evader_radius, min(h - evader_radius, e_y))
            lookahead = 3.0
            pred_ex = e_x + ev_vx * lookahead
            pred_ey = e_y + ev_vy * lookahead
            pdx = pred_ex - p_x
            pdy = pred_ey - p_y
            pdist = math.hypot(pdx, pdy)
            if pdist > 1e-6:
                p_x += (pdx / pdist) * pursuer_speed
            p_x = max(pursuer_radius, min(w - pursuer_radius, p_x))
            p_y = max(pursuer_radius, min(h - pursuer_radius, p_y))
            if math.hypot(e_x - p_x, e_y - p_y) <= evader_radius + pursuer_radius:
                return True
        return False
    
    grid_xs = list(range(GRID_MARGIN + GRID_SPACING//2, w, GRID_SPACING))
    grid_ys = list(range(GRID_MARGIN + GRID_SPACING//2, h, GRID_SPACING))
    nx = len(grid_xs)
    ny = len(grid_ys)
    basin_grid = [[False]*ny for _ in range(nx)]
    basin_stamp = -1
    
    def recompute_basin(p_x, p_y):
        nonlocal basin_grid, basin_stamp
        for i, gx in enumerate(grid_xs):
            for j, gy in enumerate(grid_ys):
                basin_grid[i][j] = will_be_captured(gx, gy, p_x, p_y, horizon=SAFE_CHECK_HORIZON)
        basin_stamp = frame_count
    
    def is_in_sampled_basin(x, y):
        i = int(round((x - (GRID_SPACING/2)) / GRID_SPACING))
        j = int(round((y - (GRID_SPACING/2)) / GRID_SPACING))
        i = max(0, min(nx-1, i))
        j = max(0, min(ny-1, j))
        return basin_grid[i][j]
    
    def get_predicted_paths(evx, evy, pux, puy, mode='SAFETY', steps=VIS_PRED_STEPS):
        e_x, e_y = float(evx), float(evy)
        p_x, p_y = float(pux), float(puy)
        e_path, p_path = [], []
        for _ in range(steps):
            dx = e_x - p_x
            dy = e_y - p_y
            dist = math.hypot(dx, dy)
            if mode == 'GOAL':
                gdx = goal_x - e_x
                gdy = goal_y - e_y
                gdist = math.hypot(gdx, gdy)
                if gdist > 1e-6:
                    ev_vx = (gdx / gdist) * evader_speed
                    ev_vy = (gdy / gdist) * evader_speed
                else:
                    ev_vx, ev_vy = 0.0, 0.0
            else:
                if dist > 1e-6:
                    ev_vx = (dx / dist) * evader_speed
                    ev_vy = (dy / dist) * evader_speed
                else:
                    ev_vx, ev_vy = 0.0, 0.0
            e_x += ev_vx
            e_y += ev_vy
            lookahead = 3.0
            pred_ex = e_x + ev_vx * lookahead
            pred_ey = e_y + ev_vy * lookahead
            pdx = pred_ex - p_x
            pdy = pred_ey - p_y
            pdist = math.hypot(pdx, pdy)
            if pdist > 1e-6:
                p_x += (pdx / pdist) * pursuer_speed
                p_y += (pdy / pdist) * pursuer_speed
            e_x = max(evader_radius, min(w - evader_radius, e_x))
            e_y = max(evader_radius, min(h - evader_radius, e_y))
            p_x = max(pursuer_radius, min(w - pursuer_radius, p_x))
            p_y = max(pursuer_radius, min(h - pursuer_radius, p_y))
            e_path.append((e_x, e_y))
            p_path.append((p_x, p_y))
            if math.hypot(e_x - p_x, e_y - p_y) <= evader_radius + pursuer_radius:
                break
        return e_path, p_path
    
    def early_warning(evx, evy, pux, puy):
        dx = evx - pux
        dy = evy - puy
        dist = math.hypot(dx, dy)
        return dist < (evader_radius + pursuer_radius + BUFFER_DIST)
    
    def avoid_goal_collision(x, y, radius, goalx, goaly, goal_r):
        dx = x - goalx
        dy = y - goaly
        dist = math.hypot(dx, dy)
        min_dist = radius + goal_r
        
        if dist < min_dist and dist > 1e-6:
            x = goalx + (dx / dist) * min_dist
            y = goaly + (dy / dist) * min_dist
        return x, y
    
    def choose_evader_action(evx, evy, goalx, goaly, pux, puy):
        gdx = goalx - evx
        gdy = goaly - evy
        gdist = math.hypot(gdx, gdy)
        goal_dir = math.atan2(gdy, gdx) if gdist > 1e-6 else 0.0
        
        pdx = evx - pux
        pdy = evy - puy
        pdist = math.hypot(pdx, pdy)
        away_dir = math.atan2(pdy, pdx) if pdist > 1e-6 else goal_dir
        
        angle_to_pursuer = math.atan2(evy - puy, evx - pux)
        tangent_angles = [angle_to_pursuer + tangent_offset, angle_to_pursuer - tangent_offset]
        
        candidates = [goal_dir, away_dir] + tangent_angles
        for k in range(NUM_CANDIDATES):
            frac = k / max(1, NUM_CANDIDATES - 1)
            diff = ((away_dir - goal_dir + math.pi) % (2*math.pi)) - math.pi
            candidates.append(goal_dir + diff * frac)
        
        safe_candidates = []
        scored_candidates = []
        
        for idx, angle in enumerate(candidates):
            nx = evx + math.cos(angle) * evader_speed
            ny = evy + math.sin(angle) * evader_speed
            nx = max(evader_radius, min(w - evader_radius, nx))
            ny = max(evader_radius, min(h - evader_radius, ny))
            
            
            if nx < evader_radius*2:
                angle += math.radians(30)  
            elif nx > w - evader_radius*2:
                angle -= math.radians(30)  
            if ny < evader_radius*2:
                angle += math.radians(30)  
            elif ny > h - evader_radius*2:
                angle -= math.radians(30) 
            
            nx = evx + math.cos(angle) * evader_speed
            ny = evy + math.sin(angle) * evader_speed
            nx = max(evader_radius, min(w - evader_radius, nx))
            ny = max(evader_radius, min(h - evader_radius, ny))
            
            if math.hypot(nx - pux, ny - puy) < alpha_avoidance * math.hypot(evx - pux, evy - puy):
                continue
            
            if not is_in_sampled_basin(nx, ny):
                ang_diff = abs(((angle - goal_dir + math.pi) % (2*math.pi)) - math.pi)
                safe_candidates.append((ang_diff, idx, nx, ny, angle))
            else:
                captured = will_be_captured(nx, ny, pux, puy, horizon=SAFE_CHECK_HORIZON)
                score = 0.0 if captured else 0.5
                scored_candidates.append((score, idx, nx, ny, angle))
        
        if safe_candidates:
            safe_candidates.sort(key=lambda t: (t[0], t[1]))
            _, chosen_idx, nx, ny, _ = safe_candidates[0]
            return nx, ny, chosen_idx, True
        if scored_candidates:
            scored_candidates.sort(key=lambda t: (-t[0], t[1]))
            _, chosen_idx, nx, ny, _ = scored_candidates[0]
            return nx, ny, chosen_idx, False
        return evx, evy, 0, False
    
    print("\n=== Running Switching Filter Simulation ===")
    
    while running and not game_over:
        clock.tick(FPS)
        screen.fill(WHITE)
        frame_count += 1
        
        current_time = pygame.time.get_ticks() / 1000.0 - start_time
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                game_over = True
        
        if not game_over:
            dist_to_pursuer = math.hypot(evader_x - pursuer_x, evader_y - pursuer_y)
            result.distance_trace.append(dist_to_pursuer)
            
            if current_time >= max_time:
                result.goals_reached = goal_counter
                result.time_elapsed = current_time
                game_over = True
                print(f"Time's up at {max_time}s.")
                break
            
            dxp = pursuer_x - _last_pursuer_pos[0]
            dyp = pursuer_y - _last_pursuer_pos[1]
            if frame_count == 1 or math.hypot(dxp, dyp) >= RECOMPUTE_MOVE_THRESH or frame_count % GRID_EVAL_EVERY_N == 0:
                recompute_basin(pursuer_x, pursuer_y)
                _last_pursuer_pos = (pursuer_x, pursuer_y)
            
            next_ex, next_ey, choice_idx, step_safe = choose_evader_action(
                evader_x, evader_y, goal_x, goal_y, pursuer_x, pursuer_y
            )
            
            current_unsafe = is_in_sampled_basin(evader_x, evader_y) or early_warning(evader_x, evader_y, pursuer_x, pursuer_y)
            unsafe_frames = unsafe_frames + 1 if current_unsafe else max(unsafe_frames - 1, 0)
            mode = "SAFETY" if current_unsafe else "GOAL"
            
            evader_x, evader_y = next_ex, next_ey
            evader_x = max(evader_radius, min(w - evader_radius, evader_x))
            evader_y = max(evader_radius, min(h - evader_radius, evader_y))
            
            pdx = evader_x - pursuer_x
            pdy = evader_y - pursuer_y
            pdist = math.hypot(pdx, pdy)
            if pdist > 1e-6:
                pursuer_x += (pdx / pdist) * pursuer_speed
                pursuer_y += (pdy / pdist) * pursuer_speed
            
            pursuer_x, pursuer_y = avoid_goal_collision(pursuer_x, pursuer_y, pursuer_radius,
                                                        goal_x, goal_y, goal_radius)
            
            pursuer_x = max(pursuer_radius, min(w - pursuer_radius, pursuer_x))
            pursuer_y = max(pursuer_radius, min(h - pursuer_radius, pursuer_y))
            
            if math.hypot(evader_x - pursuer_x, evader_y - pursuer_y) <= evader_radius + pursuer_radius:
                result.caught = True
                result.time_elapsed = current_time
                game_over = True
                print(f"Evader caught at {current_time:.2f}s!")
                break
            
            if goal_counter < max_goals and math.hypot(evader_x - goal_x, evader_y - goal_y) <= evader_radius + goal_radius:
                goal_time = current_time - (goal_spawn_time - start_time)
                goal_times.append(goal_time)
                result.goal_times.append(goal_time)
                goal_counter += 1
                print(f"Goal {goal_counter} reached at {current_time:.2f}s")
                
                if goal_counter >= max_goals:
                    result.goals_reached = goal_counter
                    result.time_elapsed = current_time
                    game_over = True
                    print("All goals collected!")
                else:
                    goal_x, goal_y = random.randint(0, w), random.randint(0, h)
                    goal_spawn_time = pygame.time.get_ticks() / 1000.0
        
        pred_e_path, pred_p_path = get_predicted_paths(evader_x, evader_y, pursuer_x, pursuer_y, mode=mode)
        
        overlay = pygame.Surface((w, h), flags=pygame.SRCALPHA)
        overlay.fill((0,0,0,0))
        for i, gx in enumerate(grid_xs):
            for j, gy in enumerate(grid_ys):
                if basin_grid[i][j]:
                    left = gx - GRID_SPACING//2
                    top  = gy - GRID_SPACING//2
                    rect = pygame.Rect(left, top, GRID_SPACING, GRID_SPACING)
                    overlay.fill((BASIN_COLOR[0], BASIN_COLOR[1], BASIN_COLOR[2], BASIN_ALPHA), rect)
        screen.blit(overlay, (0,0))
        
        for i in range(1, len(pred_e_path)):
            pygame.draw.line(screen, PRED_E_COLOR,
                             (int(pred_e_path[i-1][0]), int(pred_e_path[i-1][1])),
                             (int(pred_e_path[i][0]), int(pred_e_path[i][1])), 2)
        for i in range(1, len(pred_p_path)):
            pygame.draw.line(screen, PRED_P_COLOR,
                             (int(pred_p_path[i-1][0]), int(pred_p_path[i-1][1])),
                             (int(pred_p_path[i][0]), int(pred_p_path[i][1])), 2)
        
        pygame.draw.circle(screen, RED, (int(pursuer_x), int(pursuer_y)), pursuer_radius)
        pygame.draw.circle(screen, BLUE, (int(evader_x), int(evader_y)), evader_radius)
        if goal_counter < max_goals:
            pygame.draw.circle(screen, GREEN, (int(goal_x), int(goal_y)), goal_radius)
        pygame.draw.line(screen, BLACK, (int(evader_x), int(evader_y)), (int(pursuer_x), int(pursuer_y)), 1)
        
        if not game_over:
            screen.blit(font.render(f"Goals: {goal_counter}/{max_goals}", True, GREEN), (10, 8))
            screen.blit(font.render(f"Mode: {mode}", True, RED if mode!="GOAL" else GREEN), (10, 32))
            screen.blit(font.render(f"Grid: {nx}x{ny}", True, BLACK), (10, 56))
            screen.blit(font.render(f"Choice idx: {choice_idx}", True, BLACK), (10, 80))
            screen.blit(font.render(f"Pred H: {PREDICTION_HORIZON}", True, BLACK), (10, 104))
            screen.blit(font.render(f"Time: {current_time:.1f}s", True, BLACK), (10, 128))
        else:
            if goal_counter >= max_goals:
                msg = font.render(f"VICTORY! {goal_counter} goals", True, GREEN)
            else:
                msg = font.render("CAUGHT - game over", True, RED)
            screen.blit(msg, msg.get_rect(center=(w//2, h//2)))
        
        pygame.display.flip()
    
    pygame.quit()
    
    if not result.caught and goal_counter < max_goals:
        result.goals_reached = goal_counter
        result.time_elapsed = current_time if 'current_time' in locals() else max_time
    
    return result.to_dict()


#######################################################################
# MULTIPLE TRIAL WRAPPER
#######################################################################

def run_multiple_switching_trials(num_trials: int = 15, max_goals: int = 10, max_time: float = 30.0):
    print("\nRunning Switching Filter Trials...")
    print("==================================================")
    
    all_results = []
    
    for trial in range(num_trials):
        seed = trial
        print(f"\n=== Trial {trial+1}/{num_trials} ===")
        
        result = run_switching_filter(
            seed=seed,
            max_goals=max_goals,
            max_time=max_time
        )
        
        all_results.append(result)
        
        print(f"  Goals: {result['goals_reached']}/{max_goals}")
        print(f"  Status: {'Caught' if result['caught'] else 'Completed'}")
        print(f"  Time: {result['time_elapsed']:.2f}s")
    
    # Save results
    filename = f"switching_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({"trials": all_results}, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return all_results


#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":
    run_multiple_switching_trials(num_trials=15)
