import pygame
import math
import random
import time
import json
import datetime
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, List

class SimulationResult:
    def __init__(self):
        self.goals_reached: int = 0
        self.caught: bool = False
        self.time_elapsed: float = 0.0
        self.distance_trace: List[float] = []
        self.h_trace: List[float] = []
        self.goal_times: List[float] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'goals_reached': self.goals_reached,
            'caught': self.caught,
            'time_elapsed': self.time_elapsed,
            'distance_trace': self.distance_trace,
            'h_trace': self.h_trace,
            'goal_times': self.goal_times
        }

#######################################################################
# RUN CBF SIMULATION
#######################################################################

def run_cbf_simulation(seed: int = None, max_goals: int = 10, max_time: float = 30.0) -> Dict[str, Any]:
    
    if seed is not None:
        random.seed(seed)
    
    result = SimulationResult()
    
    w, h = 800, 600
    
    
    FPS = 120
    evader_speed = 6.0
    pursuer_speed = 3.0
    
    
    CBF_ALPHA = 0.5 
    
    
    evader_radius, pursuer_radius, goal_radius = 15, 15, 20
    MIN_GOAL_DIST_FROM_PURSUER = 100  
    
    
    COLLISION_RADIUS = evader_radius + pursuer_radius  
    
    
    CBF_SAFETY_RADIUS = 50.0  
    
    WHITE = (255,255,255)
    BLUE = (50,150,255)
    RED = (255,80,80)
    GREEN = (50,200,50)
    BLACK = (0,0,0)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("True CBF Implementation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    
    
    evader_x = w//4
    evader_y = h//2
    pursuer_x = 3*w//4
    pursuer_y = h//2
    pursuer_vx = 0.0  
    pursuer_vy = 0.0
    goal_counter = 0
    goal_times = []
    
    
    def spawn_goal_away_from_pursuer(pux, puy):
        max_attempts = 100
        for _ in range(max_attempts):
            gx = random.randint(0, w)
            gy = random.randint(0, h)
            dist = math.hypot(gx - pux, gy - puy)
            if dist >= MIN_GOAL_DIST_FROM_PURSUER:
                return gx, gy
        return random.randint(0, w), random.randint(0, h)
    
    
    goal_x, goal_y = spawn_goal_away_from_pursuer(pursuer_x, pursuer_y)
    goal_spawn_time = pygame.time.get_ticks() / 1000.0
    
    
    running = True
    game_over = False
    frame_count = 0
    cbf_is_active = False  
    
    start_time = pygame.time.get_ticks() / 1000.0
    
    
    def get_barrier_value(evx, evy, pux, puy, safety_radius=None):
        if safety_radius is None:
            safety_radius = CBF_SAFETY_RADIUS
        rx = evx - pux #distance between evader and pursuer
        ry = evy - puy #distance between evader and pursuer
        dist_sq = rx*rx + ry*ry #distance squared
        return dist_sq - safety_radius**2 #barrier function value
    
    def cbf_qp(evx, evy, pux, puy, pvx, pvy, desired_vx, desired_vy,
               safety_radius=None, alpha=None, max_speed=None):
            
        if safety_radius is None:
            safety_radius = CBF_SAFETY_RADIUS
        if alpha is None:
            alpha = CBF_ALPHA
        if max_speed is None:
            max_speed = evader_speed
        
        rx = evx - pux
        ry = evy - puy
        dist_sq = rx*rx + ry*ry
        dist = math.sqrt(dist_sq) if dist_sq > 1e-6 else 1e-6
        
        h = dist_sq - safety_radius**2
        
        grad_h_x = 2 * rx
        grad_h_y = 2 * ry
        
        rel_vx_des = desired_vx - pvx
        rel_vy_des = desired_vy - pvy
        h_dot_desired = grad_h_x * rel_vx_des + grad_h_y * rel_vy_des
        desired_norm = math.hypot(desired_vx, desired_vy)
        
        if desired_norm <= max_speed and h_dot_desired >= -alpha * h - 1e-6:
            return desired_vx, desired_vy, False
        
        rhs = -alpha * h + grad_h_x * pvx + grad_h_y * pvy
        
        def objective(v):
            return (v[0] - desired_vx)**2 + (v[1] - desired_vy)**2
        
        constraints = [{
            'type': 'ineq',
            'fun': lambda v: grad_h_x * v[0] + grad_h_y * v[1] - rhs
        }]
        
        if max_speed < float('inf'):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v: max_speed**2 - (v[0]**2 + v[1]**2)
            })
        
        if desired_norm > max_speed:
            v0 = [desired_vx * max_speed / desired_norm, desired_vy * max_speed / desired_norm]
        else:
            v0 = [desired_vx, desired_vy]
        
        result_qp = minimize(objective, v0, method='SLSQP', constraints=constraints,
                         options={'ftol': 1e-9, 'maxiter': 100, 'disp': False})
        
        if result_qp.success:
            safe_vx, safe_vy = result_qp.x[0], result_qp.x[1]
            rel_vx = safe_vx - pvx
            rel_vy = safe_vy - pvy
            h_dot = grad_h_x * rel_vx + grad_h_y * rel_vy
            if h_dot >= -alpha * h - 1e-5:
                return safe_vx, safe_vy, True
        
        grad_norm_sq = grad_h_x**2 + grad_h_y**2
        if grad_norm_sq > 1e-6:
            lambda_val = (rhs - grad_h_x * desired_vx - grad_h_y * desired_vy) / grad_norm_sq
            lambda_val = max(0.0, lambda_val)  
            safe_vx = desired_vx + lambda_val * grad_h_x
            safe_vy = desired_vy + lambda_val * grad_h_y
            safe_norm = math.hypot(safe_vx, safe_vy)
            if safe_norm > max_speed:
                safe_vx = safe_vx * max_speed / safe_norm
                safe_vy = safe_vy * max_speed / safe_norm
        else:
            nx, ny = rx / dist, ry / dist
            safe_vx, safe_vy = max_speed * nx, max_speed * ny
        
        return safe_vx, safe_vy, True
    
    print("\n=== Running CBF Simulation ===")
    
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
            
            h_value_current = get_barrier_value(evader_x, evader_y, pursuer_x, pursuer_y)
            result.h_trace.append(h_value_current)
            
            if current_time >= max_time:
                result.goals_reached = goal_counter
                result.time_elapsed = current_time
                game_over = True
                print(f"Time's up at {max_time}s.")
                break
            
            gdx = goal_x - evader_x
            gdy = goal_y - evader_y
            gdist = math.hypot(gdx, gdy)
            
            if gdist > 1e-6:
                desired_vx = (gdx / gdist) * evader_speed
                desired_vy = (gdy / gdist) * evader_speed
            else:
                desired_vx = desired_vy = 0.0
            
            safe_vx, safe_vy, cbf_is_active = cbf_qp(
                evader_x, evader_y, pursuer_x, pursuer_y,
                pursuer_vx, pursuer_vy,
                desired_vx, desired_vy
            )
            
            h_value = get_barrier_value(evader_x, evader_y, pursuer_x, pursuer_y)
            if h_value < 0 or cbf_is_active:
                mode = "SAFETY"
            else:
                mode = "GOAL"
            
            evader_x += safe_vx
            evader_y += safe_vy
            
            pdx = evader_x - pursuer_x 
            pdy = evader_y - pursuer_y
            pdist = math.hypot(pdx, pdy) 
            if pdist > 1e-6: 
                pursuer_vx = (pdx / pdist) * pursuer_speed
                pursuer_vy = (pdy / pdist) * pursuer_speed
                pursuer_x += pursuer_vx
                pursuer_y += pursuer_vy
            else:
                pursuer_vx, pursuer_vy = 0.0, 0.0
            pursuer_x = max(pursuer_radius, min(w - pursuer_radius, pursuer_x)) 
            pursuer_y = max(pursuer_radius, min(h - pursuer_radius, pursuer_y))
            evader_x = max(evader_radius, min(w - evader_radius, evader_x)) 
            evader_y = max(evader_radius, min(h - evader_radius, evader_y))
            
            dist_sq = (evader_x - pursuer_x)**2 + (evader_y - pursuer_y)**2
            if dist_sq < COLLISION_RADIUS**2:  
                result.caught = True
                result.time_elapsed = current_time
                game_over = True
                running = False
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
                    running = False
                    print("All goals collected!")
                else:
                    goal_x, goal_y = spawn_goal_away_from_pursuer(pursuer_x, pursuer_y) 
                    goal_spawn_time = pygame.time.get_ticks() / 1000.0 
        
        pygame.draw.circle(screen, (200, 200, 255), (int(pursuer_x), int(pursuer_y)), int(CBF_SAFETY_RADIUS), 2)
        pygame.draw.circle(screen, (255, 100, 100), (int(pursuer_x), int(pursuer_y)), int(COLLISION_RADIUS), 1)
        
        pygame.draw.circle(screen, RED, (int(pursuer_x), int(pursuer_y)), pursuer_radius)
        pygame.draw.circle(screen, BLUE, (int(evader_x), int(evader_y)), evader_radius)
        if goal_counter < max_goals:
            pygame.draw.circle(screen, GREEN, (int(goal_x), int(goal_y)), goal_radius)
        pygame.draw.line(screen, BLACK, (int(evader_x), int(evader_y)), (int(pursuer_x), int(pursuer_y)), 1)
        
        # draw the text
        if not game_over:
            screen.blit(font.render(f"Goals: {goal_counter}/{max_goals}", True, GREEN), (10, 8))
            mode_text = f"Mode: {mode}"
            if cbf_is_active:
                mode_text += " [CBF ACTIVE]"
            screen.blit(font.render(mode_text, True, RED if mode!="GOAL" else GREEN), (10, 32))
            screen.blit(small_font.render(f"CBF Safety R: {CBF_SAFETY_RADIUS:.1f}", True, (100, 100, 200)), (10, 56))
            screen.blit(small_font.render(f"Collision R: {COLLISION_RADIUS:.1f}", True, (200, 100, 100)), (10, 76))
            h_val = get_barrier_value(evader_x, evader_y, pursuer_x, pursuer_y)
            h_color = GREEN if h_val >= 0 else RED
            screen.blit(small_font.render(f"h(x): {h_val:.1f}", True, h_color), (10, 96))
            screen.blit(small_font.render(f"Time: {current_time:.1f}s", True, BLACK), (10, 116))
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

def run_multiple_cbf_trials(num_trials: int = 15, max_goals: int = 10, max_time: float = 30.0):
    print("\nRunning CBF Trials...")
    print("==================================================")
    
    all_results = []
    
    for trial in range(num_trials):
        seed = trial
        print(f"\n=== Trial {trial+1}/{num_trials} ===")
        
        result = run_cbf_simulation(
            seed=seed,
            max_goals=max_goals,
            max_time=max_time
        )
        
        all_results.append(result)
        
        print(f"  Goals: {result['goals_reached']}/{max_goals}")
        print(f"  Status: {'Caught' if result['caught'] else 'Completed'}")
        print(f"  Time: {result['time_elapsed']:.2f}s")
    
    # Save results
    filename = f"cbf_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({"trials": all_results}, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return all_results


#######################################################################
# MAIN
#######################################################################

if __name__ == "__main__":
    run_multiple_cbf_trials(num_trials=15)
