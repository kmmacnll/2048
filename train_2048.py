import copy
import random
import time
import multiprocessing
import numpy as np

# --- Game Engine ---
class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_tile()
        self.add_tile()

    def add_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 4 if random.random() < 0.1 else 2

    def get_available_moves(self):
        return [action for action in range(4) if self._is_move_valid(action)]

    def _is_move_valid(self, action):
        rotated_board = np.rot90(self.board, action)
        for r in range(4):
            row = rotated_board[r]
            for c in range(3):
                if (row[c] == 0 and row[c+1] != 0) or (row[c] != 0 and row[c] == row[c+1]):
                    return True
        return False

    def step(self, action):
        board_rot = np.rot90(self.board, action)
        new_board = np.zeros((4, 4), dtype=int)
        earned_score = 0

        for r in range(4):
            row = board_rot[r][board_rot[r] != 0]
            new_row = []
            skip = False
            for i in range(len(row)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(row) and row[i] == row[i+1]:
                    val = row[i] * 2
                    new_row.append(val)
                    earned_score += val
                    skip = True
                else:
                    new_row.append(row[i])
            new_row.extend([0] * (4 - len(new_row)))
            new_board[r] = new_row

        self.board = np.rot90(new_board, -action)
        self.score += earned_score
        return self.score


# --- Parametric Heuristic Agent ---
class TunableAgent:
    def __init__(self, weights, depth=3):
        # weights = [w_empty, w_smoothness, w_monotonicity, w_corner]
        self.weights = weights
        self.depth = depth

    def evaluate(self, board):
        empty_cells = len(np.where(board == 0)[0])
        if empty_cells == 0:
            return -float('inf')

        w_empty, w_smooth, w_mono, w_corner = self.weights

        smoothness = self.calculate_smoothness(board)
        monotonicity = self.calculate_monotonicity(board)
        corner_bonus = self.evaluate_corner(board)

        return (empty_cells * w_empty) + \
               (monotonicity * w_mono) - \
               (smoothness * w_smooth) + \
               (corner_bonus * w_corner)

    def calculate_smoothness(self, board):
        smoothness = 0
        for r in range(4):
            for c in range(4):
                if board[r, c] != 0:
                    val = np.log2(board[r, c])
                    if r < 3 and board[r+1, c] != 0:
                        smoothness += abs(val - np.log2(board[r+1, c]))
                    if c < 3 and board[r, c+1] != 0:
                        smoothness += abs(val - np.log2(board[r, c+1]))
        return smoothness

    def calculate_monotonicity(self, board):
        totals = [0, 0, 0, 0]
        for r in range(4):
            for c in range(3):
                curr = board[r, c] if board[r, c] == 0 else np.log2(board[r, c])
                nxt = board[r, c+1] if board[r, c+1] == 0 else np.log2(board[r, c+1])
                if curr > nxt: totals[0] += nxt - curr
                else: totals[1] += curr - nxt
        for c in range(4):
            for r in range(3):
                curr = board[r, c] if board[r, c] == 0 else np.log2(board[r, c])
                nxt = board[r+1, c] if board[r+1, c] == 0 else np.log2(board[r+1, c])
                if curr > nxt: totals[2] += nxt - curr
                else: totals[3] += curr - nxt
        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def evaluate_corner(self, board):
        max_val = np.max(board)
        if board[3, 3] == max_val:
            return np.log2(max_val) * 2.0
        return 0

    def get_best_move(self, board):
        best_score = -float('inf')
        best_action = None
        
        # Fast local move scanner
        for action in range(4):
            if self._is_move_valid(board, action):
                next_board = self._simulate_step(board, action)
                score = self.expectimax(next_board, self.depth - 1, False)
                if score > best_score:
                    best_score = score
                    best_action = action
        return best_action

    def _is_move_valid(self, board, action):
        rb = np.rot90(board, action)
        for r in range(4):
            row = rb[r]
            for c in range(3):
                if (row[c] == 0 and row[c+1] != 0) or (row[c] != 0 and row[c] == row[c+1]):
                    return True
        return False

    def _simulate_step(self, board, action):
        rb = np.rot90(board, action)
        nb = np.zeros((4, 4), dtype=int)
        for r in range(4):
            row = rb[r][rb[r] != 0]
            new_row = []
            skip = False
            for i in range(len(row)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(row) and row[i] == row[i+1]:
                    new_row.append(row[i] * 2)
                    skip = True
                else:
                    new_row.append(row[i])
            new_row.extend([0] * (4 - len(new_row)))
            nb[r] = new_row
        return np.rot90(nb, -action)

    def expectimax(self, board, depth, is_player_turn):
        if depth == 0:
            return self.evaluate(board)

        if is_player_turn:
            max_score = -float('inf')
            valid = False
            for action in range(4):
                if self._is_move_valid(board, action):
                    valid = True
                    nb = self._simulate_step(board, action)
                    score = self.expectimax(nb, depth - 1, False)
                    max_score = max(max_score, score)
            return max_score if valid else self.evaluate(board)
        else:
            empty_cells = list(zip(*np.where(board == 0)))
            if not empty_cells:
                return self.evaluate(board)
            
            # Sample empty spaces to optimize speed
            cells = empty_cells if len(empty_cells) <= 3 else random.sample(empty_cells, 3)
            weight = 1.0 / len(cells)
            total = 0
            for r, c in cells:
                board[r, c] = 2
                s2 = self.expectimax(board, depth - 1, True)
                board[r, c] = 4
                s4 = self.expectimax(board, depth - 1, True)
                board[r, c] = 0
                total += weight * (0.9 * s2 + 0.1 * s4)
            return total


# --- Single Game Worker ---
def run_single_game(weights):
    env = Game2048()
    agent = TunableAgent(weights, depth=2) # Depth 2 ensures fast multi-game runs
    
    while True:
        moves = env.get_available_moves()
        if not moves:
            break
        action = agent.get_best_move(env.board)
        if action is None:
            break
        env.step(action)
        env.add_tile()
        
    return env.score, np.max(env.board)


# --- Distributed Simulation Runner ---
def evaluate_weight_set(args):
    weights, num_games = args
    scores = []
    max_tiles = []
    
    for _ in range(num_games):
        score, max_tile = run_single_game(weights)
        scores.append(score)
        max_tiles.append(max_tile)
        
    # Fitness Function: Weighted average of score and a massive bonus for reaching 2048/4096
    avg_score = np.mean(scores)
    win_rate_2048 = sum(1 for t in max_tiles if t >= 2048) / num_games
    fitness = avg_score + (win_rate_2048 * 50000)
    return fitness, avg_score, max(max_tiles)


if __name__ == "__main__":
    NUM_GENERATIONS = 5
    POPULATION_SIZE = 8
    GAMES_PER_EVAL = 20 # Increase to 100+ for rigorous training
    
    # Initialize random population of weights [w_empty, w_smooth, w_mono, w_corner]
    population = [[random.uniform(1.0, 5.0), random.uniform(0.0, 1.0), random.uniform(0.5, 3.0), random.uniform(0.5, 3.0)] for _ in range(POPULATION_SIZE)]
    
    print(f"--- Starting 2048 Evolutionary Training Pipeline ---", flush=True)
    
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    for gen in range(NUM_GENERATIONS):
        print(f"\n[Generation {gen + 1}/{NUM_GENERATIONS}] Evaluating {POPULATION_SIZE} weight variants...", flush=True)
        start_time = time.time()
        
        tasks = [(weights, GAMES_PER_EVAL) for weights in population]
        results = pool.map(evaluate_weight_set, tasks)
        
        # Sort population by fitness
        ranked_population = sorted(zip(population, results), key=lambda x: x[1][0], reverse=True)
        
        best_weights, (best_fitness, best_avg_score, best_max_tile) = ranked_population[0]
        print(f"Best Fitness: {best_fitness:.1f} | Avg Score: {best_avg_score:.1f} | Best Max Tile Achieved: {best_max_tile}", flush=True)
        print(f"Optimal Weights Found: {best_weights}", flush=True)
        
        # Breed next generation (Elitism + Mutation)
        top_performers = [item[0] for item in ranked_population[:2]]
        new_population = list(top_performers) # Keep top 2 elite
        
        while len(new_population) < POPULATION_SIZE:
            parent = random.choice(top_performers)
            # Mutate weights slightly
            child = [w + random.uniform(-0.3, 0.3) for w in parent]
            new_population.append(child)
            
        population = new_population
        print(f"Generation completed in {round(time.time() - start_time, 2)}s", flush=True)
        
    pool.close()
    pool.join()
