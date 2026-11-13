import numpy as np
import random
import copy

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_tile()
        self.add_tile()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_tile()
        self.add_tile()
        return self.board

    def add_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            # 90% chance for 2, 10% chance for 4
            self.board[row, col] = 4 if random.random() < 0.1 else 2

    def get_available_moves(self):
        moves = []
        for action in range(4): # 0: Up, 1: Right, 2: Down, 3: Left
            if self._is_move_valid(action):
                moves.append(action)
        return moves

    def _is_move_valid(self, action):
        rotated_board = np.rot90(self.board, action)
        for r in range(4):
            row = rotated_board[r]
            for c in range(3):
                if row[c] == 0 and row[c+1] != 0:
                    return True
                if row[c] != 0 and row[c] == row[c+1]:
                    return True
        return False

    def step(self, action):
        # Rotate board to standardize execution (always slide left)
        board_rot = np.rot90(self.board, action)
        new_board = np.zeros((4, 4), dtype=int)
        earned_score = 0

        for r in range(4):
            row = board_rot[r][board_rot[r] != 0] # Remove zeros
            new_row = []
            skip = False
            for i in range(len(row)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(row) and row[i] == row[i+1]:
                    merged_val = row[i] * 2
                    new_row.append(merged_val)
                    earned_score += merged_val
                    skip = True
                else:
                    new_row.append(row[i])
            
            # Pad with zeros
            new_row.extend([0] * (4 - len(new_row)))
            new_board[r] = new_row

        # Rotate back
        self.board = np.rot90(new_board, -action)
        self.score += earned_score
        return self.board, earned_score


class AIHeuristicAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def evaluate(self, board):
        """
        Evaluates the strategic quality of a board state based on human master heuristics.
        """
        empty_cells = len(np.where(board == 0)[0])
        if empty_cells == 0:
            return -float('inf')

        # Weights for different heuristic metrics
        w_empty = 2.7
        w_smoothness = 0.1
        w_monotonicity = 1.0
        w_corner = 1.0

        smoothness = self.calculate_smoothness(board)
        monotonicity = self.calculate_monotonicity(board)
        corner_bonus = self.evaluate_corner(board)

        # Log scaling for empty tiles and bonuses makes the evaluation more stable
        return (empty_cells * w_empty) + \
               (monotonicity * w_monotonicity) - \
               (smoothness * w_smoothness) + \
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
        totals = [0, 0, 0, 0] # Up, Down, Left, Right
        # Check rows
        for r in range(4):
            for c in range(3):
                current = board[r, c] if board[r, c] == 0 else np.log2(board[r, c])
                next_val = board[r, c+1] if board[r, c+1] == 0 else np.log2(board[r, c+1])
                if current > next_val:
                    totals[0] += next_val - current
                else:
                    totals[1] += current - next_val
        # Check cols
        for c in range(4):
            for r in range(3):
                current = board[r, c] if board[r, c] == 0 else np.log2(board[r, c])
                next_val = board[r+1, c] if board[r+1, c] == 0 else np.log2(board[r+1, c])
                if current > next_val:
                    totals[2] += next_val - current
                else:
                    totals[3] += current - next_val
        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def evaluate_corner(self, board):
        # Heavily reward keeping the absolute highest tile in the bottom-right corner
        max_val = np.max(board)
        if board[3, 3] == max_val:
            return np.log2(max_val) * 2.0
        return 0

    def get_best_move(self, env):
        best_score = -float('inf')
        best_action = None
        
        # Simulate environment state copy
        sim_env = copy.deepcopy(env)
        moves = sim_env.get_available_moves()
        
        if not moves:
            return None

        for action in moves:
            sim_env.board = np.copy(env.board)
            sim_env.step(action)
            score = self.expectimax(sim_env, self.depth - 1, False)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def expectimax(self, env, depth, is_player_turn):
        if depth == 0 or not env.get_available_moves():
            return self.evaluate(env.board)

        if is_player_turn:
            max_score = -float('inf')
            for action in env.get_available_moves():
                sim_env = copy.deepcopy(env)
                sim_env.step(action)
                score = self.expectimax(sim_env, depth - 1, False)
                max_score = max(max_score, score)
            return max_score
        else:
            # Chance node: Evaluates all empty slots where '2' (90%) or '4' (10%) could spawn
            empty_cells = list(zip(*np.where(env.board == 0)))
            if not empty_cells:
                return self.evaluate(env.board)
            
            total_score = 0
            # To optimize performance on large empty spaces, sample subset if empty > 5
            cells_to_check = empty_cells if len(empty_cells) <= 5 else random.sample(empty_cells, 5)
            weight = 1.0 / len(cells_to_check)

            for r, c in cells_to_check:
                # Outcome 1: Spawn '2' (Probability 0.9)
                env.board[r, c] = 2
                score_2 = self.expectimax(env, depth - 1, True)
                
                # Outcome 2: Spawn '4' (Probability 0.1)
                env.board[r, c] = 4
                score_4 = self.expectimax(env, depth - 1, True)
                
                env.board[r, c] = 0 # Revert
                
                total_score += weight * (0.9 * score_2 + 0.1 * score_4)
                
            return total_score

# --- Execution Loop ---
if __name__ == "__main__":
    env = Game2048()
    agent = AIHeuristicAgent(depth=3) # Depth 3-4 provides optimal speed-to-performance ratio
    
    steps = 0
    while True:
        moves = env.get_available_moves()
        if not moves:
            print(f"Game Over! Final Max Tile: {np.max(env.board)}, Score: {env.score}")
            break
            
        action = agent.get_best_move(env)
        if action is None:
            break
        env.step(action)
        steps += 1
        
        # Print progress every 200 moves
        if steps % 200 == 0:
            print(f"Steps: {steps} | Max Tile: {np.max(env.board)} | Score: {env.score}")
