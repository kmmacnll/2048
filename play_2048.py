import copy
import random
import time
import numpy as np


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
        return [
            action for action in range(4) if self._is_move_valid(action)
        ]

    def _is_move_valid(self, action):
        rotated_board = np.rot90(self.board, action)
        for r in range(4):
            row = rotated_board[r]
            for c in range(3):
                if (row[c] == 0 and row[c + 1] != 0) or (
                    row[c] != 0 and row[c] == row[c + 1]
                ):
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
                if i + 1 < len(row) and row[i] == row[i + 1]:
                    merged_val = row[i] * 2
                    new_row.append(merged_val)
                    earned_score += merged_val
                    skip = True
                else:
                    new_row.append(row[i])
            new_row.extend([0] * (4 - len(new_row)))
            new_board[r] = new_row

        self.board = np.rot90(new_board, -action)
        self.score += earned_score
        return self.board, earned_score


class AIHeuristicAgent:

    def __init__(self, depth=3):
        self.depth = depth

    def evaluate(self, board):
        empty_cells = len(np.where(board == 0)[0])
        if empty_cells == 0:
            return -float("inf")

        w_empty = 2.7
        w_smoothness = 0.1
        w_monotonicity = 1.0
        w_corner = 1.0

        smoothness = self.calculate_smoothness(board)
        monotonicity = self.calculate_monotonicity(board)
        corner_bonus = self.evaluate_corner(board)

        return (
            (empty_cells * w_empty)
            + (monotonicity * w_monotonicity)
            - (smoothness * w_smoothness)
            + (corner_bonus * w_corner)
        )

    def calculate_smoothness(self, board):
        smoothness = 0
        for r in range(4):
            for c in range(4):
                if board[r, c] != 0:
                    val = np.log2(board[r, c])
                    if r < 3 and board[r + 1, c] != 0:
                        smoothness += abs(val - np.log2(board[r + 1, c]))
                    if c < 3 and board[r, c + 1] != 0:
                        smoothness += abs(val - np.log2(board[r, c + 1]))
        return smoothness

    def calculate_monotonicity(self, board):
        totals = [0, 0, 0, 0]
        for r in range(4):
            for c in range(3):
                current = (
                    board[r, c] if board[r, c] == 0 else np.log2(board[r, c])
                )
                next_val = (
                    board[r, c + 1]
                    if board[r, c + 1] == 0
                    else np.log2(board[r, c + 1])
                )
                if current > next_val:
                    totals[0] += next_val - current
                else:
                    totals[1] += current - next_val
        for c in range(4):
            for r in range(3):
                current = (
                    board[r, c] if board[r, c] == 0 else np.log2(board[r, c])
                )
                next_val = (
                    board[r + 1, c]
                    if board[r + 1, c] == 0
                    else np.log2(board[r + 1, c])
                )
                if current > next_val:
                    totals[2] += next_val - current
                else:
                    totals[3] += current - next_val
        return max(totals[0], totals[1]) + max(totals[2], totals[3])

    def evaluate_corner(self, board):
        max_val = np.max(board)
        if board[3, 3] == max_val:
            return np.log2(max_val) * 2.0
        return 0

    def get_best_move(self, env):
        best_score = -float("inf")
        best_action = None
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
            max_score = -float("inf")
            for action in env.get_available_moves():
                sim_env = copy.deepcopy(env)
                sim_env.step(action)
                score = self.expectimax(sim_env, depth - 1, False)
                max_score = max(max_score, score)
            return max_score
        else:
            empty_cells = list(zip(*np.where(env.board == 0)))
            if not empty_cells:
                return self.evaluate(env.board)

            total_score = 0
            cells_to_check = (
                empty_cells
                if len(empty_cells) <= 5
                else random.sample(empty_cells, 5)
            )
            weight = 1.0 / len(cells_to_check)

            for r, c in cells_to_check:
                env.board[r, c] = 2
                score_2 = self.expectimax(env, depth - 1, True)

                env.board[r, c] = 4
                score_4 = self.expectimax(env, depth - 1, True)

                env.board[r, c] = 0
                total_score += weight * (0.9 * score_2 + 0.1 * score_4)

            return total_score


if __name__ == "__main__":
    start_time = time.time()
    env = Game2048()
    agent = AIHeuristicAgent(depth=3)

    steps = 0
    while True:
        moves = env.get_available_moves()
        if not moves:
            break

        action = agent.get_best_move(env)
        if action is None:
            break

        env.step(action)
        env.add_tile()  # Spawn new tile after valid move
        steps += 1

        if steps % 100 == 0:
            print(
                f"Steps: {steps} | Max Tile: {np.max(env.board)} | Score: {env.score}",
                flush=True,
            )

    elapsed = round(time.time() - start_time, 2)
    summary = (
        f"### 2048 AI Run Results\n"
        f"- **Final Max Tile:** {np.max(env.board)}\n"
        f"- **Final Score:** {env.score}\n"
        f"- **Total Moves:** {steps}\n"
        f"- **Execution Time:** {elapsed}s\n\n"
        f"```\n{env.board}\n```"
    )
    print("\n" + summary, flush=True)

    # Write results to GitHub Actions Job Summary if running in CI
    import os

    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)
