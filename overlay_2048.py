import copy
import random
import time
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from mss import mss

# ==========================================
# CALIBRATION: Adjust these box coordinates
# to match your browser's 2048 grid location on screen.
# Format: {"top": y, "left": x, "width": w, "height": h}
# ==========================================
MONITOR_BOX = {"top": 200, "left": 600, "width": 500, "height": 500}

# Standard 2048 Tile RGB Colors for color-based classification
TILE_COLORS = {
    (238, 228, 218): 2,  # 2
    (237, 224, 200): 4,  # 4
    (242, 177, 121): 8,  # 8
    (245, 149, 99): 16,  # 16
    (246, 124, 95): 32,  # 32
    (246, 94, 59): 64,  # 64
    (237, 207, 114): 128,  # 128
    (237, 204, 97): 256,  # 256
    (237, 200, 80): 512,  # 512
    (237, 197, 63): 1024,  # 1024
    (237, 194, 46): 2048,  # 2048
    (60, 58, 50): 4096,  # 4096+ (Deep game states)
    (205, 193, 180): 0,  # Empty cell background
}


def get_closest_tile_value(rgb_color):
  """Finds the closest matching 2048 tile value based on RGB Euclidean distance."""
  min_dist = float("inf")
  closest_val = 0
  for col, val in TILE_COLORS.items():
    dist = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_color, col))
    if dist < min_dist:
      min_dist = dist
      closest_val = val
  return closest_val


class Game2048Engine:
  """Expectimax AI engine to calculate optimal moves."""

  def __init__(self, depth=3):
    self.depth = depth

  def get_available_moves(self, board):
    moves = []
    for action in range(4):
      if self._is_move_valid(board, action):
        moves.append(action)
    return moves

  def _is_move_valid(self, board, action):
    rotated_board = np.rot90(board, action)
    for r in range(4):
      row = rotated_board[r]
      for c in range(3):
        if (row[c] == 0 and row[c + 1] != 0) or (
            row[c] != 0 and row[c] == row[c + 1]
        ):
          return True
    return False

  def step_board(self, board, action):
    board_rot = np.rot90(board, action)
    new_board = np.zeros((4, 4), dtype=int)
    for r in range(4):
      row = board_rot[r][board_rot[r] != 0]
      new_row = []
      skip = False
      for i in range(len(row)):
        if skip:
          skip = False
          continue
        if i + 1 < len(row) and row[i] == row[i + 1]:
          new_row.append(row[i] * 2)
          skip = True
        else:
          new_row.append(row[i])
      new_row.extend([0] * (4 - len(new_row)))
      new_board[r] = new_row
    return np.rot90(new_board, -action)

  def evaluate(self, board):
    empty_cells = len(np.where(board == 0)[0])
    if empty_cells == 0:
      return -float("inf")
    # Heuristics: Empty tiles + Monotonicity + Corner Bias
    return (empty_cells * 2.7) + self.calculate_monotonicity(board)

  def calculate_monotonicity(self, board):
    totals = [0, 0, 0, 0]
    for r in range(4):
      for c in range(3):
        curr = board[r, c] if board[r, c] == 0 else np.log2(board[r, c])
        nxt = board[r, c + 1] if board[r, c + 1] == 0 else np.log2(board[r, c + 1])
        if curr > nxt:
          totals[0] += nxt - curr
        else:
          totals[1] += curr - nxt
    return max(totals[0], totals[1])

  def get_best_move(self, board):
    best_score = -float("inf")
    best_action = None
    moves = self.get_available_moves(board)
    if not moves:
      return None

    for action in moves:
      next_board = self.step_board(board, action)
      score = self.expectimax(next_board, self.depth - 1, False)
      if score > best_score:
        best_score = score
        best_action = action
    return best_action

  def expectimax(self, board, depth, is_player_turn):
    if depth == 0 or not self.get_available_moves(board):
      return self.evaluate(board)

    if is_player_turn:
      max_score = -float("inf")
      for action in self.get_available_moves(board):
        next_board = self.step_board(board, action)
        score = self.expectimax(next_board, depth - 1, False)
        max_score = max(max_score, score)
      return max_score
    else:
      empty_cells = list(zip(*np.where(board == 0)))
      if not empty_cells:
        return self.evaluate(board)
      total_score = 0
      cells_to_check = (
          empty_cells
          if len(empty_cells) <= 4
          else random.sample(empty_cells, 4)
      )
      weight = 1.0 / len(cells_to_check)
      for r, c in cells_to_check:
        board[r, c] = 2
        s2 = self.expectimax(board, depth - 1, True)
        board[r, c] = 4
        s4 = self.expectimax(board, depth - 1, True)
        board[r, c] = 0
        total_score += weight * (0.9 * s2 + 0.1 * s4)
      return total_score


class AnalysisHUD:
  """Floating Tkinter Transparent Overlay Window."""

  def __init__(self):
    self.root = tk.Tk()
    self.root.title("2048 Analysis Engine")
    self.root.geometry("180x180+100+100")
    self.root.attributes("-topmost", True)
    self.root.attributes("-alpha", 0.85)
    self.root.configure(bg="#1e1e1e")

    self.label_title = tk.Label(
        self.root,
        text="BEST MOVE",
        font=("Helvetica", 12, "bold"),
        fg="#aaaaaa",
        bg="#1e1e1e",
    )
    self.label_title.pack(pady=5)

    self.label_arrow = tk.Label(
        self.root,
        text="---",
        font=("Helvetica", 48, "bold"),
        fg="#00ffcc",
        bg="#1e1e1e",
    )
    self.label_arrow.pack(pady=5)

    self.label_status = tk.Label(
        self.root,
        text="Scanning...",
        font=("Helvetica", 10),
        fg="#ffffff",
        bg="#1e1e1e",
    )
    self.label_status.pack(pady=5)

    self.sct = mss()
    self.engine = Game2048Engine(depth=3)

    # Start update loop
    self.update_analysis()
    self.root.mainloop()

  def parse_board_from_screen(self):
    # Capture screen region
    img = np.array(self.sct.grab(MONITOR_BOX))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    h, w, _ = img_rgb.shape
    cell_h, cell_w = h // 4, w // 4
    board = np.zeros((4, 4), dtype=int)

    for r in range(4):
      for c in range(4):
        # Sample the center pixel of each grid cell
        center_y = int((r + 0.5) * cell_h)
        center_x = int((c + 0.5) * cell_w)
        pixel_color = img_rgb[center_y, center_x]
        board[r, c] = get_closest_tile_value(pixel_color)

    return board

  def update_analysis(self):
    try:
      board = self.parse_board_from_screen()
      best_move = self.engine.get_best_move(board)

      arrows = {0: "⬆️ UP", 1: "➡️ RIGHT", 2: "⬇️ DOWN", 3: "⬅️ LEFT"}

      if best_move is not None:
        self.label_arrow.config(text=arrows[best_move])
        max_tile = np.max(board)
        self.label_status.config(text=f"Max Tile: {max_tile}")
      else:
        self.label_arrow.config(text="💀")
        self.label_status.config(text="Game Over")
    except Exception as e:
      self.label_status.config(text="Error / Check Box")

    # Refresh every 300ms
    self.root.after(300, self.update_analysis)


if __name__ == "__main__":
  AnalysisHUD()
