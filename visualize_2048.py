import copy
import random
import sys
import numpy as np
try:
  import pygame
except ImportError:
  print("Pygame is required! Run: pip install pygame numpy")
  sys.exit(1)

# --- Configuration & Colors ---
WIDTH, HEIGHT = 900, 650
FPS = 30  # Animation frame rate
NUM_CONCURRENT_GAMES = 12

# Authentic 2048 Color Palette
BG_COLOR = (250, 248, 239)
PANEL_BG = (238, 228, 218)
GRID_BG = (187, 173, 160)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)

TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (60, 58, 50),
}


# --- Game Engine ---
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


# --- Expectimax AI Agent ---
class ExpectimaxAgent:

  def __init__(self, depth=2):
    self.depth = depth

  def evaluate(self, board):
    empty_cells = len(np.where(board == 0)[0])
    if empty_cells == 0:
      return -float("inf")
    # Heuristic weights: Empty tiles + Monotonicity + Corner Bonus
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

  def get_best_move(self, env):
    best_score = -float("inf")
    best_action = None
    moves = env.get_available_moves()
    if not moves:
      return None

    for action in moves:
      sim_env = copy.deepcopy(env)
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
      cells = (
          empty_cells if len(empty_cells) <= 3 else random.sample(empty_cells, 3)
      )
      weight = 1.0 / len(cells)
      total_score = 0
      for r, c in cells:
        env.board[r, c] = 2
        s2 = self.expectimax(env, depth - 1, True)
        env.board[r, c] = 4
        s4 = self.expectimax(env, depth - 1, True)
        env.board[r, c] = 0
        total_score += weight * (0.9 * s2 + 0.1 * s4)
      return total_score


# --- Pygame Visualizer App ---
class SimulationVisualizer:

  def __init__(self):
    pygame.init()
    self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        "2048 AI Battle Arena - Longest Survivor View"
    )
    self.clock = pygame.time.Clock()
    self.font_large = pygame.font.SysFont("Helvetica", 28, bold=True)
    self.font_med = pygame.font.SysFont("Helvetica", 20, bold=True)
    self.font_small = pygame.font.SysFont("Helvetica", 16)

    # Initialize simulation pool
    self.agent = ExpectimaxAgent(depth=2)
    self.games = [Game2048() for _ in range(NUM_CONCURRENT_GAMES)]
    self.failure_counter = {
        64: 0,
        128: 0,
        256: 0,
        512: 0,
        1024: 0,
        2048: 0,
        4096: 0,
    }
    self.total_games_played = 0

  def run(self):
    running = True
    while running:
      # Event Handling
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False

      # Step all simulations forward one move per frame
      for i, game in enumerate(self.games):
        moves = game.get_available_moves()
        if not moves:
          # Game Over: Record failure max tile
          max_tile = int(np.max(game.board))
          if max_tile in self.failure_counter:
            self.failure_counter[max_tile] += 1
          else:
            self.failure_counter[max_tile] = (
                self.failure_counter.get(max_tile, 0) + 1
            )
          self.total_games_played += 1
          # Reset game to keep pool active
          game.reset()
        else:
          action = self.agent.get_best_move(game)
          if action is not None:
            game.step(action)
            game.add_tile()

      # Identify the "Hero" Game (Longest surviving / Highest max tile & score)
      hero_game = max(
          self.games, key=lambda g: (np.max(g.board), g.score)
      )

      # --- Render GUI ---
      self.screen.fill(BG_COLOR)
      self.draw_hero_board(hero_game)
      self.draw_hud(hero_game)

      pygame.display.flip()
      self.clock.tick(FPS)

    pygame.quit()
    sys.exit()

  def draw_hero_board(self, game):
    # Draw Main Title & Subtitle
    title = self.font_large.render("LONGEST SURVIVOR (HERO VIEW)", True, TEXT_DARK)
    self.screen.blit(title, (40, 30))

    # Draw 4x4 Grid Container
    board_rect = pygame.Rect(40, 90, 480, 480)
    pygame.draw.rect(self.screen, GRID_BG, board_rect, border_radius=8)

    cell_size = 105
    margin = 15
    for r in range(4):
      for c in range(4):
        val = game.board[r, c]
        color = TILE_COLORS.get(val, (60, 58, 50))
        x = board_rect.x + margin + c * (cell_size + margin)
        y = board_rect.y + margin + r * (cell_size + margin)
        cell_rect = pygame.Rect(x, y, cell_size, cell_size)
        pygame.draw.rect(self.screen, color, cell_rect, border_radius=6)

        if val > 0:
          text_color = TEXT_DARK if val <= 4 else TEXT_LIGHT
          # Dynamic font sizing based on digit length
          font_dyn = pygame.font.SysFont(
              "Helvetica", 32 if val < 1000 else 24, bold=True
          )
          txt = font_dyn.render(str(val), True, text_color)
          txt_rect = txt.get_rect(
              center=(x + cell_size // 2, y + cell_size // 2)
          )
          self.screen.blit(txt, txt_rect)

  def draw_hud(self, hero_game):
    # Draw Side Panel for Failure Counter / Graveyard Histogram
    panel_rect = pygame.Rect(560, 30, 300, 580)
    pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=10)

    title = self.font_med.render("GRAVEYARD COUNTER", True, TEXT_DARK)
    self.screen.blit(title, (585, 50))

    subtitle = self.font_small.render(
        f"Total Simulations Run: {self.total_games_played}", True, TEXT_DARK
    )
    self.screen.blit(subtitle, (585, 80))

    # Divider line
    pygame.draw.line(
        self.screen, GRID_BG, (585, 110), (835, 110), width=2
    )

    # Render Failure counts by max tile
    y_offset = 130
    header_lbl = self.font_small.render("Max Tile Reached | Failures", True, TEXT_DARK)
    self.screen.blit(header_lbl, (585, y_offset))
    y_offset += 30

    sorted_tiles = sorted(self.failure_counter.keys())
    for tile in sorted_tiles:
      count = self.failure_counter[tile]
      row_text = f"Tile {tile:5d}  -->  {count:3d} deaths"
      lbl = self.font_med.render(row_text, True, (180, 50, 50))
      self.screen.blit(lbl, (585, y_offset))
      y_offset += 35

    # Hero stats at bottom
    hero_stats = f"Hero Score: {hero_game.score}"
    stat_lbl = self.font_med.render(hero_stats, True, TEXT_DARK)
    self.screen.blit(stat_lbl, (585, 540))


if __name__ == "__main__":
  app = SimulationVisualizer()
  app.run()
