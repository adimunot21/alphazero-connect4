"""
Connect Four Game Engine.

Board: 6 rows × 7 columns. Row 0 is the TOP, row 5 is the BOTTOM.
Pieces fall down, so they stack from the bottom.

Players: 1 and -1
Empty cells: 0

The board is always presented to the neural network from the
CURRENT player's perspective — we flip pieces (multiply by -1)
when switching turns. This means the network only learns "how to
play when my pieces are +1" and it works for both players.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Board dimensions
ROWS = 6
COLS = 7
WIN_LENGTH = 4


class ConnectFour:
    """
    Complete Connect Four game state.

    Designed for speed (MCTS will call this thousands of times)
    and correctness (every edge case handled).
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1    # Player 1 starts
        self.winner = None         # None = ongoing, 1/-1 = winner, 0 = draw
        self.done = False
        self.move_count = 0
        self.last_move = None      # (row, col) of most recent move
        # Track the next open row for each column (speeds up move execution)
        self.column_heights = np.full(COLS, ROWS - 1, dtype=np.int8)  # bottom row = 5

    def clone(self):
        """Create a deep copy. MCTS needs to simulate without modifying the real game."""
        g = ConnectFour()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.winner = self.winner
        g.done = self.done
        g.move_count = self.move_count
        g.last_move = self.last_move
        g.column_heights = self.column_heights.copy()
        return g

    def get_legal_moves(self):
        """Return list of columns that aren't full."""
        return [c for c in range(COLS) if self.column_heights[c] >= 0]

    def make_move(self, col):
        """
        Drop a piece in the given column. Returns self for chaining.
        Raises ValueError if the move is illegal.
        """
        if self.done:
            raise ValueError("Game is already over")
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} is out of range")

        row = self.column_heights[col]
        if row < 0:
            raise ValueError(f"Column {col} is full")

        # Place the piece
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.column_heights[col] -= 1
        self.move_count += 1

        # Check for win (only need to check around the last move)
        if self._check_win(row, col):
            self.winner = self.current_player
            self.done = True
        elif self.move_count == ROWS * COLS:
            self.winner = 0  # draw
            self.done = True

        # Switch player
        self.current_player *= -1
        return self

    def _check_win(self, row, col):
        """
        Check if the last move at (row, col) creates 4 in a row.

        Only checks lines through the placed piece — much faster than
        scanning the entire board. There are 4 directions to check:
        horizontal, vertical, diagonal-right, diagonal-left.
        """
        player = self.board[row, col]

        # 4 direction vectors: (row_delta, col_delta)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1  # count the piece itself

            # Count in the positive direction
            for i in range(1, WIN_LENGTH):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                else:
                    break

            # Count in the negative direction
            for i in range(1, WIN_LENGTH):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
                    count += 1
                else:
                    break

            if count >= WIN_LENGTH:
                return True

        return False

    def get_canonical_board(self):
        """
        Return the board from the current player's perspective.

        If current_player is 1: board as-is (1 = my pieces, -1 = opponent)
        If current_player is -1: flip all pieces (multiply by -1)

        This lets the neural network always see "my pieces = +1".
        """
        return self.board * self.current_player

    def get_nn_input(self):
        """
        Convert board to neural network input tensor.

        3 channels × 6 rows × 7 columns:
          Channel 0: current player's pieces (1 where they have a piece)
          Channel 1: opponent's pieces (1 where they have a piece)
          Channel 2: constant plane indicating whose turn (all 1s = player 1's turn)

        Returns numpy array of shape (3, 6, 7).
        """
        canonical = self.get_canonical_board()
        # Channel 0: my pieces
        my_pieces = (canonical == 1).astype(np.float32)
        # Channel 1: opponent pieces
        opp_pieces = (canonical == -1).astype(np.float32)
        # Channel 2: turn indicator (always 1 since canonical board is from current player's view)
        turn_plane = np.ones((ROWS, COLS), dtype=np.float32)
        return np.stack([my_pieces, opp_pieces, turn_plane], axis=0)

    def get_legal_moves_mask(self):
        """Binary mask: 1 for legal moves, 0 for illegal (full columns)."""
        return np.array([1.0 if self.column_heights[c] >= 0 else 0.0
                         for c in range(COLS)], dtype=np.float32)

    def get_result(self, player):
        """
        Get the game result from a specific player's perspective.
        Returns: +1 (player won), -1 (player lost), 0 (draw), None (game ongoing)
        """
        if not self.done:
            return None
        if self.winner == 0:
            return 0
        return 1 if self.winner == player else -1

    def __repr__(self):
        """Pretty-print the board."""
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        lines.append("  " + " ".join(str(c) for c in range(COLS)))
        lines.append("  " + "-" * (COLS * 2 - 1))
        for r in range(ROWS):
            row_str = "  ".join(symbols[self.board[r, c]] for c in range(COLS))
            lines.append(f"  {row_str}")
        lines.append("  " + "-" * (COLS * 2 - 1))
        player_str = "X" if self.current_player == 1 else "O"
        if self.done:
            if self.winner == 0:
                lines.append("  Result: Draw")
            else:
                winner_str = "X" if self.winner == 1 else "O"
                lines.append(f"  Winner: {winner_str}")
        else:
            lines.append(f"  Turn: {player_str} (player {self.current_player})")
        return "\n".join(lines)


def render_board(game, save_path=None, title=None):
    """Render the board as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Blue background (the board)
    board_rect = patches.FancyBboxPatch(
        (-0.5, -0.5), COLS, ROWS,
        boxstyle="round,pad=0.1",
        facecolor="#2856a3", edgecolor="#1a3d7c", linewidth=3
    )
    ax.add_patch(board_rect)

    # Draw pieces
    for r in range(ROWS):
        for c in range(COLS):
            val = game.board[r, c]
            if val == 1:
                color = "#e74c3c"     # red
            elif val == -1:
                color = "#f1c40f"     # yellow
            else:
                color = "#1a3d7c"     # dark blue (empty hole)

            circle = plt.Circle((c, ROWS - 1 - r), 0.4, color=color, ec="black", linewidth=1.5)
            ax.add_patch(circle)

    # Column numbers
    for c in range(COLS):
        ax.text(c, ROWS + 0.1, str(c), ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_xlim(-0.7, COLS - 0.3)
    ax.set_ylim(-0.7, ROWS + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.close()


def play_random_game(verbose=False):
    """Play a complete game with two random players."""
    game = ConnectFour()
    while not game.done:
        legal = game.get_legal_moves()
        move = np.random.choice(legal)
        game.make_move(move)
        if verbose:
            print(game)
            print()
    return game


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: Basic gameplay
    print("=" * 40)
    print("TEST 1: Basic game")
    print("=" * 40)
    game = ConnectFour()
    print(game)
    print()

    # Make some moves
    moves = [3, 3, 4, 4, 5, 5, 6]  # Player 1 wins horizontally on bottom row
    for col in moves:
        game.make_move(col)

    print(f"After moves {moves}:")
    print(game)
    assert game.done, "Game should be over"
    assert game.winner == 1, f"Player 1 should win, got {game.winner}"
    print("✓ Horizontal win detected correctly\n")

    # Test 2: Vertical win
    print("=" * 40)
    print("TEST 2: Vertical win")
    print("=" * 40)
    game = ConnectFour()
    moves = [0, 1, 0, 1, 0, 1, 0]  # Player 1 stacks 4 in column 0
    for col in moves:
        game.make_move(col)
    print(game)
    assert game.winner == 1, f"Player 1 should win vertically, got {game.winner}"
    print("✓ Vertical win detected correctly\n")

    # Test 3: Diagonal win
    print("=" * 40)
    print("TEST 3: Diagonal win")
    print("=" * 40)
    game = ConnectFour()
    # Build a diagonal: player 1 at (5,0), (4,1), (3,2), (2,3)
    moves = [0, 1, 1, 2, 3, 2, 2, 3, 3, 6, 3]
    for col in moves:
        game.make_move(col)
    print(game)
    assert game.winner == 1, f"Player 1 should win diagonally, got {game.winner}"
    print("✓ Diagonal win detected correctly\n")

    # Test 4: Clone independence
    print("=" * 40)
    print("TEST 4: Clone")
    print("=" * 40)
    game = ConnectFour()
    game.make_move(3)
    game.make_move(4)
    clone = game.clone()
    clone.make_move(3)
    assert game.move_count == 2, "Original should be unchanged"
    assert clone.move_count == 3, "Clone should have 3 moves"
    print("✓ Clone is independent\n")

    # Test 5: NN input shape
    print("=" * 40)
    print("TEST 5: Neural network input")
    print("=" * 40)
    game = ConnectFour()
    game.make_move(3)
    game.make_move(4)
    nn_input = game.get_nn_input()
    print(f"NN input shape: {nn_input.shape}")
    assert nn_input.shape == (3, 6, 7), f"Expected (3,6,7), got {nn_input.shape}"
    mask = game.get_legal_moves_mask()
    print(f"Legal moves mask: {mask}")
    assert mask.sum() == 7, "All columns should be legal at start"
    print("✓ NN input shape correct\n")

    # Test 6: Random game
    print("=" * 40)
    print("TEST 6: Random game")
    print("=" * 40)
    results = {1: 0, -1: 0, 0: 0}
    for _ in range(1000):
        g = play_random_game()
        results[g.winner] += 1
    print(f"1000 random games: P1 wins={results[1]}, P2 wins={results[-1]}, draws={results[0]}")
    print("(Player 1 should win slightly more often — first-move advantage)")
    print("✓ Random games complete\n")

    # Test 7: Render
    game = ConnectFour()
    for col in [3, 3, 4, 2, 2, 4, 0, 1, 5, 6]:
        if game.done:
            break
        game.make_move(col)
    render_board(game, save_path="notebooks/board_example.png", title="Example Position")

    print("All tests passed!")