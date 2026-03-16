# Chapter 1: The Game Engine — Connect Four From the Ground Up

## Why the Engine Matters

The game engine is the foundation of everything. MCTS calls it thousands of times per move to simulate future positions. The neural network needs board positions converted to tensors. Self-play generates hundreds of full games per training iteration.

Three requirements drive every design decision:

1. **Speed**: MCTS calls `make_move`, `get_legal_moves`, and `clone` thousands of times per AI turn. Slow functions here make the entire system slow.
2. **Correctness**: One missed win detection or illegal move bug corrupts the entire training pipeline. Self-play generates the training data — if the engine is wrong, the AI learns wrong.
3. **Neural network compatibility**: The engine must convert positions into the exact tensor format the network expects.

## Part 1: Board Representation

### The Grid

Connect Four is played on a 6-row, 7-column grid. We represent it as a NumPy array:

```python
self.board = np.zeros((ROWS, COLS), dtype=np.int8)
```

```
Row 0 (top):    [0, 0, 0, 0, 0, 0, 0]    ← pieces fall DOWN into columns
Row 1:          [0, 0, 0, 0, 0, 0, 0]
Row 2:          [0, 0, 0, 0, 0, 0, 0]
Row 3:          [0, 0, 0, 0, 0, 0, 0]
Row 4:          [0, 0, 0, 0, 0, 0, 0]
Row 5 (bottom): [0, 0, 0, 0, 0, 0, 0]    ← pieces land here first
```

Three values:
- `0` = empty
- `1` = Player 1 (X)
- `-1` = Player 2 (O)

### Why int8?

`dtype=np.int8` uses 1 byte per cell instead of 8 bytes (`int64`). The board is 42 cells, so int8 uses 42 bytes vs 336 bytes. This matters because MCTS clones the board thousands of times — smaller boards mean less memory allocation and faster copying.

### Why -1 Instead of 2 for Player 2?

Using `1` and `-1` is a deliberate choice that simplifies several operations:

**Switching players:** `self.current_player *= -1` — one multiplication instead of an if-else.

**Canonical board:** `self.board * self.current_player` — one multiplication flips the entire board perspective (explained in Part 4).

**Result from a player's perspective:** If player 1 won (`winner = 1`), then `winner == player` is `True` for player 1 and `False` for player -1. Clean comparisons.

If we used 1 and 2, all of these would require conditional logic.

## Part 2: Column Heights — The Speed Optimization

### The Naive Approach

To drop a piece in column 3, you'd scan from the bottom up:

```python
# Slow: scan every row
for row in range(ROWS - 1, -1, -1):
    if self.board[row, col] == 0:
        self.board[row, col] = self.current_player
        break
```

This is O(ROWS) per move — scanning up to 6 rows every time.

### The Fast Approach

We track the next open row for each column:

```python
self.column_heights = np.full(COLS, ROWS - 1, dtype=np.int8)  # all start at row 5 (bottom)
```

Now dropping a piece is O(1):

```python
row = self.column_heights[col]    # instant lookup
self.board[row, col] = self.current_player
self.column_heights[col] -= 1     # next piece goes one row higher
```

After player 1 drops in column 3:
```
column_heights: [5, 5, 5, 4, 5, 5, 5]
                              ↑ was 5, now 4 (next piece goes to row 4)
```

After another piece in column 3:
```
column_heights: [5, 5, 5, 3, 5, 5, 5]
                              ↑ now 3
```

When `column_heights[col] < 0`, the column is full. Legal move check becomes:

```python
def get_legal_moves(self):
    return [c for c in range(COLS) if self.column_heights[c] >= 0]
```

This optimization seems minor, but MCTS calls `make_move` and `get_legal_moves` tens of thousands of times per AI turn. O(1) vs O(ROWS) adds up.

## Part 3: Win Detection

### The Challenge

After every move, we need to check if someone won. Scanning the entire board for four-in-a-row is wasteful — if someone won, it's because of the **last move played**. We only need to check lines passing through the last-placed piece.

### Four Directions

From any position, four-in-a-row can occur in 4 directions:

```
Horizontal:    ─ ─ ─ ─        direction (0, 1)
Vertical:      │               direction (1, 0)
               │
               │
               │
Diagonal ╲:    ╲              direction (1, 1)
                ╲
                 ╲
                  ╲
Diagonal ╱:       ╱           direction (1, -1)
                ╱
               ╱
              ╱
```

### The Algorithm

For each direction, count consecutive same-colored pieces extending from the placed piece in both directions:

```python
def _check_win(self, row, col):
    player = self.board[row, col]
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
```

### A Traced Example

Piece placed at row 5, col 3. Checking horizontal direction (0, 1):

```
Board row 5: [0, 0, 0, X, X, X, X]
                       ↑ placed here

Positive direction (right): (5,4)=X ✓, (5,5)=X ✓, (5,6)=X ✓  → count +3
Negative direction (left):  (5,2)=0 ✗                          → count +0

Total count = 1 + 3 + 0 = 4 → WIN!
```

For each direction, we scan at most 3 cells in each way (only need 4 total). That's at most 4 directions × 6 cells = 24 array accesses. Far faster than scanning all 42 cells × 4 directions.

### Why `break` Is Important

```python
if 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
    count += 1
else:
    break
```

The `break` stops scanning as soon as the chain is broken. If cells are `[X, X, O, X]`, we count 2 (not 3) — the O breaks the chain. Without `break`, we'd count non-consecutive pieces.

## Part 4: The Canonical Board — The Key Design Decision

### The Problem

The neural network needs to evaluate positions for both players. Without any trick, the network would need to learn two completely separate strategies — "how to play as player 1" AND "how to play as player 2."

### The Solution: Canonical Form

We always present the board from the **current player's perspective**. The current player's pieces are always `+1`, the opponent's are always `-1`:

```python
def get_canonical_board(self):
    return self.board * self.current_player
```

When it's player 1's turn (`current_player = 1`): board × 1 = board unchanged.
When it's player -1's turn (`current_player = -1`): board × -1 = all pieces flipped.

### A Concrete Example

Actual board state:
```
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . O . . .        1 = X (player 1), -1 = O (player 2)
. . . X . . .
```

If it's player 1's turn, the canonical board is the same:
```
My pieces (+1) = X    Opponent (-1) = O
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . O . . .        "I'm X, opponent is O"
. . . X . . .
```

If it's player -1's turn, multiply by -1:
```
My pieces (+1) = O    Opponent (-1) = X
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . +1. . .        "I'm O (now +1), opponent is X (now -1)"
. . . -1. . .
```

The network always sees "my pieces = +1, opponent's pieces = -1." It learns one strategy that works for both players.

### Why This Cuts Learning in Half

Without canonical form, the network would see:
- Position A: "X has 3 in a row in column 3" → learn to complete it
- Position B: "O has 3 in a row in column 3" → learn to complete it (same pattern, different player!)

With canonical form, both situations look identical to the network: "I (+1) have 3 in a row in column 3" → learn to complete it. One lesson instead of two.

## Part 5: Neural Network Input Encoding

### The Three-Channel Representation

The neural network doesn't receive the raw board (a 6×7 grid of -1, 0, 1). Instead, we encode it as three binary channels:

```python
def get_nn_input(self):
    canonical = self.get_canonical_board()
    my_pieces = (canonical == 1).astype(np.float32)      # Channel 0
    opp_pieces = (canonical == -1).astype(np.float32)     # Channel 1
    turn_plane = np.ones((ROWS, COLS), dtype=np.float32)  # Channel 2
    return np.stack([my_pieces, opp_pieces, turn_plane], axis=0)
```

**Channel 0 — My pieces**: 1 where I have a piece, 0 everywhere else.
**Channel 1 — Opponent's pieces**: 1 where the opponent has a piece, 0 everywhere else.
**Channel 2 — Turn indicator**: all 1s (always the same since we use canonical form, but standard in AlphaZero implementations).

### Why Three Channels Instead of One?

The raw board has three states per cell (-1, 0, 1). A convolutional neural network processes each channel independently in its first layer. Separating "my pieces" and "opponent's pieces" into distinct channels lets the network learn different filters for each — one set detecting my threats, another detecting opponent threats.

With a single channel, the network would need to learn that +1 and -1 have opposite meanings, which is harder. The three-channel representation makes this explicit.

### Shape: (3, 6, 7)

```
Channel 0 (my pieces):
  [[0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 1, 0, 0, 0]]     ← my piece at (5,3)

Channel 1 (opponent pieces):
  [[0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 1, 0, 0, 0],      ← opponent at (4,3)
   [0, 0, 0, 0, 0, 0, 0]]

Channel 2 (turn indicator):
  [[1, 1, 1, 1, 1, 1, 1],
   [1, 1, 1, 1, 1, 1, 1],
   ...all ones...              ]
```

This is the standard image format for CNNs: (channels, height, width). Just like a color image has 3 channels (red, green, blue), our board has 3 channels (my pieces, opponent pieces, turn).

## Part 6: Legal Move Masking

```python
def get_legal_moves_mask(self):
    return np.array([1.0 if self.column_heights[c] >= 0 else 0.0
                     for c in range(COLS)], dtype=np.float32)
```

Returns `[1, 1, 1, 1, 1, 1, 1]` when all columns are open, or `[1, 1, 1, 0, 1, 1, 1]` when column 3 is full.

This mask is multiplied with the neural network's policy output to zero out illegal moves. Without masking, the network might suggest playing in a full column — leading to an error in `make_move`.

## Part 7: Clone — Deep Copy for MCTS

```python
def clone(self):
    g = ConnectFour()
    g.board = self.board.copy()
    g.current_player = self.current_player
    g.winner = self.winner
    g.done = self.done
    g.move_count = self.move_count
    g.last_move = self.last_move
    g.column_heights = self.column_heights.copy()
    return g
```

MCTS needs to simulate hypothetical futures: "What if I play column 3, then the opponent plays column 5, then I play column 2..." Each simulation needs its own copy of the game state. Modifying a clone must NOT affect the original.

Two things need explicit `.copy()`:
- `self.board.copy()` — NumPy arrays are passed by reference. Without `.copy()`, modifying the clone's board modifies the original's.
- `self.column_heights.copy()` — same reason.

Scalars (`current_player`, `winner`, `done`, `move_count`) are automatically copied by value in Python — no `.copy()` needed.

### How MCTS Uses Clone

```python
# MCTS simulation
child_game = game.clone()        # copy the current state
child_game.make_move(column)     # try a move (modifies the CLONE only)
value = evaluate(child_game)     # evaluate the resulting position
# game is unchanged — ready for the next simulation
```

Each of the 50-200 MCTS simulations starts with a clone. That's 50-200 copies per AI move, with potentially dozens of clones per simulation (as MCTS walks deeper into the tree). Speed matters.

## Part 8: Game Result

```python
def get_result(self, player):
    if not self.done:
        return None
    if self.winner == 0:
        return 0
    return 1 if self.winner == player else -1
```

Returns the result from a specific player's perspective:
- `+1` if that player won
- `-1` if that player lost
- `0` for a draw
- `None` if the game is still ongoing

This is used by MCTS to backpropagate results and by self-play to generate training targets. The perspective matters — if Player 1 won, that's `+1` from Player 1's perspective and `-1` from Player 2's perspective.

## Part 9: What We Verified

Our test suite confirmed:

| Test | What It Checks | Why It Matters |
|------|---------------|----------------|
| Horizontal win | 4 in a row on bottom row | Most common win type |
| Vertical win | 4 stacked in column 0 | Tests the stacking mechanic |
| Diagonal win | 4 on a diagonal | Hardest to detect, most missed bugs |
| Clone independence | Modifying clone doesn't affect original | MCTS would break without this |
| NN input shape | Output is (3, 6, 7) | Network expects exactly this shape |
| Legal moves mask | All 1s when board is empty, 0 for full columns | Prevents illegal move suggestions |
| 1000 random games | P1 wins ~56%, completes without errors | Stress test for edge cases |

The 56% Player 1 win rate in random play confirms the first-move advantage. With perfect play, Player 1 wins 100% — so even random play shows the advantage slightly.

## Summary

| Component | What | Why |
|-----------|------|-----|
| `np.int8` board | 6×7 grid, values -1/0/1 | Compact, fast to copy |
| `column_heights` | Next open row per column | O(1) move execution |
| Directional win check | Only check around last move | O(1) instead of O(N²) |
| Canonical board | Multiply by `current_player` | Network learns one strategy for both players |
| 3-channel NN input | My pieces / opponent / turn | Standard CNN input format |
| Legal move mask | Binary array, length 7 | Prevents illegal network suggestions |
| Clone with `.copy()` | Independent deep copy | MCTS simulations don't corrupt real game |

The game engine is called more than any other component — every MCTS simulation, every self-play game, every evaluation match passes through it. Getting it right and fast is what makes everything else work.

## What's Next

In [Chapter 2](02_mcts.md), we build Monte Carlo Tree Search — the algorithm that explores the game tree to find the best move. MCTS is the core of AlphaZero's strength, and understanding it deeply is the single most important concept in this project.