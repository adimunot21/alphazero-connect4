# Chapter 4: Self-Play and Training — The AlphaZero Loop

## The Central Insight

Traditional supervised learning needs labeled data: "this image is a cat," "this board position should play column 3." Someone has to provide those labels — human experts, existing databases, manual annotation.

AlphaZero has no labeled data. No human expert games. No opening books. It generates its own training data through **self-play** and its own labels through **MCTS search**.

The key insight: **MCTS + network produces better moves than the network alone.** Searching 50 future positions is more accurate than a raw guess. So the MCTS policy is a better "label" than the network's own output. Training the network to match MCTS makes the network stronger, which makes MCTS stronger, which produces even better labels.

This is bootstrapping — the system pulls itself up by its own bootstraps.

## Part 1: Self-Play — Generating Training Data

### One Self-Play Game

```python
def self_play_game(network, num_simulations=50, temperature_threshold=15):
    game = ConnectFour()
    mcts = MCTS(network=network, num_simulations=num_simulations)
    history = []

    move_num = 0
    while not game.done:
        # Run MCTS to get a move distribution
        action_probs = mcts.search(game)

        # Store position + MCTS policy + current player
        nn_input = game.get_nn_input()
        history.append((nn_input, action_probs, game.current_player))

        # Select move with temperature
        if move_num < temperature_threshold:
            move = np.random.choice(COLS, p=action_probs)
        else:
            move = np.argmax(action_probs)

        game.make_move(move)
        move_num += 1

    # Fill in results
    training_examples = []
    for nn_input, policy, player in history:
        result = game.get_result(player)
        training_examples.append((nn_input, policy, result if result is not None else 0))

    return training_examples
```

### What Gets Stored Per Position

Each position produces one training example with three components:

```
(nn_input, mcts_policy, game_result)

nn_input:     (3, 6, 7) tensor — the board from current player's perspective
mcts_policy:  (7,) array — MCTS visit count distribution over columns
game_result:  +1 (this position's player won), -1 (lost), or 0 (draw)
```

A typical 20-move game produces 20 training examples — one for each position that was played.

### The Two Training Targets

**Policy target: the MCTS visit distribution.** NOT the network's own policy, NOT the move that was actually played. MCTS searched ahead and found a better answer than the raw network. The network should learn to match it.

```
Network's raw policy:  [0.10, 0.12, 0.15, 0.30, 0.18, 0.10, 0.05]
MCTS policy (target):  [0.02, 0.05, 0.08, 0.55, 0.20, 0.08, 0.02]

MCTS concentrated on column 3 (0.55) after searching ahead.
The network should learn to give column 3 higher probability.
```

**Value target: who actually won the game.** After the game ends, every position gets labeled with the outcome from that position's player's perspective.

```
Position at move 5 (player 1's turn):
  Player 1 eventually won → value target = +1

Position at move 6 (player -1's turn):
  Player 1 eventually won → from player -1's perspective → value target = -1
```

### Temperature Strategy

```python
if move_num < temperature_threshold:     # first 15 moves
    move = np.random.choice(COLS, p=action_probs)   # sample (explore)
else:
    move = np.argmax(action_probs)                    # best move (exploit)
```

**Why explore early?** Without temperature, every game from the same starting position follows the same moves (always picking MCTS's top choice). The network would only see one opening sequence and never learn alternatives.

With temperature = 1 for the first 15 moves, the agent sometimes plays its second or third choice. This creates diverse positions:

```
Game 1: opens with columns [3, 3, 4, 2, ...]    (standard center play)
Game 2: opens with columns [0, 3, 5, 1, ...]    (unusual flank opening)
Game 3: opens with columns [3, 4, 3, 3, ...]    (aggressive center stack)
```

Each game explores different parts of the game tree, giving the network a broader education.

**Why exploit late?** Late-game positions should be played optimally so the game result accurately reflects who's winning. If both sides play randomly in the endgame, the outcome is noisy — a position that's actually winning might end in a loss due to random play, giving the value head a wrong label.

### Self-Play Statistics Over Training

```
Iteration 1:  P1 53%, P2 45%, Draw 2%    avg game length: ~19 moves
              (nearly random — untrained network)

Iteration 10: P1 66%, P2 31%, Draw 3%    avg game length: ~23 moves
              (learning P1 advantage, games getting longer)

Iteration 19: P1 55%, P2 34%, Draw 11%   avg game length: ~24 moves
              (stronger play → more draws, both sides defend well)
```

The increasing draw rate is a sign of improvement — when both sides play well, it's harder to win. In perfectly played Connect Four, Player 1 always wins, but with 50 simulations the AI can't play perfectly, so draws emerge when both sides play defensively.

## Part 2: Data Augmentation — Free Training Data

### Horizontal Symmetry

Connect Four is symmetric — flipping the board left-right and reversing the policy produces an equally valid position:

```
Original board:              Flipped board:
. . . . . . .               . . . . . . .
. . . . . . .               . . . . . . .
. . . . . . .               . . . . . . .
. . . . . . .               . . . . . . .
. . . O . . .               . . . O . . .
. . . X . . .               . . . X . . .

Policy: [0.05, 0.10, 0.15, 0.40, 0.15, 0.10, 0.05]
Flipped: [0.05, 0.10, 0.15, 0.40, 0.15, 0.10, 0.05]
(this position is symmetric, but in general columns get reversed)
```

An asymmetric example:

```
Original:                    Flipped:
. . . . . . .               . . . . . . .
. . . . . . .               . . . . . . .
. . . . . . .               . . . . . . .
. . . . . . .               . . . . . . .
X . . O . . .               . . . O . . X
X . . O . . .               . . . O . . X

Policy: [0.05, 0.05, 0.10, 0.20, 0.30, 0.20, 0.10]
Flipped: [0.10, 0.20, 0.30, 0.20, 0.10, 0.05, 0.05]
```

The flipped position is equally valid — column 0 in the original becomes column 6 in the flip. The network should evaluate both the same way.

### Implementation

```python
class AlphaZeroDataset(Dataset):
    def __init__(self, examples, augment=True):
        self.examples = list(examples)
        self.augment = augment

    def __len__(self):
        return len(self.examples) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        if idx < len(self.examples):
            nn_input, policy, result = self.examples[idx]
            flipped = False
        else:
            nn_input, policy, result = self.examples[idx - len(self.examples)]
            flipped = True

        state = torch.FloatTensor(nn_input)
        pi = torch.FloatTensor(policy)
        v = torch.FloatTensor([result])

        if flipped:
            state = torch.flip(state, dims=[2])  # flip columns
            pi = torch.flip(pi, dims=[0])         # reverse policy

        return state, pi, v
```

`torch.flip(state, dims=[2])` flips the width dimension (columns). `torch.flip(pi, dims=[0])` reverses the policy array (column 0 ↔ column 6, column 1 ↔ column 5, etc.).

This doubles our training data for free. 2000 self-play positions become 4000 training examples without playing any additional games.

## Part 3: The Training Step

### Two Losses

```python
def train_on_data(network, optimizer, dataset, epochs, batch_size, device):
    network.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for states, target_pis, target_vs in loader:
            states = states.to(device)
            target_pis = target_pis.to(device)
            target_vs = target_vs.squeeze(-1).to(device)

            policy_logits, values = network(states)

            # Policy loss: cross-entropy with MCTS policy
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(target_pis * log_probs, dim=1).mean()

            # Value loss: MSE between predicted and actual outcome
            value_loss = F.mse_loss(values, target_vs)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Policy Loss: Soft Cross-Entropy

```python
log_probs = F.log_softmax(policy_logits, dim=1)
policy_loss = -torch.sum(target_pis * log_probs, dim=1).mean()
```

This is cross-entropy with **soft labels**. Normal cross-entropy uses hard labels (the correct class is 1, everything else is 0). Our MCTS targets are probabilities:

```
Target:   [0.02, 0.05, 0.08, 0.55, 0.20, 0.08, 0.02]
                                ↑ MCTS's preferred move (55%)
```

The loss pushes the network's output toward this distribution. If the network currently predicts uniform (all ~0.14), the gradient says "increase column 3's probability and decrease the others."

**Why not standard `F.cross_entropy`?** PyTorch's `cross_entropy` expects hard labels (a single class index). Our targets are probability distributions over all 7 columns. We compute the cross-entropy manually: `-sum(target × log(prediction))`.

**Why `log_softmax` instead of `softmax` then `log`?** Numerical stability. Computing `softmax` then `log` can produce `-inf` for very small probabilities. `log_softmax` combines both operations in a numerically stable way.

### Value Loss: Mean Squared Error

```python
value_loss = F.mse_loss(values, target_vs)
```

Straightforward: the network predicts a value, the target is the game outcome (+1, -1, or 0). MSE penalizes predictions proportionally to the square of their error.

```
Network predicts: +0.3  (thinks this player is slightly ahead)
Actual outcome:   -1.0  (this player lost)
Loss: (0.3 - (-1.0))² = (1.3)² = 1.69   (large error → large loss)
```

### Combined Loss

```python
loss = policy_loss + value_loss
```

Both losses have roughly similar magnitudes (policy loss starts ~1.9, value loss starts ~0.5), so equal weighting works. Some implementations add a weight: `loss = policy_loss + c × value_loss`, but for Connect Four the simple sum is fine.

### Weight Decay

```python
optimizer = torch.optim.Adam(
    network.parameters(),
    lr=0.001,
    weight_decay=1e-4,
)
```

`weight_decay=1e-4` adds L2 regularization — it slightly penalizes large weights, preventing overfitting. This is important because the replay buffer might have limited diversity (especially early in training when games are short and similar).

## Part 4: The Replay Buffer

```python
replay_buffer = deque(maxlen=50000)
```

The replay buffer stores training positions from recent self-play games. New positions are added each iteration; old positions are dropped when the buffer is full (FIFO — first in, first out).

**Why a buffer instead of just using the latest games?** Early iterations produce only ~2000 positions per iteration. Training on just 2000 examples would massively overfit. The buffer accumulates positions across iterations, providing more diverse training data.

**Why limit the buffer size?** Old positions were generated by a weaker network — they represent weaker play and might teach the current (stronger) network bad habits. Limiting to 50,000 keeps roughly the last 25 iterations of data, ensuring the training data is relatively fresh.

### Buffer Growth Over Training

```
Iteration 1:   1,885 positions  (all new, small buffer)
Iteration 5:   9,657 positions  (growing, more diverse)
Iteration 10: 20,775 positions  (substantial training set)
Iteration 15: 31,555 positions  (approaching capacity)
Iteration 20: 44,261 positions  (old positions being dropped)
```

## Part 5: The Complete Training Loop

```python
for iteration in range(1, num_iterations + 1):

    # 1. SELF-PLAY
    network.eval()
    new_examples = generate_self_play_data(
        network, num_games=100, num_simulations=50
    )
    replay_buffer.extend(new_examples)

    # 2. TRAIN
    dataset = AlphaZeroDataset(list(replay_buffer), augment=True)
    network.to(device)
    policy_loss, value_loss = train_on_data(
        network, optimizer, dataset,
        epochs=10, batch_size=64, device=device
    )

    # 3. EVALUATE
    network.eval()
    evaluate_against_random(network)
    evaluate_against_pure_mcts(network)
```

### Why `network.eval()` During Self-Play

During self-play, we need the network to make predictions (for MCTS). We call `.eval()` to disable BatchNorm's batch statistics and dropout. The network should behave deterministically — same input always gives same output. Without `.eval()`, BatchNorm would use per-batch statistics that vary with the number of MCTS simulations.

### Why `network.to(device)` Before Training

Self-play runs MCTS on CPU (tree search is sequential, can't be parallelized on GPU). Training uses the GPU for batched forward/backward passes. We move the network between devices:

```
self_play:  network.to("cpu")     → MCTS predictions
train:      network.to("cuda")    → batched gradient descent
evaluate:   network.to("cpu")     → MCTS predictions again
```

## Part 6: The Self-Improvement Cycle — Why It Works

### Iteration 1 (Random Network)

```
Network policy: nearly uniform [0.14, 0.14, ..., 0.14]
Network value: near zero for all positions

MCTS: uses 50 simulations with random-ish priors
      Still beats random because search alone helps
      Generates OK training data (MCTS policy is slightly better than uniform)

Training: network learns basic patterns
  "Column 3 is often visited more by MCTS → increase its prior"
  "Positions where I have 3 in a row often lead to wins → predict positive value"
```

### Iteration 5 (Learning Network)

```
Network policy: starting to prefer center, avoid edges
Network value: can identify obvious winning/losing positions

MCTS: guided by better priors, search focuses on good moves faster
      50 simulations achieve what 100 would have in iteration 1
      Generates better training data (MCTS policy is more accurate)

Training: network learns deeper patterns
  "When I have a diagonal threat, column X extends it → strong preference"
  "Opponent's double threat means I'm likely losing → predict negative value"
```

### Iteration 20 (Strong Network)

```
Network policy: sharp — strongly prefers good moves, ignores bad ones
Network value: accurately predicts win/loss most of the time

MCTS: network priors immediately guide search to the best moves
      50 simulations explore tactical variations deeply
      Generates expert-level training data

Training: network fine-tunes subtle patterns
  "In this specific configuration, column 2 sets up a trap 4 moves ahead"
  "This position looks even but actually favors Player 1 because of center control"
```

### The Feedback Loop

```
Better network → MCTS searches more efficiently
  → Better self-play games → Better training targets
  → Network improves → MCTS searches even more efficiently
  → ...
```

Each component amplifies the other. The network makes MCTS smarter. MCTS makes the network's training data better. Better training data makes the network smarter. The cycle continues until convergence.

## Part 7: Training Progression — Our Results

### Loss Curves

```
Iteration:    1      5      10     15     20
Policy loss:  1.91   1.56   1.44   1.42   1.48
Value loss:   0.48   0.32   0.32   0.33   0.36
```

**Policy loss drops from 1.91 to ~1.44.** The network gets better at predicting MCTS's move choices. `ln(7) ≈ 1.95` is the loss for a uniform policy over 7 columns — our network started there (random) and improved to ~1.44 (significantly better than random).

**Value loss drops from 0.48 to ~0.33.** The maximum possible MSE for values in [-1, +1] is 4.0 (predicting +1 when the answer is -1). Our value loss of 0.33 means the network's predictions are typically off by about `√0.33 ≈ 0.57`. Not perfect, but useful — it can distinguish winning from losing positions most of the time.

**Policy loss plateaus after iteration 10.** The network quickly learns the obvious moves (block threats, extend rows) and then improvement slows on more subtle positional judgments. More training data and simulations would push it further.

### Win Rates

```
vs Random:     100% from iteration 1 (MCTS dominates regardless of network)
vs Pure MCTS:  10% → 70% → 80% → 80% → 80%
```

The vs Pure MCTS number is the true measure of the network's contribution. At 80%, network-guided MCTS wins 4 out of 5 games against pure MCTS with the same simulation count. The network makes each simulation ~4x more effective by focusing the search on good moves.

## Part 8: What Could Be Improved

### More Simulations

We used 50 simulations per move during self-play. AlphaZero for Chess/Go used 800. More simulations produce higher-quality training data:
- Better MCTS policies (more accurate visit distributions)
- More consistent game outcomes (fewer random outcomes from insufficient search)
- Computationally expensive — the main bottleneck is MCTS speed

### More Iterations

We trained for 20 iterations. AlphaZero for Chess trained for hundreds of thousands. More iterations means:
- More diverse training data (the replay buffer turns over multiple times)
- The network sees more positions and patterns
- Eventually converges to much stronger play

### Larger Network

Our 377K parameter network captures the major patterns but likely misses subtle tactics. A larger network (more residual blocks, more filters) could represent more complex strategies. The tradeoff: larger networks are slower for MCTS (each position evaluation takes longer).

### Tree Reuse

We build a new MCTS tree from scratch for every move. In practice, the subtree rooted at the opponent's move could be reused — all those simulations are still valid. This effectively gives ~2x more simulations per move for free.

## Summary

The AlphaZero training pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│  ITERATION LOOP (repeat 20 times):                          │
│                                                             │
│  1. SELF-PLAY (100 games):                                  │
│     For each game:                                          │
│       While game not over:                                  │
│         Run MCTS (50 sims, guided by network)               │
│         Store (board, MCTS_policy, player)                  │
│         Pick move (temp=1 early, temp=0 late)               │
│       Fill in game results as value targets                 │
│     Add all positions to replay buffer                      │
│                                                             │
│  2. TRAIN (10 epochs):                                      │
│     On replay buffer (with horizontal flip augmentation):   │
│     Policy loss = cross_entropy(network_policy, MCTS_policy)│
│     Value loss = MSE(network_value, game_outcome)           │
│     Total loss = policy_loss + value_loss                   │
│                                                             │
│  3. EVALUATE:                                               │
│     Network+MCTS vs Random (should be ~100%)                │
│     Network+MCTS vs Pure MCTS (measures network quality)    │
└─────────────────────────────────────────────────────────────┘
```

| Concept | What | Why |
|---------|------|-----|
| Self-play | AI plays against itself | Generates training data with no human input |
| MCTS policy as target | Train network to match MCTS | MCTS (with search) is smarter than raw network |
| Game outcome as value target | Train network to predict winner | Ground truth signal for position evaluation |
| Temperature | Stochastic early, deterministic late | Diverse openings + accurate endgames |
| Data augmentation | Horizontal flip | Double training data for free |
| Replay buffer | Store recent positions | Prevent overfitting, ensure diversity |
| Feedback loop | Better network → better MCTS → better data | Self-improvement from zero knowledge |

## What's Next

In [Chapter 5](05_evaluation.md), we evaluate the trained AI — pitting it against different opponents, letting humans play against it, and analyzing what strategic concepts it learned from pure self-play.