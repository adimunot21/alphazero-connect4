# Chapter 2: Monte Carlo Tree Search — Smart Exploration of the Game Tree

## The Problem MCTS Solves

You're at a Connect Four position and need to pick the best move. There are 7 possible columns. For each, your opponent has up to 7 responses. For each of those, you have up to 7 replies. The tree expands exponentially — after just 10 moves, there are roughly 280 million possible positions.

Three approaches to this:

**Exhaustive search (Minimax):** Explore every possible future. Guarantees finding the best move. Impossible for Connect Four — 4.5 trillion positions would take centuries.

**Fixed-depth search (like chess engines):** Explore all positions up to depth N, then evaluate with a hand-crafted function. Works well but requires a good evaluation function (hand-written by human experts) and misses tactics beyond the search depth.

**MCTS:** Selectively explore the most promising branches. No depth limit — some branches go deep, others are abandoned early. No hand-crafted evaluation — positions are evaluated by random play (pure MCTS) or a neural network (AlphaZero). The tree grows organically toward the most important parts of the game tree.

## Part 1: The Tree Structure

### What a Node Represents

Each node in the MCTS tree represents a game position. It stores everything needed for the search:

```python
class MCTSNode:
    def __init__(self, prior=0.0):
        self.visit_count = 0       # N(s,a): times this node was visited
        self.value_sum = 0.0       # W(s,a): total backpropagated value
        self.prior = prior         # P(s,a): network's initial guess of move quality
        self.children = {}         # action → child MCTSNode
        self.game_state = None     # the ConnectFour state at this node
```

**`visit_count`**: How many MCTS simulations have passed through this node. High visit count = we've explored this position thoroughly. Used in UCB formula and for final move selection.

**`value_sum`**: Sum of all values backpropagated through this node. Divided by `visit_count` gives the average value — our best estimate of how good this position is.

**`prior`**: The neural network's initial probability for the move leading to this node. Before any simulations, this is all we know about the move's quality. As simulations accumulate, the actual observed value (`value_sum / visit_count`) gradually overrides the prior.

**`children`**: Dictionary mapping column numbers (0-6) to child nodes. An empty dict means this node hasn't been expanded yet (it's a leaf).

### The Q-Value

```python
@property
def q_value(self):
    if self.visit_count == 0:
        return 0.0
    return self.value_sum / self.visit_count
```

The average observed value. After 10 simulations through this node where 7 resulted in wins (+1) and 3 in losses (-1):

```
value_sum = 7 × 1 + 3 × (-1) = 4
visit_count = 10
q_value = 4 / 10 = 0.4   → "this position is slightly favorable"
```

Q-values range from -1 (always lose from here) to +1 (always win from here). 0 means even.

### Tree Example

```
Root: position after 3 moves
├── Col 0: visits=12, Q=+0.3, prior=0.08
├── Col 1: visits=5,  Q=-0.1, prior=0.10
├── Col 2: visits=8,  Q=+0.2, prior=0.15
├── Col 3: visits=45, Q=+0.6, prior=0.35   ← most visited (best move)
│   ├── Opp Col 0: visits=8,  Q=-0.5
│   ├── Opp Col 3: visits=20, Q=-0.7       ← explored deeply
│   │   ├── Col 2: visits=6,  Q=+0.8
│   │   └── Col 5: visits=10, Q=+0.6
│   └── Opp Col 6: visits=12, Q=-0.4
├── Col 4: visits=15, Q=+0.4, prior=0.20
├── Col 5: visits=10, Q=+0.1, prior=0.07
└── Col 6: visits=5,  Q=0.0,  prior=0.05
```

Notice:
- Column 3 has 45 visits — MCTS spent most of its time here because it looks best
- The tree is deeper under column 3 — we explored opponent's responses carefully
- Column 1 and 6 have few visits — MCTS quickly determined they're not as good
- The tree is **asymmetric** — deeper where it matters, shallow where it doesn't

## Part 2: The Four-Step Loop

MCTS repeats four steps for each simulation (we run 50-200 simulations per move):

```
┌─────────────────────────────────────────────────────┐
│                  ONE MCTS SIMULATION                 │
│                                                      │
│  1. SELECT:    Walk down the tree, picking the       │
│                child with highest UCB at each node   │
│                                                      │
│  2. EXPAND:    At a leaf, create children for        │
│                all legal moves                       │
│                                                      │
│  3. EVALUATE:  Get the value of this new position    │
│                (random rollout or neural network)    │
│                                                      │
│  4. BACKPROP:  Send the value back up the path,      │
│                updating every node along the way     │
│                                                      │
│  Repeat 50-200 times                                │
└─────────────────────────────────────────────────────┘
```

Let's examine each step in detail.

## Part 3: SELECT — Navigating the Tree

### The UCB Formula

At each internal node, we need to choose which child to visit next. This is the exploration-exploitation dilemma again (from the bandits chapter of the RL project, if you did it). We want to:
- **Exploit**: visit children that have high Q-values (known good moves)
- **Explore**: visit children that haven't been tried much (unknown moves that might be great)

The UCB (Upper Confidence Bound) formula balances both:

```
UCB(s, a) = Q(s,a) + c_puct × P(s,a) × √(N(parent)) / (1 + N(child))
```

Where:
- `Q(s,a)` = average value of this child = **exploitation**
- `P(s,a)` = prior probability from the neural network
- `N(parent)` = visit count of the parent node
- `N(child)` = visit count of this child
- `c_puct` = exploration constant (we use 1.41)

The second term is the **exploration bonus**:

```
Exploration = c_puct × P(s,a) × √(N(parent)) / (1 + N(child))
```

Three properties make this work:

**High prior → more exploration.** If the network thinks a move is promising (high P), the bonus is larger. The network's initial guess guides early exploration — we don't waste simulations on moves the network thinks are terrible.

**Low visit count → more exploration.** The denominator `(1 + N(child))` means unvisited children have the maximum bonus. As a child gets visited more, the bonus shrinks. We don't keep exploring a move we already know is bad.

**Growing parent visits → growing exploration.** The numerator `√(N(parent))` grows (slowly) as the parent accumulates visits. This means even children that were passed over early eventually get explored — no move is permanently abandoned.

### A Worked Example

Parent has been visited 100 times. Three children:

```
Child A: visits=50, Q=+0.3, prior=0.35
  UCB = 0.3 + 1.41 × 0.35 × √100 / (1+50) = 0.3 + 0.097 = 0.397

Child B: visits=40, Q=+0.4, prior=0.20
  UCB = 0.4 + 1.41 × 0.20 × √100 / (1+40) = 0.4 + 0.069 = 0.469  ← HIGHEST

Child C: visits=10, Q=-0.1, prior=0.15
  UCB = -0.1 + 1.41 × 0.15 × √100 / (1+10) = -0.1 + 0.192 = 0.092
```

MCTS selects Child B — it has the best combination of observed value (+0.4) and exploration potential. Child A has been visited a lot (50 times) so its exploration bonus is small. Child C has a high exploration bonus (only 10 visits) but its Q-value is negative — we know it's probably bad.

### The Code

```python
def _select_child(self, node):
    best_score = -float("inf")
    best_action = None
    best_child = None
    parent_visits = node.visit_count

    for action, child in node.children.items():
        q = child.q_value
        u = (self.c_puct * child.prior *
             math.sqrt(parent_visits) / (1 + child.visit_count))
        score = q + u

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
```

Simple: iterate over all children, compute UCB for each, return the highest. No sorting needed — we just track the max.

## Part 4: EXPAND — Growing the Tree

When we reach a leaf node (no children), we create child nodes for all legal moves:

```python
def _expand(self, node):
    game = node.game_state
    legal_moves = game.get_legal_moves()

    if not legal_moves or game.done:
        return

    if self.network is not None:
        policy, _ = self._network_predict(game)
    else:
        policy = np.ones(COLS, dtype=np.float32)
        mask = game.get_legal_moves_mask()
        policy *= mask
        total = policy.sum()
        if total > 0:
            policy /= total

    for action in legal_moves:
        child = MCTSNode(prior=policy[action])
        child_game = game.clone()
        child_game.make_move(action)
        child.game_state = child_game
        node.children[action] = child
```

### Two Modes

**Pure MCTS (no network):** Each legal move gets equal prior: `1/num_legal_moves`. The search starts unbiased and lets the rollout results guide exploration.

**AlphaZero MCTS (with network):** The network's policy output becomes the prior. If the network says column 3 has probability 0.35, that child starts with `prior = 0.35`. The search is biased toward moves the network thinks are good — but UCB's exploration term ensures all moves eventually get tried.

### Why Clone the Game State

```python
child_game = game.clone()
child_game.make_move(action)
child.game_state = child_game
```

Each child needs its own game state because different children represent different future positions. Without cloning, all children would share (and corrupt) the same state.

## Part 5: EVALUATE — Assessing New Positions

After expanding a leaf, we need a value estimate for the new position. Two methods:

### Random Rollout (Pure MCTS)

Play random moves until the game ends. Return the result.

```python
def _rollout(self, game):
    eval_player = game.current_player * -1  # who just moved (parent's player)

    while not game.done:
        legal = game.get_legal_moves()
        move = np.random.choice(legal)
        game.make_move(move)

    result = game.get_result(eval_player)
    return result if result is not None else 0
```

Random rollouts are noisy — one random game might end in a win, the next in a loss from the same position. But averaged over many simulations, the results converge to the true value. This is the "Monte Carlo" in Monte Carlo Tree Search.

**Limitation:** A critical position (one move from winning/losing) might look average because random play doesn't exploit the opportunity. The random player might wander past a winning move and eventually lose. This is why AlphaZero's neural network evaluation is so much better.

### Neural Network Evaluation (AlphaZero)

Ask the network: "How good is this position?"

```python
def _network_evaluate(self, node):
    _, value = self._network_predict(node.game_state)
    return -value  # negate: value is from current player's perspective,
                   # but we need it from the parent's perspective
```

No random play needed. The network has seen thousands of positions and learned to estimate values accurately. One forward pass replaces hundreds of random games.

**Why negate?** The network evaluates from the current player's perspective at the leaf node. But we're backpropagating from the parent's perspective (the player who just moved). If the position is +0.7 for the current player, it's -0.7 for the player who moved to get here.

## Part 6: BACKPROPAGATE — Updating the Tree

The value flows back up the path from leaf to root, updating every node along the way:

```python
def _backpropagate(self, path, value):
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        value = -value  # flip perspective at each level
```

### Why Negate at Each Level

Players alternate turns. If a position is good for me, it's bad for my opponent:

```
Level 0 (my turn):      value = +0.6   "good for me"
Level 1 (opponent):     value = -0.6   "bad for opponent (= good for me)"
Level 2 (my turn):      value = +0.6   "good for me"
Level 3 (opponent):     value = -0.6   "bad for opponent"
```

By negating at each level, each node accumulates values from its own player's perspective. When the root node computes its Q-value, it correctly reflects how good each move is for the current player.

### A Complete Simulation Trace

Starting position: it's X's turn, 3 legal moves (columns 0, 3, 6).

```
Simulation 1:
  SELECT:    Root → pick child Col 3 (highest UCB, prior=0.35)
  EXPAND:    Col 3 is a leaf → create children for all opponent responses
  EVALUATE:  Network says this position is +0.4 (good for O, since O just moved)
             From X's perspective: -0.4
  BACKPROP:  
    Col 3 node: value_sum += -0.4, visits = 1
    Root node:  value_sum += +0.4, visits = 1  (negated: from O's perspective)
    Wait — that's wrong. Let me trace more carefully.
```

Actually, let's be precise about perspectives:

```
Root: X to move
  │
  └── Col 3: O to move (X just played col 3)
       │
       └── (leaf — just expanded)

Evaluate leaf from O's perspective: value = +0.3 (good for O)
For the parent (Col 3 node), we need X's perspective: -0.3

Backprop (reversed path = [leaf, Col 3 node, Root]):
  leaf:         value_sum += +0.3, visit_count += 1    (O's perspective)
  Col 3 node:   value_sum += -0.3, visit_count += 1    (X's perspective)
  Root:          value_sum += +0.3, visit_count += 1    (O's persp... wait)
```

The key insight: the root doesn't accumulate values for itself. The root's children do. After many simulations:

```
Root children:
  Col 0: visits=15, value_sum=-3.0   → Q = -0.20 (slightly bad for X)
  Col 3: visits=60, value_sum=+24.0  → Q = +0.40 (good for X)
  Col 6: visits=25, value_sum=+5.0   → Q = +0.20 (decent for X)
```

MCTS selects column 3 — highest visit count AND highest Q-value.

## Part 7: Move Selection — Converting Simulations to Actions

After all simulations, we extract a policy from the root's children:

```python
action_probs = np.zeros(COLS, dtype=np.float32)
for action, child in root.children.items():
    action_probs[action] = child.visit_count

total = action_probs.sum()
if total > 0:
    action_probs /= total
```

The visit count distribution IS the MCTS policy. Why visit counts, not Q-values?

**Visit counts are more robust.** A child visited 2 times with Q=+0.8 is less trustworthy than a child visited 50 times with Q=+0.5. Visit counts naturally incorporate confidence — well-explored moves have high counts, uncertain moves have low counts. UCB ensures that good moves get visited more, so high visit count ≈ high quality with high confidence.

### Temperature

For training self-play, we add temperature to encourage diverse moves:

```python
if temperature == 0:
    action = np.argmax(action_probs)    # always pick the best
else:
    adjusted = action_probs ** (1.0 / temperature)
    adjusted /= adjusted.sum()
    action = np.random.choice(COLS, p=adjusted)
```

**Temperature = 0:** Always pick the most-visited move. Used during evaluation (play the best move).

**Temperature = 1:** Sample proportional to visit counts. A move with 60% of visits gets picked 60% of the time, not 100%. Used during early self-play moves to create diverse training positions.

Without temperature, every self-play game from the same opening would play identically. The AI would never explore alternative strategies.

### Temperature Threshold

We use temperature = 1 for the first 15 moves, then switch to temperature = 0:

```
Moves 1-15:  temperature = 1.0  (diverse openings)
Moves 16+:   temperature → 0   (play the best move)
```

Early moves explore different openings — critical for generating diverse training data. Late moves play optimally — providing accurate training targets for the value head.

## Part 8: Dirichlet Noise — Ensuring Root Exploration

Even with UCB's exploration term, a strong network might be so confident about one move that other moves never get explored from the root:

```
Network policy: Col 3 = 0.85, all others < 0.03
UCB will almost always select Col 3 first, then Col 3's subtree absorbs
most simulations, and other columns barely get explored.
```

Dirichlet noise solves this by mixing random noise into the root's priors:

```python
def _add_dirichlet_noise(self, root):
    legal_actions = list(root.children.keys())
    noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))

    for i, action in enumerate(legal_actions):
        child = root.children[action]
        child.prior = ((1 - self.dirichlet_epsilon) * child.prior +
                       self.dirichlet_epsilon * noise[i])
```

```
Before noise: Col 3 prior = 0.85
After noise:  Col 3 prior = 0.75 × 0.85 + 0.25 × noise[3]
              ≈ 0.64 + 0.25 × (random value)
```

25% of the prior comes from random noise. This guarantees every legal move has a non-trivial prior, even if the network strongly prefers one move. Over many self-play games, the noise is different each time, so different alternatives get explored.

### Why Only at the Root?

Deeper nodes don't need noise. The root is the only position that repeats across simulations — every simulation starts from the root. Deeper nodes are visited less frequently, so UCB's exploration term is already sufficient. Adding noise everywhere would make the search too random.

### Dirichlet Distribution

`np.random.dirichlet([α, α, ..., α])` generates a random probability vector (sums to 1). The parameter α controls concentration:

```
α = 0.03:  Very spiky — one element gets most of the probability
           Good for Go (19×19 = 361 moves, need focused noise)
α = 0.3:   Moderate spread
           Good for Connect Four (7 moves)
α = 10.0:  Nearly uniform — all elements get similar probability
```

We use α = 0.3 for Connect Four's 7-column action space.

## Part 9: Pure MCTS vs AlphaZero MCTS

| Aspect | Pure MCTS | AlphaZero MCTS |
|--------|-----------|----------------|
| Priors | Uniform (all moves equal) | Network policy (informed) |
| Evaluation | Random rollout (play to end) | Network value (instant) |
| Speed per simulation | Slow (rollout plays ~20 random moves) | Fast (one network forward pass) |
| Quality per simulation | Low (random play is noisy) | High (trained network) |
| Our results at 50 sims | 100% vs random | 100% vs random AND 80% vs pure MCTS |

The network helps in two ways:
1. **Better priors** guide the search toward promising moves immediately, instead of wasting simulations discovering that bad moves are bad.
2. **Better evaluation** gives accurate position assessments without the noise of random rollouts.

Together, these make each simulation ~10x more informative, which is why 50 network-guided simulations outperform 50 random-rollout simulations.

## Part 10: The Complete Search Function

```python
def search(self, game):
    root = MCTSNode(prior=0.0)
    root.game_state = game.clone()

    self._expand(root)

    if self.network is not None:
        self._add_dirichlet_noise(root)

    for _ in range(self.num_simulations):
        node = root
        path = [node]

        # SELECT
        while node.is_expanded() and not node.game_state.done:
            action, node = self._select_child(node)
            path.append(node)

        # EVALUATE
        if node.game_state.done:
            value = -node.game_state.get_result(node.game_state.current_player)
        else:
            self._expand(node)
            if self.network is not None:
                value = self._network_evaluate(node)
            else:
                value = self._rollout(node.game_state.clone())

        # BACKPROPAGATE
        self._backpropagate(path, value)

    # Extract policy from visit counts
    action_probs = np.zeros(COLS, dtype=np.float32)
    for action, child in root.children.items():
        action_probs[action] = child.visit_count
    total = action_probs.sum()
    if total > 0:
        action_probs /= total

    return action_probs
```

### Line-by-Line

```python
root = MCTSNode(prior=0.0)
root.game_state = game.clone()
```
- Create a fresh tree for this search. Each call to `search()` builds a new tree from scratch. We don't reuse trees between moves (though you could — that's an optimization called "tree reuse").
- Clone the game so the search doesn't modify the real state.

```python
self._expand(root)
```
- Immediately expand the root — create children for all legal moves. Every search starts with the root fully expanded.

```python
if self.network is not None:
    self._add_dirichlet_noise(root)
```
- Add exploration noise to root priors. Only in AlphaZero mode (not pure MCTS, which already has uniform priors).

```python
while node.is_expanded() and not node.game_state.done:
    action, node = self._select_child(node)
    path.append(node)
```
- Walk down the tree following UCB until we reach a leaf (unexpanded node) or a terminal position (game over). Track the path for backpropagation.

```python
if node.game_state.done:
    value = -node.game_state.get_result(node.game_state.current_player)
```
- If the game is already over at this node, use the actual result. No need for evaluation.
- The negation: `get_result` returns from `current_player`'s perspective, but we need it from the parent's perspective (who just moved).

```python
self._backpropagate(path, value)
```
- Send the value back up. Every node on the path gets updated.

## Summary

MCTS is a four-step loop that builds a search tree focused on the most promising branches:

```
SELECT:     UCB = Q(s,a) + c × P(s,a) × √N(parent) / (1+N(child))
            Balance exploitation (high Q) with exploration (low visits, high prior)

EXPAND:     Create children with priors from network (or uniform for pure MCTS)

EVALUATE:   Network value estimate (AlphaZero) or random rollout (pure MCTS)

BACKPROP:   Update visit counts and value sums along the path
            Negate value at each level (alternating players)
```

| Concept | What | Why |
|---------|------|-----|
| UCB formula | Balances exploit/explore | Don't get stuck on one move, don't waste time on bad moves |
| Visit counts as policy | Most-visited = best | More robust than Q-values (incorporates confidence) |
| Temperature | Controls move randomness | Diverse training data (high T) vs optimal play (low T) |
| Dirichlet noise | Random noise at root | Prevents the network from dominating root exploration |
| Value negation | Flip sign each level | Each node sees values from its own player's perspective |

## What's Next

In [Chapter 3](03_network.md), we build the neural network that replaces random rollouts with intelligent evaluation. It's a ResNet with two heads — one predicting which moves are promising (policy), one predicting who's winning (value). The same architecture DeepMind used, scaled to fit Connect Four.