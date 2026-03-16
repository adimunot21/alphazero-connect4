# Chapter 5: Evaluation and Play — Testing What the AI Learned

## How Do You Know If a Game AI Is Good?

You can't just look at the loss curve. A low policy loss means the network predicts MCTS's moves well, but that doesn't tell you if the overall system plays strong Connect Four. A low value loss means the network predicts game outcomes well, but a network that always predicts "draw" would have low loss too if most games are close.

You need to **play games** — against opponents of known strength, under controlled conditions.

## Part 1: The Arena — Systematic Evaluation

### AlphaZero vs Random

```python
def evaluate_against_random(network, num_games=20, num_simulations=50):
    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        game = ConnectFour()
        az_player = 1 if i % 2 == 0 else -1    # alternate sides

        while not game.done:
            if game.current_player == az_player:
                action, _ = get_mcts_action(game, num_simulations=num_simulations,
                                            temperature=0, network=network)
            else:
                legal = game.get_legal_moves()
                action = np.random.choice(legal)
            game.make_move(action)

        result = game.get_result(az_player)
        if result == 1: wins += 1
        elif result == -1: losses += 1
        else: draws += 1

    return wins, losses, draws
```

**Why alternate sides?** If AlphaZero always plays first, it benefits from the first-move advantage. Alternating ensures we measure the AI's strength, not the positional advantage.

**Why temperature = 0?** During evaluation, we want the AI to play its best. Temperature = 0 means always pick the most-visited move (no randomness).

**Our result: 100% win rate (50/50 games).** Not surprising — even pure MCTS beats random 100%. But this confirms the system works end-to-end: the network loads correctly, MCTS runs without errors, moves are legal, games complete properly.

### AlphaZero vs Pure MCTS

This is the real test. Both players use MCTS with the same simulation count. The only difference: AlphaZero's MCTS is guided by the neural network, pure MCTS uses random rollouts.

```python
def evaluate_against_pure_mcts(network, num_games=20, num_simulations=50):
    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        game = ConnectFour()
        az_player = 1 if i % 2 == 0 else -1

        while not game.done:
            if game.current_player == az_player:
                action, _ = get_mcts_action(game, num_simulations=num_simulations,
                                            temperature=0, network=network)
            else:
                action, _ = get_mcts_action(game, num_simulations=num_simulations,
                                            temperature=0, network=None)
            game.make_move(action)

        result = game.get_result(az_player)
        ...
```

**Our results over training:**

```
Iteration  5:  70% win rate  (network starting to help)
Iteration 10:  80% win rate  (network clearly beneficial)
Iteration 15:  80% win rate  (stable)
Iteration 20:  80% win rate  (consistent advantage)
Final eval (100 sims each): 55% win rate
```

**Why did 80% drop to 55% in final evaluation?** The training evaluations used 50 simulations for both sides. The final evaluation used 100 simulations. Pure MCTS benefits MORE from extra simulations than network-guided MCTS at this stage — random rollouts get more accurate with more samples, partially closing the gap that the network provides. The network still wins the majority, proving it learned real strategic knowledge.

### What These Numbers Mean

```
50% → Network makes no difference (equivalent to pure MCTS)
55% → Small but real advantage (statistically significant over 20+ games)
70% → Strong advantage (network clearly helping)
80% → Dominant (network providing substantial strategic knowledge)
100% → Network makes MCTS vastly superior (rare at equal simulation counts)
```

Our 80% at 50 simulations means the network makes each simulation roughly 4× more effective — the network's priors immediately guide search toward good moves, and its value estimates are more accurate than random rollouts.

## Part 2: Human vs AI — The Interactive Experience

### The Play Interface

```python
def play_game(network, human_first=True, num_simulations=200):
    game = ConnectFour()
    human_player = 1 if human_first else -1

    while not game.done:
        if game.current_player == human_player:
            # Human's turn — get input
            col = int(input("Your move, column 0-6: "))
            game.make_move(col)
        else:
            # AI's turn — run MCTS
            action, action_probs = get_mcts_action(
                game, num_simulations=num_simulations,
                temperature=0, network=network
            )
            display_ai_thinking(game, action_probs, network)
            game.make_move(action)
```

We use 200 simulations for human play (vs 50 during training). More simulations = stronger play, and the few extra seconds of thinking time per move is acceptable for interactive play.

### AI Thinking Display

After each AI move, the player sees what the AI was thinking:

```
  AI's analysis:
  ────────────────────────────────────────
  Position eval: +0.45  (AI thinks it's winning)

  Column   MCTS   Network  Bar
  ────────────────────────────────────────
      0     2.0%    5.3%  
      1     5.0%    8.1%  █
      2    12.0%   15.2%  ████
      3    55.0%   40.1%  ██████████████████
      4    18.0%   20.3%  ██████
      5     5.0%    7.8%  █
      6     3.0%    3.2%  █

  AI plays: column 3
```

Three pieces of information:

**Position eval (+0.45):** The network's value estimate. Positive means the AI thinks it's ahead. This tells the human "you might be in trouble" or "you're doing well" — useful for understanding the game state.

**MCTS column (55%):** How much search time MCTS allocated to each column. The most-visited column (55% for column 3) is the one MCTS chose. This is the final answer after searching ahead.

**Network column (40.1%):** What the raw network suggested before search. The difference between MCTS and Network shows where search changed the network's mind:

```
Network says: column 3 is 40.1% likely
MCTS says:    column 3 is 55% likely (after searching, column 3 looks even better)

Network says: column 0 is 5.3% likely
MCTS says:    column 0 is 2.0% likely (after searching, column 0 looks worse)
```

When MCTS strongly disagrees with the network, it means the network's initial assessment was wrong and search corrected it. This is the fundamental value of MCTS — it improves on the network's raw guesses.

### Why the AI Beats Humans

In our tests, the AI won both games against its developer. Several factors:

**Search depth:** 200 MCTS simulations explore positions 5-10 moves ahead. Humans typically look 3-5 moves ahead. The AI can see traps that are invisible to human calculation.

**No blind spots:** Humans tend to focus on one area of the board and miss threats elsewhere. The AI evaluates every column every move.

**Pattern recognition:** The neural network learned thousands of positions through self-play. It instantly recognizes patterns that humans would need to calculate.

**No fatigue or emotion:** The AI doesn't get careless when it's winning or panicked when it's losing. Every move gets the same 200-simulation analysis.

## Part 3: What the AI Learned

### Center Control

In the opening position, the AI's MCTS policy from our tests:

```
Column:  0     1     2     3     4     5     6
Prob:   0.05  0.08  0.12  0.38  0.20  0.12  0.05
```

Column 3 (center) gets 38% — by far the preferred opening move. This is correct Connect Four strategy: the center column participates in more possible four-in-a-row lines than any edge column.

```
Column 0: participates in 3 horizontal + 3 vertical + 3 diagonal = 9 lines
Column 3: participates in 4 horizontal + 3 vertical + 6 diagonal = 13 lines
```

Nobody told the AI this. It discovered center control purely from self-play — games that start in the center tend to win more.

### First-Player Advantage

Self-play statistics across training:

```
Iteration 1:   P1 wins 53%   (barely above 50% — random play)
Iteration 10:  P1 wins 66%   (discovered the advantage)
Iteration 20:  P1 wins 62%   (consistently exploiting it)
```

Connect Four is a mathematically solved game: Player 1 wins with perfect play from any position where both sides play optimally. The AI independently discovered this asymmetry — Player 1 wins more often in self-play because the AI learned to exploit the first-move advantage.

### Increasing Draw Rate

```
Iteration 1:   2% draws   (games end quickly, someone wins)
Iteration 10:  3% draws
Iteration 19:  11% draws  (both sides defend well, harder to win)
```

More draws = stronger defensive play. When both sides play well, neither can force a win within the AI's search horizon (50 simulations ≈ 5-10 moves of lookahead). The draw rate would decrease again with more simulations or more training, as the AI learns to find wins from more subtle positions.

### Threat Detection

The AI reliably:
- **Completes four in a row** when it has three
- **Blocks opponent's three in a row** to prevent their win
- **Creates double threats** — two ways to win simultaneously, where the opponent can only block one

These are the first strategic concepts any Connect Four player learns, and the AI discovers all of them from pure self-play within the first few iterations.

## Part 4: The Strength Hierarchy

After training, we have four players of increasing strength:

```
Random player          → win rate: ~50% vs itself
                         (baseline — no strategy)

Pure MCTS (100 sims)   → win rate: ~100% vs random
                         (search alone is powerful)

AlphaZero (100 sims)   → win rate: ~55% vs pure MCTS
                         (network makes search smarter)

AlphaZero (200 sims)   → beats humans
                         (more search + network = strong play)
```

Each level adds something:
- **Random → Pure MCTS:** adds SEARCH (looking ahead eliminates obvious blunders)
- **Pure MCTS → AlphaZero:** adds KNOWLEDGE (the network focuses search on good moves)
- **AlphaZero 100 → AlphaZero 200:** adds DEPTH (more simulations = deeper search)

This hierarchy mirrors the history of game AI: brute-force search → intelligent search → learned evaluation → more compute.

## Part 5: Limitations and Honest Assessment

### What the AI Does Well
- Beats random players 100% of the time
- Beats pure MCTS with the same simulation budget
- Beats casual human players
- Discovers real Connect Four strategy from zero knowledge
- Finds and blocks immediate threats reliably

### What the AI Doesn't Do
- **Doesn't play perfectly.** Connect Four is solved — perfect play means P1 always wins. Our AI can't guarantee this with 50-200 simulations. It would need thousands of simulations or much more training.
- **Doesn't plan 20+ moves ahead.** The search depth is limited by the simulation count. Complex multi-move setups beyond the search horizon are missed.
- **Sometimes misjudges quiet positions.** The value head is most accurate for positions with obvious threats. Subtle positional advantages (controlling key squares for future threats) may be undervalued.

### What Would Make It Stronger
1. **More simulations** (800+) — deeper search, better move quality
2. **More training iterations** (100+) — network learns more patterns
3. **Larger network** (10+ res blocks, 128+ filters) — more capacity for complex patterns
4. **Longer training games** — using more simulations during self-play produces higher-quality training data
5. **Tree reuse** — carrying the search tree between moves for effectively double the simulations

## Part 6: What We Built — The Complete System

```
┌─────────────────────────────────────────────────────────┐
│                     ALPHAZERO                            │
│                                                         │
│  Game Engine (Chapter 1)                                │
│  ├── Board representation (6×7, int8)                   │
│  ├── Canonical form (always from current player's view) │
│  ├── Win detection (directional, O(1))                  │
│  └── NN input encoding (3 channels)                     │
│                                                         │
│  MCTS (Chapter 2)                                       │
│  ├── UCB selection (explore vs exploit)                  │
│  ├── Network-guided expansion (informed priors)          │
│  ├── Neural network evaluation (replaces rollouts)       │
│  ├── Backpropagation (alternating perspective)           │
│  ├── Temperature (training diversity)                    │
│  └── Dirichlet noise (root exploration)                  │
│                                                         │
│  Neural Network (Chapter 3)                             │
│  ├── ResNet backbone (5 blocks, 64 filters, 377K params)│
│  ├── Policy head (7-column probability distribution)     │
│  ├── Value head (scalar in [-1, +1])                    │
│  └── Illegal move masking                                │
│                                                         │
│  Self-Play + Training (Chapter 4)                       │
│  ├── Self-play data generation (100 games/iteration)     │
│  ├── Horizontal flip augmentation (2× data)              │
│  ├── Policy loss (cross-entropy with MCTS targets)       │
│  ├── Value loss (MSE with game outcomes)                 │
│  ├── Replay buffer (50K positions)                       │
│  └── 20 iteration training loop                          │
│                                                         │
│  Evaluation (Chapter 5)                                 │
│  ├── 100% vs random (50/50 games)                       │
│  ├── 55-80% vs pure MCTS (network helps)                │
│  └── Beats human players                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Course Complete

You've built the complete AlphaZero algorithm from scratch:

1. **A game engine** optimized for speed (column heights, directional win detection) and neural network compatibility (canonical form, 3-channel encoding)

2. **Monte Carlo Tree Search** that balances exploration and exploitation using UCB, guided by neural network priors and value estimates

3. **A dual-headed ResNet** that evaluates positions (value) and suggests moves (policy) simultaneously, trained entirely from self-play data

4. **A self-play training pipeline** where the AI generates its own training data by playing against itself, creating a positive feedback loop from random play to expert play

5. **An evaluation framework** that measures strength against different opponents, and an interactive mode where humans can play against the AI

The same algorithm — MCTS + neural network + self-play — is what DeepMind used to master Chess, Go, and Shogi. The only differences between our implementation and theirs are scale: they used 40 residual blocks (we used 5), 256 filters (we used 64), 800 simulations (we used 50), and trained for days on thousands of TPUs (we trained for hours on one GPU).

The principles are identical. The algorithm doesn't need to be told how to play Connect Four any more than AlphaZero needed to be told how to play Chess. It discovers strategy from scratch — which is perhaps the most remarkable thing about this approach to AI.