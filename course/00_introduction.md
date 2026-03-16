# Chapter 0: Introduction — How Computers Play Games

## The Problem

You want a computer to play Connect Four well. How?

The simplest approach: write rules by hand. "If you have 3 in a row, complete it. If the opponent has 3 in a row, block. Otherwise, play in the center." This works for simple situations but misses complex strategy — double threats, setups that pay off 5 moves later, sacrificing position for long-term advantage.

The deeper approach: let the computer figure out the strategy itself. Give it the rules of the game, let it play millions of games against itself, and let it discover what works. This is AlphaZero.

## A Brief History of Game AI

### 1997: Deep Blue Beats Kasparov at Chess

IBM's Deep Blue used two things:
- **Search**: explore millions of possible future moves (the game tree)
- **Hand-crafted evaluation**: human experts wrote a function that scores board positions ("a knight is worth 3 points, control of the center is worth 0.5 points...")

Deep Blue searched 200 million positions per second, but every position was evaluated by rules written by human chess experts. The computer was fast but not creative — it could only recognize patterns that humans told it to look for.

### 2016: AlphaGo Beats Lee Sedol at Go

Go has 10^170 possible positions (chess has 10^47). You can't search them all. Google DeepMind's AlphaGo replaced the hand-crafted evaluation with a **neural network** that learned to evaluate positions from human expert games. It also used **Monte Carlo Tree Search** instead of exhaustive search — sampling promising moves rather than trying everything.

AlphaGo still needed human expert games to learn from.

### 2017: AlphaZero — No Human Knowledge

AlphaZero eliminated the last piece of human input. No expert games, no hand-crafted evaluation, no opening books. Just:
1. The rules of the game
2. A neural network (randomly initialized — knows nothing)
3. Self-play (the AI plays against itself)

Starting from random play, AlphaZero discovered chess, Go, and Shogi strategy from scratch — and surpassed all previous AI systems in each game within 24 hours of training.

This is what we're building. Same algorithm, applied to Connect Four.

## The Three Components

AlphaZero has three parts that work together:

### 1. The Game Tree

Every game can be represented as a tree. The root is the current position. Each branch is a possible move. Each node is a resulting position.

```
         Current position
        ╱     |      ╲
    Col 0   Col 3   Col 6      ← your possible moves
     ╱╲      |       ╱╲
   ...  ... Col 2  ...  ...    ← opponent's responses
              |
             ...               ← your responses to those
```

Connect Four has 7 possible moves per turn (7 columns). After 2 moves (one per player), there are up to 7 × 7 = 49 positions. After 10 moves, roughly 7^10 ≈ 280 million positions. The full game tree has about 4.5 trillion positions.

You can't search them all. You need to be smart about which branches to explore.

### 2. Monte Carlo Tree Search (MCTS)

MCTS is the "smart search" algorithm. Instead of exploring every branch (impossible) or evaluating only the top few (might miss something), MCTS uses **random sampling** to estimate which moves are best.

The core idea: if you play random games from a position and one move leads to winning 70% of the time while another wins 30%, the first move is probably better. MCTS refines this estimate by playing more games from promising positions.

MCTS has been used in game AI since the mid-2000s. What AlphaZero added was using a neural network to replace the random games with intelligent evaluation.

### 3. The Neural Network

The neural network takes a board position and outputs two things:

**Policy**: "Which moves look promising?" — a probability distribution over columns.
```
[0.05, 0.08, 0.12, 0.35, 0.20, 0.12, 0.08]
  Col0  Col1  Col2  Col3  Col4  Col5  Col6

"Column 3 looks best (35%), columns 2 and 4 are decent..."
```

**Value**: "Who's winning?" — a single number from -1 to +1.
```
+0.6  →  "I'm probably winning from this position"
-0.3  →  "I'm slightly behind"
 0.0  →  "It's roughly even"
```

An untrained network outputs uniform policy (all columns equally likely) and near-zero value (no idea who's winning). Through self-play training, it learns to evaluate positions accurately.

## How the Three Components Work Together

### During a single move:

```
1. Current board position
        │
        ▼
2. MCTS runs 50-200 simulations:
   For each simulation:
     a. Walk down the tree, picking promising branches
        (guided by the network's POLICY — explore moves the network likes)
     b. Reach a new position
     c. Ask the network: "How good is this position?"
        (the network's VALUE replaces random rollouts)
     d. Send the answer back up the tree
        │
        ▼
3. After all simulations, count how many times each move was explored
   The most-explored move is the best move
        │
        ▼
4. Play that move
```

MCTS uses the network to guide its search AND to evaluate positions. The network makes MCTS smarter. But MCTS also makes the network's output better — searching 200 positions ahead is more accurate than the network's raw guess.

### During training:

```
1. Play a full game of self-play (both sides use MCTS + network)
   Record every position, the MCTS move distribution, and who won
        │
        ▼
2. Train the network:
   - Policy target: match MCTS's move distribution
     (MCTS searched ahead and found better moves than the raw network)
   - Value target: predict who actually won the game
        │
        ▼
3. The network gets stronger
   → MCTS guided by the stronger network plays better
   → Better games produce better training data
   → The network gets even stronger
   → Repeat
```

This positive feedback loop is the heart of AlphaZero. The system bootstraps from zero knowledge to expert play.

## Why Connect Four?

Connect Four is the perfect AlphaZero learning project:

**Complex enough to be interesting.** 4.5 trillion positions, requiring real strategy — center control, diagonal setups, double threats, long-term planning. A neural network genuinely helps.

**Simple enough to train on a free GPU.** 6×7 board, 7 possible moves per turn, games last 15-25 moves. We can train a strong AI in a few hours on a single GPU. AlphaZero for Go required 5,000 TPUs for 3 days.

**Solved game.** Connect Four was mathematically solved in 1988 — Player 1 wins with perfect play. This gives us a ground truth to compare against. As our AI improves, we should see Player 1 winning more of the self-play games.

**Visual and intuitive.** You can look at a Connect Four board and understand the position. When the AI makes a move, you can often see why — "it's setting up a diagonal" or "it's blocking my three in a row." This makes the learning process tangible.

## What You'll Build

| Phase | What | Key Concept |
|-------|------|-------------|
| 1 | Connect Four engine | Board representation, canonical form, fast win detection |
| 2 | Monte Carlo Tree Search | UCB, game tree exploration, random rollouts |
| 3 | Policy + Value network | ResNet, dual-headed architecture, illegal move masking |
| 4-5 | Self-play + Training | The AlphaZero loop, self-improvement cycle |
| 6 | Human vs AI | Interactive play, seeing the AI's thinking |

Each component builds on the previous one. MCTS needs the game engine. The network needs the game engine's board representation. Self-play needs MCTS and the network. Training needs self-play data.

## Setup

### Create Environment

```bash
conda create -n alphazero python=3.11 -y
conda activate alphazero
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2" matplotlib tqdm
```

### Project Structure

```bash
mkdir -p ~/projects/alphazero-connect4/{src,checkpoints,notebooks,course}
cd ~/projects/alphazero-connect4
```

```
alphazero-connect4/
├── src/
│   ├── game.py         ← Connect Four engine
│   ├── mcts.py         ← Monte Carlo Tree Search
│   ├── network.py      ← Policy + Value neural network
│   ├── self_play.py    ← Self-play data generation
│   ├── train.py        ← Training loop
│   └── play.py         ← Human vs AI
├── checkpoints/        ← Saved model weights
├── notebooks/          ← Plots and analysis
└── course/             ← These course files
```

## What's Next

In [Chapter 1](01_game_engine.md), we build the Connect Four game engine — the foundation everything else rests on. The key challenge: representing the board in a way that's fast (MCTS calls the engine thousands of times per move), correct (every edge case handled), and neural-network-friendly (converting positions to tensors).