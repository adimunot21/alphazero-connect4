# AlphaZero for Connect Four

A from-scratch implementation of DeepMind's AlphaZero algorithm — the same system that mastered Chess, Go, and Shogi — applied to Connect Four. The AI learns to play purely through self-play, with zero human game knowledge.

## How It Works

```
┌──────────────────────────────────────────────────────┐
│                  THE ALPHAZERO LOOP                    │
│                                                       │
│   ┌─────────┐     guides      ┌───────────────┐      │
│   │ Neural  │───────────────▶│    MCTS        │      │
│   │ Network │                 │  (searches     │      │
│   │         │◀───────────────│   future moves) │      │
│   └─────────┘    trains on    └───────┬───────┘      │
│        ▲                              │               │
│        │                              │ picks moves   │
│        │         trains on            ▼               │
│        └─────────────────────  Self-Play Games        │
│                                                       │
│   Start: random network → random play                 │
│   End:   expert network → expert play                 │
└──────────────────────────────────────────────────────┘
```

Three components, each built from scratch:

1. **Monte Carlo Tree Search (MCTS)**: Explores thousands of future positions to find the best move. Uses UCB to balance exploration and exploitation in the game tree.

2. **Policy + Value Neural Network**: A ResNet that takes a board position and outputs a move probability distribution (policy) and a position evaluation (value). Replaces MCTS's random rollouts with intelligent evaluation.

3. **Self-Play Training**: The AI plays against itself, generating training data. MCTS (guided by the network) produces better move choices than the network alone. Training the network to match MCTS creates a positive feedback loop — from random play to expert play.

## Results

### Training Progression (20 iterations)
| Metric | Start | End |
|--------|-------|-----|
| Policy loss | 1.91 | 1.48 |
| Value loss | 0.48 | 0.36 |
| vs Random | 100% | 100% |
| vs Pure MCTS | 10% (network hurting) | 80% (network helping) |

### Final Evaluation
| Opponent | Games | Win Rate |
|----------|-------|----------|
| Random player | 50 | **100%** (50W-0L-0D) |
| Pure MCTS (100 sims) | 20 | **55%** (11W-9L-0D) |
| Human (the developer) | 2 | **100%** (2W-0L-0D) |

### What the AI Learned (from zero human input)
- **Center column control**: Prefers column 3 in the opening (38% probability)
- **First-player advantage**: P1 wins 60-68% of self-play games (Connect Four is a solved P1 win with perfect play)
- **Threat creation and blocking**: Finds forced wins and blocks opponent threats
- **Increasingly drawn games**: As both sides improve, draws increase from 0% to 11% of self-play games

## Architecture

### Neural Network (377K parameters)
```
Input: 3×6×7 (my pieces, opponent pieces, turn indicator)
  → ConvBlock (3→64 filters)
  → 5 × ResBlock (64 filters each, skip connections)
  → Policy Head: Conv→FC→7 probabilities (one per column)
  → Value Head: Conv→FC→FC→tanh scalar in [-1, +1]
```

### MCTS (50-200 simulations per move)
```
Repeat N times:
  SELECT:     Walk tree using UCB = Q(s,a) + c·P(s,a)·√N(parent)/(1+N(child))
  EXPAND:     Add children for all legal moves, priors from network policy
  EVALUATE:   Network value estimate (replaces random rollout)
  BACKPROP:   Send value up the tree, alternating sign per level
```

## Project Structure

```
alphazero-connect4/
├── src/
│   ├── game.py            ← Connect Four engine (board, moves, win detection)
│   ├── mcts.py            ← Monte Carlo Tree Search (pure + AlphaZero mode)
│   ├── network.py         ← Policy + Value ResNet (377K params)
│   ├── self_play.py       ← Self-play data generation
│   ├── train.py           ← AlphaZero training loop
│   └── play.py            ← Human vs AI interactive mode
├── notebooks/
│   └── training_progress.png
├── checkpoints/           ← Trained model (local only)
├── course/                ← Detailed written course
└── requirements.txt
```

## Setup

```bash
conda create -n alphazero python=3.11 -y
conda activate alphazero
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2" matplotlib tqdm
```

## Usage

```bash
# Play against the AI
python -m src.play

# Run game engine tests
python -m src.game

# Test MCTS (pure, no network — plays 50 games vs random)
python -m src.mcts

# Test network shapes
python -m src.network

# Train (quick local test — full training should be on GPU)
python -m src.train
```

## Training

Full training (20 iterations, 100 games/iteration) requires a GPU. Trained on Kaggle (T4 GPU) in ~4 hours. See `src/train.py` for the training configuration.

## Key Insights

1. **MCTS alone is strong.** Pure MCTS (random rollouts, no neural network) beats a random player 100% of the time. The search itself is powerful.

2. **The network makes MCTS smarter, not faster.** Both use 50 simulations, but network-guided MCTS spends those simulations on better moves (guided by the policy prior) and evaluates positions more accurately (value head vs random rollout).

3. **Self-play creates a curriculum.** Early games are random (easy training data). As the network improves, games get harder. The AI automatically generates training data at exactly the right difficulty level.

4. **Connect Four is a solved game.** Player 1 wins with perfect play. The AI independently discovers this — P1 wins increasingly dominate self-play as training progresses.

## Course: Learn AlphaZero From Scratch

This project includes a detailed written course explaining every concept and every line of code.

| Chapter | Topic | What You'll Learn |
|---------|-------|-------------------|
| [0: Introduction](course/00_introduction.md) | What is AlphaZero? | Game trees, minimax, why MCTS + neural networks |
| [1: Game Engine](course/01_game_engine.md) | Connect Four implementation | Board representation, canonical form, win detection |
| [2: MCTS](course/02_mcts.md) | Monte Carlo Tree Search | UCB, selection, expansion, rollout, backpropagation |
| [3: Neural Network](course/03_network.md) | Policy + Value ResNet | Dual-headed architecture, residual blocks, illegal move masking |
| [4: Self-Play & Training](course/04_training.md) | The AlphaZero loop | Self-play data, policy/value targets, the improvement cycle |
| [5: Evaluation & Play](course/05_evaluation.md) | Testing and playing | Arena evaluation, human vs AI, what the AI learned |

## Built With
- PyTorch (Conv2d, Linear, autograd)
- NumPy, Matplotlib
- No game AI libraries — MCTS, network, self-play, and training all from scratch
- Trained on Kaggle (T4 GPU)