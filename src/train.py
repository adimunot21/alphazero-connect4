"""
AlphaZero Training Loop.

The complete cycle:
  1. Self-play: generate games using MCTS + current network
  2. Train: update the network on the self-play data
  3. Evaluate: pit the new network against the old one
  4. Repeat

The network improves because:
  - MCTS + network produces better moves than the network alone (search helps)
  - Training the network to match MCTS policies distills the search into the network
  - Stronger network → better MCTS → better training data → stronger network
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque

from src.game import ConnectFour, COLS
from src.network import AlphaZeroNetwork
from src.self_play import generate_self_play_data
from src.mcts import get_mcts_action


# -----------------------------------------------------------------------
# Training Dataset
# -----------------------------------------------------------------------

class AlphaZeroDataset(Dataset):
    """
    Dataset of self-play positions.

    Each example: (board_nn_input, mcts_policy, game_result)

    Data augmentation: Connect Four is horizontally symmetric.
    Flipping the board left-right doubles our training data for free.
    """

    def __init__(self, examples, augment=True):
        self.examples = list(examples)
        self.augment = augment

    def __len__(self):
        return len(self.examples) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        # First half: original, second half: horizontally flipped
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
            state = torch.flip(state, dims=[2])  # flip columns (dim 2 = width)
            pi = torch.flip(pi, dims=[0])         # flip policy (reverse column order)

        return state, pi, v


# -----------------------------------------------------------------------
# Training Config
# -----------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Self-play
    "num_games_per_iteration": 100,   # games per self-play round
    "num_simulations": 50,            # MCTS simulations per move
    "temperature_threshold": 15,       # moves before switching to greedy

    # Training
    "num_iterations": 20,              # total training iterations
    "epochs_per_iteration": 10,        # training epochs on each batch of data
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 1e-4,             # L2 regularization

    # Replay buffer
    "max_buffer_size": 50000,          # keep this many recent positions

    # Evaluation
    "eval_games": 20,                  # games to play in evaluation
    "eval_simulations": 50,            # MCTS sims for evaluation

    # Network
    "num_res_blocks": 5,
    "num_filters": 64,
}


# -----------------------------------------------------------------------
# Training Step
# -----------------------------------------------------------------------

def train_on_data(network, optimizer, dataset, epochs, batch_size, device):
    """
    Train the network on self-play data.

    Two losses:
    - Policy loss: cross-entropy between network policy and MCTS policy
      "Learn to suggest the same moves that MCTS found through search"
    - Value loss: MSE between network value and game outcome
      "Learn to predict who will win from each position"
    """
    network.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_policy_loss = 0
    total_value_loss = 0
    total_batches = 0

    for epoch in range(epochs):
        for states, target_pis, target_vs in loader:
            states = states.to(device)
            target_pis = target_pis.to(device)
            target_vs = target_vs.squeeze(-1).to(device)

            # Forward pass
            policy_logits, values = network(states)

            # Policy loss: cross-entropy with MCTS policy
            # We use log_softmax + manual dot product instead of F.cross_entropy
            # because target_pis are soft labels (probabilities), not hard labels
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(target_pis * log_probs, dim=1).mean()

            # Value loss: MSE between predicted value and game outcome
            value_loss = F.mse_loss(values, target_vs)

            # Total loss
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_batches += 1

    avg_p = total_policy_loss / max(total_batches, 1)
    avg_v = total_value_loss / max(total_batches, 1)
    return avg_p, avg_v


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate_against_random(network, num_games=20, num_simulations=50):
    """Play AlphaZero (network + MCTS) against a random opponent."""
    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        game = ConnectFour()
        # Alternate who goes first
        az_player = 1 if i % 2 == 0 else -1

        while not game.done:
            if game.current_player == az_player:
                action, _ = get_mcts_action(game, num_simulations=num_simulations,
                                            temperature=0, network=network)
            else:
                legal = game.get_legal_moves()
                action = np.random.choice(legal)
            game.make_move(action)

        result = game.get_result(az_player)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


def evaluate_against_pure_mcts(network, num_games=20, num_simulations=50):
    """Play AlphaZero against pure MCTS (no network)."""
    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        game = ConnectFour()
        az_player = 1 if i % 2 == 0 else -1

        while not game.done:
            if game.current_player == az_player:
                # AlphaZero: network + MCTS
                action, _ = get_mcts_action(game, num_simulations=num_simulations,
                                            temperature=0, network=network)
            else:
                # Pure MCTS: no network
                action, _ = get_mcts_action(game, num_simulations=num_simulations,
                                            temperature=0, network=None)
            game.make_move(action)

        result = game.get_result(az_player)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    return wins, losses, draws


# -----------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------

def train_alphazero(config=None, device="cpu"):
    """
    Full AlphaZero training pipeline.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    print("=" * 60)
    print("AlphaZero Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Iterations: {cfg['num_iterations']}")
    print(f"Games per iteration: {cfg['num_games_per_iteration']}")
    print(f"MCTS simulations: {cfg['num_simulations']}")
    print(f"Training epochs per iteration: {cfg['epochs_per_iteration']}")
    print()

    # Initialize network and optimizer
    network = AlphaZeroNetwork(
        num_res_blocks=cfg["num_res_blocks"],
        num_filters=cfg["num_filters"],
    ).to(device)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    n_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {n_params:,}")

    # Replay buffer — keeps recent self-play data
    replay_buffer = deque(maxlen=cfg["max_buffer_size"])

    # Training history
    history = {
        "policy_loss": [],
        "value_loss": [],
        "vs_random": [],
        "vs_mcts": [],
    }

    # Initial evaluation (untrained network)
    print("\nInitial evaluation (untrained)...")
    w, l, d = evaluate_against_random(network, num_games=10, num_simulations=cfg["eval_simulations"])
    print(f"  vs Random: W={w} L={l} D={d} ({w/(w+l+d):.0%} win rate)")

    t0 = time.time()

    for iteration in range(1, cfg["num_iterations"] + 1):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{cfg['num_iterations']}")
        print(f"{'='*60}")

        # --- 1. Self-play ---
        print(f"\n1. Self-play ({cfg['num_games_per_iteration']} games)...")
        network.eval()
        new_examples = generate_self_play_data(
            network,
            num_games=cfg["num_games_per_iteration"],
            num_simulations=cfg["num_simulations"],
            temperature_threshold=cfg["temperature_threshold"],
        )
        replay_buffer.extend(new_examples)
        print(f"   Buffer size: {len(replay_buffer):,} positions")

        # --- 2. Train ---
        print(f"\n2. Training ({cfg['epochs_per_iteration']} epochs on {len(replay_buffer):,} positions)...")
        dataset = AlphaZeroDataset(list(replay_buffer), augment=True)
        network.to(device)
        policy_loss, value_loss = train_on_data(
            network, optimizer, dataset,
            epochs=cfg["epochs_per_iteration"],
            batch_size=cfg["batch_size"],
            device=device,
        )
        print(f"   Policy loss: {policy_loss:.4f}")
        print(f"   Value loss:  {value_loss:.4f}")
        history["policy_loss"].append(policy_loss)
        history["value_loss"].append(value_loss)

        # --- 3. Evaluate ---
        print(f"\n3. Evaluation...")
        network.eval()
        network.to("cpu")  # MCTS runs on CPU

        w, l, d = evaluate_against_random(
            network, num_games=cfg["eval_games"],
            num_simulations=cfg["eval_simulations"],
        )
        wr_random = w / max(w + l + d, 1)
        print(f"   vs Random:    W={w} L={l} D={d} ({wr_random:.0%})")
        history["vs_random"].append(wr_random)

        # vs pure MCTS every 5 iterations (it's slow)
        if iteration % 5 == 0 or iteration == cfg["num_iterations"]:
            w, l, d = evaluate_against_pure_mcts(
                network, num_games=10,
                num_simulations=cfg["eval_simulations"],
            )
            wr_mcts = w / max(w + l + d, 1)
            print(f"   vs Pure MCTS: W={w} L={l} D={d} ({wr_mcts:.0%})")
            history["vs_mcts"].append((iteration, wr_mcts))

        elapsed = time.time() - iter_start
        total_elapsed = time.time() - t0
        print(f"\n   Iteration time: {elapsed:.0f}s | Total: {total_elapsed:.0f}s")

        # Save checkpoint
        if iteration % 5 == 0 or iteration == cfg["num_iterations"]:
            os.makedirs("checkpoints", exist_ok=True)
            path = f"checkpoints/alphazero_iter{iteration}.pt"
            torch.save({
                "iteration": iteration,
                "model_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "history": history,
            }, path)
            print(f"   Saved checkpoint: {path}")

    # --- Final summary ---
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {total_time:.0f}s total")
    print(f"{'='*60}")
    print(f"Final vs Random: {history['vs_random'][-1]:.0%}")
    if history["vs_mcts"]:
        print(f"Final vs Pure MCTS: {history['vs_mcts'][-1][1]:.0%}")

    return network, history


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_training(history, save_path="notebooks/training_progress.png"):
    """Plot training metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Losses
    ax = axes[0]
    iters = range(1, len(history["policy_loss"]) + 1)
    ax.plot(iters, history["policy_loss"], "b-", linewidth=2, label="Policy loss")
    ax.plot(iters, history["value_loss"], "r-", linewidth=2, label="Value loss")
    ax.set_title("Training Losses")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Win rate vs random
    ax = axes[1]
    ax.plot(iters, history["vs_random"], "g-o", linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Win Rate vs Random")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Win rate vs pure MCTS
    ax = axes[2]
    if history["vs_mcts"]:
        mcts_iters, mcts_wr = zip(*history["vs_mcts"])
        ax.plot(mcts_iters, mcts_wr, "m-o", linewidth=2, markersize=8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% (even)")
    ax.set_title("Win Rate vs Pure MCTS")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("AlphaZero Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Quick local test with minimal settings
    # Full training should be done on Colab
    print("Running quick local test (2 iterations, 10 games each)...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    network, history = train_alphazero(
        config={
            "num_iterations": 2,
            "num_games_per_iteration": 10,
            "num_simulations": 25,
            "epochs_per_iteration": 5,
            "eval_games": 10,
            "eval_simulations": 25,
        },
        device=device,
    )

    plot_training(history)
    print("\nLocal test complete! Full training should be run on Colab.")