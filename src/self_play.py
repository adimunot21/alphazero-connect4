"""
Self-Play Data Generation.

The AI plays against itself using MCTS guided by the neural network.
Each game produces training data: (board_state, mcts_policy, game_outcome).

The key insight: MCTS produces BETTER moves than the raw network policy
because it searches ahead. Training the network to match MCTS's policy
makes the network stronger → which makes MCTS stronger → which produces
even better training data → positive feedback loop.
"""

import numpy as np
import torch
from src.game import ConnectFour, COLS
from src.mcts import MCTS


def self_play_game(network, num_simulations=50, temperature_threshold=15):
    """
    Play one complete game of self-play.

    Args:
        network: the current neural network
        num_simulations: MCTS simulations per move
        temperature_threshold: use temperature=1 for the first N moves (exploration),
                              then temperature→0 (exploitation)

    Returns:
        training_examples: list of (board_nn_input, mcts_policy, None)
            The result (None) gets filled in after the game ends.
    """
    game = ConnectFour()
    mcts = MCTS(network=network, num_simulations=num_simulations)
    history = []  # (nn_input, mcts_policy, current_player)

    move_num = 0
    while not game.done:
        # Run MCTS from current position
        action_probs = mcts.search(game)

        # Store training data (result filled in later)
        nn_input = game.get_nn_input()
        history.append((nn_input, action_probs, game.current_player))

        # Select move with temperature
        if move_num < temperature_threshold:
            # Early game: sample proportionally (explore diverse openings)
            move = np.random.choice(COLS, p=action_probs)
        else:
            # Late game: pick the best move (exploit)
            move = np.argmax(action_probs)

        game.make_move(move)
        move_num += 1

    # Game is over — fill in results
    training_examples = []
    for nn_input, policy, player in history:
        # Result from this position's player's perspective
        result = game.get_result(player)
        if result is None:
            result = 0
        training_examples.append((nn_input, policy, result))

    return training_examples


def generate_self_play_data(network, num_games=100, num_simulations=50,
                            temperature_threshold=15, verbose=True):
    """
    Generate a batch of self-play games.

    Returns:
        all_examples: list of (nn_input, mcts_policy, result)
    """
    all_examples = []
    results = {1: 0, -1: 0, 0: 0}

    for i in range(num_games):
        examples = self_play_game(network, num_simulations, temperature_threshold)
        all_examples.extend(examples)

        # Track the game result (from player 1's perspective)
        game_result = examples[0][2]  # first position is always player 1's turn
        results[game_result] += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"  Game {i+1}/{num_games}: "
                  f"{len(all_examples)} positions | "
                  f"P1 wins: {results[1]}, P2 wins: {results[-1]}, Draws: {results[0]}")

    if verbose:
        print(f"  Total: {len(all_examples)} training positions from {num_games} games")

    return all_examples


if __name__ == "__main__":
    from src.network import AlphaZeroNetwork

    print("=" * 50)
    print("Self-play test with untrained network")
    print("=" * 50)

    net = AlphaZeroNetwork()
    examples = generate_self_play_data(net, num_games=20, num_simulations=25)

    print(f"\nSample training example:")
    nn_input, policy, result = examples[0]
    print(f"  Board shape: {nn_input.shape}")
    print(f"  Policy: {[f'{p:.3f}' for p in policy]}")
    print(f"  Result: {result}")

    # Verify data quality
    for nn_input, policy, result in examples:
        assert nn_input.shape == (3, 6, 7), f"Bad shape: {nn_input.shape}"
        assert abs(policy.sum() - 1.0) < 1e-5, f"Policy doesn't sum to 1: {policy.sum()}"
        assert result in [-1, 0, 1], f"Bad result: {result}"

    print("\n✓ All training examples are valid!")