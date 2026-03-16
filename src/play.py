"""
Human vs AlphaZero — Interactive Connect Four.

Play against the trained AI in your terminal.
The AI shows its thinking: which columns it considered,
its value estimate, and MCTS visit distribution.
"""

import torch
import numpy as np
from src.game import ConnectFour, COLS, render_board
from src.network import AlphaZeroNetwork
from src.mcts import MCTS, get_mcts_action


def load_model(checkpoint_path="checkpoints/alphazero_final.pt"):
    """Load trained network from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network = AlphaZeroNetwork()
    network.load_state_dict(ckpt["model_state"])
    network.eval()
    n_params = sum(p.numel() for p in network.parameters())
    print(f"Loaded model ({n_params:,} parameters)")
    return network


def display_ai_thinking(game, action_probs, network):
    """Show what the AI is thinking."""
    # Get raw network policy and value
    policy, value = network.predict(game)

    print(f"\n  AI's analysis:")
    print(f"  {'─' * 40}")

    # Value estimate
    if value > 0.3:
        assessment = "AI thinks it's winning"
    elif value < -0.3:
        assessment = "AI thinks you're winning"
    else:
        assessment = "AI thinks it's roughly even"
    print(f"  Position eval: {value:+.2f}  ({assessment})")

    # MCTS visit distribution vs raw network policy
    print(f"\n  {'Column':>8s}  {'MCTS':>8s}  {'Network':>8s}  {'Bar'}")
    print(f"  {'─' * 40}")
    for c in range(COLS):
        mcts_pct = action_probs[c] * 100
        net_pct = policy[c] * 100
        bar = "█" * int(mcts_pct / 3)
        legal = "  " if game.get_legal_moves_mask()[c] > 0 else " X"
        print(f"  {c:>5d}{legal}  {mcts_pct:>6.1f}%  {net_pct:>6.1f}%  {bar}")

    best = np.argmax(action_probs)
    print(f"\n  AI plays: column {best}")
    print()


def play_game(network, human_first=True, num_simulations=100):
    """Play one game of human vs AI."""
    game = ConnectFour()
    human_player = 1 if human_first else -1
    ai_player = -human_player

    human_symbol = "X" if human_player == 1 else "O"
    ai_symbol = "X" if ai_player == 1 else "O"

    print(f"\n{'═' * 50}")
    print(f"  You are {human_symbol} ({'first' if human_first else 'second'})")
    print(f"  AI is {ai_symbol} ({'second' if human_first else 'first'})")
    print(f"  AI uses {num_simulations} MCTS simulations per move")
    print(f"  Type a column number (0-6) to play")
    print(f"  Type 'q' to quit")
    print(f"{'═' * 50}\n")

    while not game.done:
        print(game)
        print()

        if game.current_player == human_player:
            # Human's turn
            while True:
                try:
                    inp = input(f"  Your move ({human_symbol}), column 0-6: ").strip()
                    if inp.lower() == 'q':
                        print("  Goodbye!")
                        return None
                    col = int(inp)
                    if col not in game.get_legal_moves():
                        print(f"  Column {col} is not a legal move. Try: {game.get_legal_moves()}")
                        continue
                    break
                except ValueError:
                    print("  Enter a number 0-6, or 'q' to quit")

            game.make_move(col)

        else:
            # AI's turn
            print(f"  AI is thinking ({num_simulations} simulations)...")
            action, action_probs = get_mcts_action(
                game, num_simulations=num_simulations,
                temperature=0, network=network
            )
            display_ai_thinking(game, action_probs, network)
            game.make_move(action)

    # Game over
    print(game)
    print()

    if game.winner == 0:
        print("  ╔═══════════════╗")
        print("  ║   IT'S A DRAW  ║")
        print("  ╚═══════════════╝")
    elif game.winner == human_player:
        print("  ╔════════════════╗")
        print("  ║   YOU WIN! 🎉  ║")
        print("  ╚════════════════╝")
    else:
        print("  ╔════════════════╗")
        print("  ║   AI WINS! 🤖  ║")
        print("  ╚════════════════╝")

    return game.winner


def main():
    """Main play loop."""
    print("╔════════════════════════════════════════════╗")
    print("║     CONNECT FOUR — AlphaZero Edition       ║")
    print("╚════════════════════════════════════════════╝")

    # Load model
    try:
        network = load_model()
    except FileNotFoundError:
        print("\nNo trained model found at checkpoints/alphazero_final.pt")
        print("Running with pure MCTS (no neural network)...")
        network = None

    # Game settings
    num_simulations = 200  # more sims = stronger but slower

    scores = {"human": 0, "ai": 0, "draw": 0}
    game_num = 0

    while True:
        game_num += 1
        # Alternate who goes first
        human_first = (game_num % 2 == 1)

        print(f"\n{'━' * 50}")
        print(f"  Game {game_num} | Score — You: {scores['human']}  AI: {scores['ai']}  Draws: {scores['draw']}")
        print(f"{'━' * 50}")

        human_player = 1 if human_first else -1
        result = play_game(network, human_first=human_first,
                          num_simulations=num_simulations)

        if result is None:  # player quit
            break
        elif result == human_player:
            scores["human"] += 1
        elif result == 0:
            scores["draw"] += 1
        else:
            scores["ai"] += 1

        # Play again?
        again = input("\n  Play again? (y/n): ").strip().lower()
        if again != 'y':
            break

    print(f"\n{'═' * 50}")
    print(f"  FINAL SCORE")
    print(f"  You: {scores['human']}  |  AI: {scores['ai']}  |  Draws: {scores['draw']}")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()