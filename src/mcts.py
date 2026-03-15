"""
Monte Carlo Tree Search (MCTS).

Two modes:
1. PURE MCTS (this phase): evaluates positions by playing random games to the end
2. ALPHAZERO MCTS (Phase 4): evaluates positions using a neural network

Both share the same tree structure and selection logic. The only difference
is what happens at leaf nodes — random rollout vs network evaluation.

The 4-step loop (repeated N times per move):

  SELECT:     Walk down the tree from root, at each node picking the child
              with highest UCB score (balances exploitation + exploration)

  EXPAND:     At a leaf node, create child nodes for all legal moves

  EVALUATE:   Determine the value of this new position
              Pure MCTS: play randomly until game over
              AlphaZero: ask the neural network

  BACKPROP:   Send the value back up the tree, updating visit counts
              and value estimates at every node along the path
"""

import math
import numpy as np
from src.game import ConnectFour, COLS


class MCTSNode:
    """
    One node in the search tree = one game position.

    Stores:
    - visit_count: how many times MCTS has visited this node
    - value_sum: total value accumulated (sum of all backpropagated results)
    - prior: the neural network's initial estimate of how good this move is
             (uniform for pure MCTS, network policy for AlphaZero)
    - children: dict mapping action → child MCTSNode
    """

    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior          # P(s,a) — prior probability from network
        self.children = {}          # action → MCTSNode
        self.game_state = None      # the ConnectFour state at this node

    @property
    def q_value(self):
        """Average value — exploitation signal."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        """A node is expanded if it has children."""
        return len(self.children) > 0


class MCTS:
    """
    Monte Carlo Tree Search engine.

    Can run in two modes:
    - network=None: pure MCTS with random rollouts
    - network=a neural net: AlphaZero-style MCTS with network evaluation
    """

    def __init__(self, network=None, num_simulations=100, c_puct=1.41,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        """
        Args:
            network: policy+value network (None for pure MCTS)
            num_simulations: how many times to run the select→evaluate→backprop loop
            c_puct: exploration constant in UCB formula
                    Higher = more exploration. 1.41 ≈ √2 is standard for pure MCTS.
                    AlphaZero uses ~1.0-2.0.
            dirichlet_alpha: noise parameter for root exploration (AlphaZero only)
            dirichlet_epsilon: fraction of noise to mix into root priors
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, game):
        """
        Run MCTS from the given game state. Returns visit count distribution
        over moves (the MCTS policy).

        Args:
            game: a ConnectFour instance (not modified)

        Returns:
            action_probs: numpy array of shape (COLS,) — visit count proportions
        """
        # Create root node
        root = MCTSNode(prior=0.0)
        root.game_state = game.clone()

        # Expand root
        self._expand(root)

        # Add Dirichlet noise to root priors (encourages exploration of all moves)
        if self.network is not None:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # === SELECT ===
            # Walk down the tree, picking the child with highest UCB
            while node.is_expanded() and not node.game_state.done:
                action, node = self._select_child(node)
                path.append(node)

            # === EVALUATE ===
            if node.game_state.done:
                # Game is over at this node — use actual result
                # Result from the perspective of the PARENT's current player
                # (the player who made the move to get here)
                value = -node.game_state.get_result(node.game_state.current_player)
            else:
                # Expand this leaf node
                self._expand(node)

                if self.network is not None:
                    # AlphaZero mode: ask the neural network
                    value = self._network_evaluate(node)
                else:
                    # Pure MCTS mode: random rollout
                    value = self._rollout(node.game_state.clone())

            # === BACKPROPAGATE ===
            # Send value back up the tree
            # Value alternates sign at each level (my win = your loss)
            self._backpropagate(path, value)

        # Extract policy from root visit counts
        action_probs = np.zeros(COLS, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        # Normalize to probabilities
        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def _select_child(self, node):
        """
        Pick the child with the highest UCB score.

        UCB(s, a) = Q(s,a) + c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))

        This balances:
        - Q(s,a): exploitation — moves that have proven good (high average value)
        - The second term: exploration — moves that haven't been tried much
          (high prior and/or low visit count)

        The exploration term shrinks as a child is visited more (denominator grows),
        naturally shifting from exploration to exploitation.
        """
        best_score = -float("inf")
        best_action = None
        best_child = None

        parent_visits = node.visit_count

        for action, child in node.children.items():
            # Q-value: average observed value (exploitation)
            q = child.q_value

            # Exploration bonus: high prior + low visits = high bonus
            u = (self.c_puct * child.prior *
                 math.sqrt(parent_visits) / (1 + child.visit_count))

            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand(self, node):
        """
        Create child nodes for all legal moves.

        If we have a neural network, use its policy output as priors.
        Otherwise, use uniform priors (all moves equally likely).
        """
        game = node.game_state
        legal_moves = game.get_legal_moves()

        if not legal_moves or game.done:
            return

        if self.network is not None:
            # Get network's policy for this position
            policy, _ = self._network_predict(game)
        else:
            # Uniform priors for pure MCTS
            policy = np.ones(COLS, dtype=np.float32)
            # Mask illegal moves
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

    def _rollout(self, game):
        """
        Pure MCTS evaluation: play random moves until the game ends.
        Returns value from the perspective of the player who is about
        to move in the PARENT position (the player who just moved).
        """
        # Remember whose perspective we're evaluating from
        # (the player who just moved to reach this position)
        eval_player = game.current_player * -1  # the parent's player

        while not game.done:
            legal = game.get_legal_moves()
            move = np.random.choice(legal)
            game.make_move(move)

        result = game.get_result(eval_player)
        return result if result is not None else 0

    def _network_predict(self, game):
        """
        Get the neural network's policy and value for a position.
        Returns (policy, value) where policy is masked and normalized.
        """
        import torch

        nn_input = game.get_nn_input()
        # Send to same device as the network
        device = next(self.network.parameters()).device
        state_tensor = torch.FloatTensor(nn_input).unsqueeze(0).to(device)

        self.network.eval()
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        value = value.cpu().item()

        # Mask illegal moves and renormalize
        mask = game.get_legal_moves_mask()
        policy *= mask
        total = policy.sum()
        if total > 0:
            policy /= total
        else:
            # Fallback: uniform over legal moves (shouldn't happen with a trained net)
            policy = mask / mask.sum()

        return policy, value

    def _network_evaluate(self, node):
        """Get value estimate from the neural network."""
        _, value = self._network_predict(node.game_state)
        # Value is from current player's perspective at this node
        # We need it from the parent's perspective (negate it)
        return -value

    def _add_dirichlet_noise(self, root):
        """
        Add Dirichlet noise to root node priors.

        This ensures the search explores all moves from the root,
        even if the network strongly prefers one move. Critical for
        self-play training — without noise, the AI would always play
        the same openings and never discover alternatives.

        noise_prior = (1 - ε) × network_prior + ε × Dirichlet(α)

        α = 0.3 for Connect Four (smaller boards use higher α)
        ε = 0.25 (25% noise, 75% network prior)
        """
        legal_actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))

        for i, action in enumerate(legal_actions):
            child = root.children[action]
            child.prior = ((1 - self.dirichlet_epsilon) * child.prior +
                           self.dirichlet_epsilon * noise[i])

    def _backpropagate(self, path, value):
        """
        Send the evaluation back up the tree.

        Value alternates sign at each level because players alternate.
        If the position is good for me (+0.5), it's bad for my opponent (-0.5).
        """
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # flip perspective at each level


def get_mcts_action(game, num_simulations=100, temperature=0.0, network=None):
    """
    Convenience function: run MCTS and return an action.

    temperature controls exploration:
        0.0: always pick the most visited move (deterministic, for evaluation)
        1.0: sample proportional to visit counts (stochastic, for training)
    """
    mcts = MCTS(network=network, num_simulations=num_simulations)
    action_probs = mcts.search(game)

    if temperature == 0:
        # Greedy: pick the most visited move
        action = np.argmax(action_probs)
    else:
        # Sample with temperature
        # Raise probs to power 1/temp, then renormalize
        adjusted = action_probs ** (1.0 / temperature)
        total = adjusted.sum()
        if total > 0:
            adjusted /= total
        action = np.random.choice(COLS, p=adjusted)

    return action, action_probs


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: MCTS beats random
    print("=" * 50)
    print("TEST 1: Pure MCTS vs Random (50 games)")
    print("=" * 50)

    mcts_wins = 0
    random_wins = 0
    draws = 0

    for i in range(50):
        game = ConnectFour()
        # MCTS plays as player 1, random as player 2
        while not game.done:
            if game.current_player == 1:
                action, _ = get_mcts_action(game, num_simulations=100, temperature=0)
            else:
                legal = game.get_legal_moves()
                action = np.random.choice(legal)
            game.make_move(action)

        if game.winner == 1:
            mcts_wins += 1
        elif game.winner == -1:
            random_wins += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            print(f"  Game {i+1}/50: MCTS={mcts_wins}, Random={random_wins}, Draw={draws}")

    win_rate = mcts_wins / 50
    print(f"\nResults: MCTS wins={mcts_wins}, Random wins={random_wins}, Draws={draws}")
    print(f"MCTS win rate: {win_rate:.0%}")
    assert win_rate > 0.7, f"MCTS should beat random most of the time, got {win_rate:.0%}"
    print("✓ MCTS significantly beats random\n")

    # Test 2: MCTS finds obvious winning move
    print("=" * 50)
    print("TEST 2: MCTS finds forced win")
    print("=" * 50)
    game = ConnectFour()
    # Set up a position where player 1 has 3 in a row on the bottom
    for col in [3, 0, 4, 0, 5]:
        game.make_move(col)
    # Player 1 should play column 6 or 2 to win
    print(game)
    print()

    action, probs = get_mcts_action(game, num_simulations=200, temperature=0)
    print(f"MCTS visit distribution: {[f'{p:.2f}' for p in probs]}")
    print(f"MCTS chose column: {action}")
    assert action in [2, 6], f"Should pick column 2 or 6 to win, got {action}"
    print("✓ MCTS found the winning move\n")

    # Test 3: MCTS blocks opponent's win
    print("=" * 50)
    print("TEST 3: MCTS blocks forced loss")
    print("=" * 50)
    game = ConnectFour()
    # Player 1 plays scattered, player 2 builds 3 in a row
    for col in [0, 3, 0, 4, 1, 5]:
        game.make_move(col)
    # Player 1 MUST block column 6 or 2
    print(game)
    print()

    action, probs = get_mcts_action(game, num_simulations=200, temperature=0)
    print(f"MCTS visit distribution: {[f'{p:.2f}' for p in probs]}")
    print(f"MCTS chose column: {action}")
    assert action in [2, 6], f"Should block at column 2 or 6, got {action}"
    print("✓ MCTS blocked the opponent's win\n")

    # Test 4: Visit counts and probs
    print("=" * 50)
    print("TEST 4: Action probabilities")
    print("=" * 50)
    game = ConnectFour()
    _, probs = get_mcts_action(game, num_simulations=200, temperature=0)
    print(f"Opening position probs: {[f'{p:.2f}' for p in probs]}")
    print(f"Center column (3) prob: {probs[3]:.2f}")
    print("(Center should be preferred — it controls the most lines)")
    assert probs[3] > 0.1, "Center column should have significant probability"
    print("✓ Reasonable opening policy\n")

    print("All MCTS tests passed!")