"""
AlphaZero Neural Network — Policy + Value dual-headed ResNet.

Input:  board position as 3 channels × 6 rows × 7 columns
Output: policy (probability over 7 columns) + value (who's winning, -1 to +1)

Architecture:
  Input (3×6×7)
    → Conv block (initial feature extraction)
    → N × Residual blocks (deep feature processing)
    → Policy head → softmax over 7 columns
    → Value head → tanh scalar in [-1, +1]

This is the same architecture DeepMind used in AlphaZero, scaled down.
AlphaZero for Go used 20+ residual blocks with 256 filters.
We use 5 blocks with 64 filters — enough for Connect Four.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.game import ROWS, COLS


class ConvBlock(nn.Module):
    """
    Initial convolution block: Conv → BatchNorm → ReLU.

    Transforms the raw board representation (3 channels) into
    a rich feature map (num_filters channels). Each filter learns
    to detect a different spatial pattern — horizontal threats,
    diagonal setups, blocked columns, etc.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """
    Residual block: two convolutions with a skip connection.

    x → Conv → BN → ReLU → Conv → BN → + x → ReLU
                                         ↑
                                    skip connection

    The skip connection lets gradients flow directly through the network,
    enabling deeper networks without vanishing gradients. Same concept
    as the residual connections in the Transformer.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)    # skip connection
        return out


class AlphaZeroNetwork(nn.Module):
    """
    Dual-headed network: shared backbone → policy head + value head.

    The shared backbone learns to understand the board position.
    The policy head learns which moves are promising.
    The value head learns who's winning.

    Both heads share the same features from the backbone — understanding
    "there's a diagonal threat on the right" is useful for both deciding
    where to play AND evaluating who's ahead.
    """

    def __init__(self, num_res_blocks=5, num_filters=64):
        super().__init__()

        # Input: 3 channels (my pieces, opponent pieces, turn indicator)
        self.input_block = ConvBlock(3, num_filters)

        # Residual backbone
        self.res_blocks = nn.Sequential(*[
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # --- Policy head ---
        # Reduces channels, then flattens and projects to 7 (one per column)
        self.policy_conv = ConvBlock(num_filters, 2)   # 64 → 2 channels
        self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)  # 2×6×7=84 → 7

        # --- Value head ---
        # Reduces channels, flattens, then two FC layers to a single value
        self.value_conv = ConvBlock(num_filters, 1)    # 64 → 1 channel
        self.value_fc1 = nn.Linear(1 * ROWS * COLS, 64)  # 1×6×7=42 → 64
        self.value_fc2 = nn.Linear(64, 1)                  # 64 → 1

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 6, 7) — board representation

        Returns:
            policy_logits: (batch, 7) — raw scores per column (NOT softmaxed)
            value: (batch,) — position evaluation in [-1, +1]
        """
        # Shared backbone
        out = self.input_block(x)       # (B, 64, 6, 7)
        out = self.res_blocks(out)      # (B, 64, 6, 7)

        # Policy head
        p = self.policy_conv(out)       # (B, 2, 6, 7)
        p = p.view(p.size(0), -1)      # (B, 84)
        p = self.policy_fc(p)           # (B, 7)

        # Value head
        v = self.value_conv(out)        # (B, 1, 6, 7)
        v = v.view(v.size(0), -1)      # (B, 42)
        v = F.relu(self.value_fc1(v))   # (B, 64)
        v = torch.tanh(self.value_fc2(v))  # (B, 1) → [-1, +1]
        v = v.squeeze(-1)              # (B,)

        return p, v

    def predict(self, game):
        """
        Convenience method: get policy + value for a single game state.
        Handles conversion from game to tensor.

        Returns:
            policy: numpy array (7,) — probabilities (masked + normalized)
            value: float — position evaluation
        """
        self.eval()
        nn_input = game.get_nn_input()
        device = next(self.parameters()).device
        state_tensor = torch.FloatTensor(nn_input).unsqueeze(0).to(device)

        with torch.no_grad():
            policy_logits, value = self(state_tensor)

        # Softmax → mask illegal moves → renormalize
        policy = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
        mask = game.get_legal_moves_mask()
        policy *= mask
        total = policy.sum()
        if total > 0:
            policy /= total
        else:
            policy = mask / mask.sum()

        return policy, value.item()


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    from src.game import ConnectFour
    import numpy as np

    # Test 1: Architecture
    print("=" * 50)
    print("TEST 1: Network architecture")
    print("=" * 50)
    net = AlphaZeroNetwork(num_res_blocks=5, num_filters=64)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # Print layer summary
    for name, param in net.named_parameters():
        print(f"  {name:40s} {str(list(param.shape)):20s} ({param.numel():,})")
    print()

    # Test 2: Forward pass shapes
    print("=" * 50)
    print("TEST 2: Forward pass")
    print("=" * 50)
    batch = torch.randn(8, 3, 6, 7)  # batch of 8 positions
    policy_logits, values = net(batch)
    print(f"Input shape:   {batch.shape}")
    print(f"Policy shape:  {policy_logits.shape}  (expected: [8, 7])")
    print(f"Value shape:   {values.shape}   (expected: [8])")
    assert policy_logits.shape == (8, 7), f"Wrong policy shape: {policy_logits.shape}"
    assert values.shape == (8,), f"Wrong value shape: {values.shape}"
    print(f"Value range:   [{values.min().item():.3f}, {values.max().item():.3f}]  (should be in [-1, 1])")
    assert values.min() >= -1 and values.max() <= 1, "Values should be in [-1, 1]"
    print("✓ Shapes correct\n")

    # Test 3: Predict on actual game state
    print("=" * 50)
    print("TEST 3: Predict on game position")
    print("=" * 50)
    game = ConnectFour()
    game.make_move(3)  # X plays center
    game.make_move(3)  # O plays center

    policy, value = net.predict(game)
    print(f"Position:\n{game}\n")
    print(f"Policy: {[f'{p:.3f}' for p in policy]}")
    print(f"Value:  {value:.3f}")
    print(f"Policy sums to: {policy.sum():.4f}")
    assert abs(policy.sum() - 1.0) < 1e-5, f"Policy should sum to 1, got {policy.sum()}"
    print("✓ Policy is valid probability distribution\n")

    # Test 4: Illegal move masking
    print("=" * 50)
    print("TEST 4: Illegal move masking")
    print("=" * 50)
    game = ConnectFour()
    # Fill column 3 completely
    for _ in range(3):
        game.make_move(3)
        game.make_move(3)
    print(f"Position (column 3 is full):\n{game}\n")

    policy, value = net.predict(game)
    print(f"Policy: {[f'{p:.3f}' for p in policy]}")
    print(f"Column 3 probability: {policy[3]:.6f}")
    assert policy[3] == 0.0, f"Full column should have 0 probability, got {policy[3]}"
    assert abs(policy.sum() - 1.0) < 1e-5, "Policy should still sum to 1"
    print("✓ Illegal moves correctly masked\n")

    # Test 5: Batch of game states
    print("=" * 50)
    print("TEST 5: Batch prediction")
    print("=" * 50)
    games = [ConnectFour() for _ in range(16)]
    # Make different moves in each game
    for i, g in enumerate(games):
        for _ in range(i % 5):
            legal = g.get_legal_moves()
            g.make_move(np.random.choice(legal))
            if g.done:
                break

    # Batch forward pass
    states = torch.FloatTensor(np.stack([g.get_nn_input() for g in games]))
    net.eval()
    with torch.no_grad():
        p, v = net(states)
    print(f"Batch input: {states.shape}")
    print(f"Batch policy: {p.shape}")
    print(f"Batch values: {v.shape}")
    print(f"All values in [-1,1]: {(v >= -1).all() and (v <= 1).all()}")
    print("✓ Batch prediction works\n")

    print(f"Network has {n_params:,} parameters")
    print("All tests passed!")