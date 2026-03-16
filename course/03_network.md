# Chapter 3: The Neural Network — Teaching a Computer to See Board Positions

## What the Network Does

The neural network replaces two things that MCTS previously did badly:

1. **Random rollouts → Value head.** Instead of playing random moves to the end (noisy, slow), the network instantly estimates who's winning from any position: a single number between -1 and +1.

2. **Uniform priors → Policy head.** Instead of treating all moves as equally promising, the network says "column 3 looks best (35%), columns 2 and 4 are decent (15% each), the rest are weak." MCTS explores the promising moves first.

Both outputs come from a single forward pass through one network — the shared backbone learns to understand the board, and two separate heads extract different information from that understanding.

## Part 1: Why a Convolutional Network?

### Board Positions Are Images

A Connect Four board is a 6×7 grid — structurally identical to a tiny image. Patterns in the game are **spatial**: horizontal threats, vertical stacks, diagonal setups. These are exactly the kind of patterns convolutional neural networks (CNNs) excel at detecting.

A regular fully-connected network would see the board as a flat list of 42 numbers with no spatial structure. A CNN preserves the 2D layout and learns spatial filters — "detect a horizontal line of 3," "detect a diagonal pair," "detect an empty cell above a threat."

### Convolution — The Core Operation

A convolution slides a small filter (typically 3×3) across the board, computing a weighted sum at each position:

```
Filter (learned):        Board region:          Output:
[0.5, -0.3,  0.1]      [1, 1, 1]
[0.2,  0.8, -0.1]  ×   [0, 0, 0]    →    single number
[-0.1, 0.0,  0.4]      [0,-1, 0]

Output = sum of element-wise products
       = 0.5×1 + (-0.3)×1 + 0.1×1 + 0.2×0 + 0.8×0 + ... 
```

The network learns 64 different 3×3 filters, each detecting a different pattern. One filter might activate strongly when it sees "three of my pieces in a row." Another might detect "opponent's piece with empty space above."

**`padding=1`** ensures the output is the same size as the input (6×7). Without padding, each convolution would shrink the grid by 2 in each dimension.

## Part 2: The Input — Three Channels

From Chapter 1, the network input is a `(3, 6, 7)` tensor:

```
Channel 0: My pieces        [0s and 1s — where I have pieces]
Channel 1: Opponent pieces   [0s and 1s — where they have pieces]
Channel 2: Turn indicator    [all 1s]
```

This is analogous to an RGB image being (3, height, width). The three channels let the network's first layer learn separate filters for "my patterns" and "opponent patterns" from the start.

## Part 3: Architecture — Building Blocks

### The ConvBlock

The entry point — transforms raw input channels into feature maps:

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
```

**`nn.Conv2d(3, 64, kernel_size=3, padding=1)`** — 64 filters, each 3×3, applied to the 3-channel input. Output: `(batch, 64, 6, 7)` — 64 feature maps of the same spatial size.

**`nn.BatchNorm2d(64)`** — Normalizes each feature map to have zero mean and unit variance across the batch. This stabilizes training and allows higher learning rates. Without it, the activations at different layers can drift to very different scales, making optimization harder.

How BatchNorm works for one feature map:
```
Before: values might be [-50, 120, -10, 85, ...]   (arbitrary scale)
After:  values are roughly [-1.2, 1.5, -0.3, 0.9, ...]  (normalized)
```

**`F.relu`** — Sets negative values to zero: `max(0, x)`. This non-linearity lets the network represent complex patterns. Without it, stacking linear layers (convolutions) would just produce another linear function.

### The ResBlock

The workhorse — processes features with a skip connection:

```python
class ResBlock(nn.Module):
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
        out = F.relu(out + residual)
        return out
```

The data flow:

```
Input x ──────────────────────┐
    │                          │ (skip connection)
    ▼                          │
  Conv 3×3 → BatchNorm → ReLU │
    │                          │
    ▼                          │
  Conv 3×3 → BatchNorm        │
    │                          │
    ▼                          ▼
  Add: out + x ─────────────────
    │
    ▼
  ReLU
    │
  Output
```

**Why skip connections?** The same reason as in Transformers — they let gradients flow directly through the network during backpropagation. Without them, a 5-block network's gradients would need to pass through 10 convolution layers, potentially vanishing to zero. The skip connection provides a "highway" for gradients, making deeper networks trainable.

**Why the same number of channels in and out?** The skip connection adds input to output: `out + residual`. For addition to work, both tensors must have the same shape. Keeping `channels` constant (64) throughout the backbone ensures this.

### What Each ResBlock Learns

With 5 residual blocks, the network learns hierarchical features:

```
Block 1: Low-level — individual pieces, adjacent pairs
Block 2: Patterns — two-in-a-row, basic threats
Block 3: Tactics — three-in-a-row, blocking moves, forks
Block 4: Strategy — center control, double threats
Block 5: Evaluation — overall position assessment, who's winning
```

This hierarchy emerges naturally from stacking — each block builds on the features discovered by the previous one, similar to how image CNNs learn edges → textures → parts → objects.

## Part 4: The Policy Head

Predicts which column to play — a probability distribution over 7 columns.

```python
# In AlphaZeroNetwork.__init__:
self.policy_conv = ConvBlock(num_filters, 2)        # 64 → 2 channels
self.policy_fc = nn.Linear(2 * ROWS * COLS, COLS)   # 84 → 7
```

The data flow:

```
Backbone output: (batch, 64, 6, 7)     ← rich feature maps
      │
      ▼ ConvBlock(64 → 2)
Policy features: (batch, 2, 6, 7)      ← compress to 2 channels
      │
      ▼ Flatten
Flat vector: (batch, 84)               ← 2 × 6 × 7 = 84 values
      │
      ▼ Linear(84 → 7)
Policy logits: (batch, 7)              ← one score per column
```

**Why compress to 2 channels?** The backbone's 64 channels contain rich information, but the policy head only needs to output 7 numbers. Compressing to 2 channels acts as a bottleneck that forces the policy head to extract only the most relevant spatial information for move selection. This is a standard AlphaZero design choice.

**No softmax here.** The network outputs raw logits. Softmax is applied later — either in the loss function (during training) or explicitly when we need probabilities (during MCTS). This is numerically more stable than applying softmax inside the network.

## Part 5: The Value Head

Predicts who's winning — a single number between -1 and +1.

```python
# In AlphaZeroNetwork.__init__:
self.value_conv = ConvBlock(num_filters, 1)     # 64 → 1 channel
self.value_fc1 = nn.Linear(1 * ROWS * COLS, 64) # 42 → 64
self.value_fc2 = nn.Linear(64, 1)                # 64 → 1
```

```python
# In forward():
v = self.value_conv(out)             # (batch, 1, 6, 7)
v = v.view(v.size(0), -1)           # (batch, 42)
v = F.relu(self.value_fc1(v))       # (batch, 64)
v = torch.tanh(self.value_fc2(v))   # (batch, 1) → [-1, +1]
v = v.squeeze(-1)                   # (batch,)
```

**Why `tanh`?** The value must be between -1 (losing) and +1 (winning). `tanh(x)` maps any real number to this range:
```
tanh(-3) ≈ -0.995    (very confident: losing)
tanh(-1) ≈ -0.762    (probably losing)
tanh(0)  =  0.000    (even)
tanh(1)  ≈  0.762    (probably winning)
tanh(3)  ≈  0.995    (very confident: winning)
```

**Why 1 channel then FC layers?** The value represents the ENTIRE position with a single number — it can't be spatial. So we compress to 1 channel (a summary of the whole board), flatten, and use fully-connected layers to combine everything into one scalar.

## Part 6: The Complete Forward Pass

```python
def forward(self, x):
    # Shared backbone
    out = self.input_block(x)       # (B, 3, 6, 7) → (B, 64, 6, 7)
    out = self.res_blocks(out)      # (B, 64, 6, 7) → (B, 64, 6, 7)

    # Policy head
    p = self.policy_conv(out)       # (B, 64, 6, 7) → (B, 2, 6, 7)
    p = p.view(p.size(0), -1)      # (B, 2, 6, 7) → (B, 84)
    p = self.policy_fc(p)           # (B, 84) → (B, 7)

    # Value head
    v = self.value_conv(out)        # (B, 64, 6, 7) → (B, 1, 6, 7)
    v = v.view(v.size(0), -1)      # (B, 1, 6, 7) → (B, 42)
    v = F.relu(self.value_fc1(v))   # (B, 42) → (B, 64)
    v = torch.tanh(self.value_fc2(v))  # (B, 64) → (B, 1)
    v = v.squeeze(-1)              # (B, 1) → (B,)

    return p, v
```

Shape trace for a batch of 8 positions:

```
Input:          (8, 3, 6, 7)       3 channels, 6×7 board
InputBlock:     (8, 64, 6, 7)      64 feature maps
ResBlock ×5:    (8, 64, 6, 7)      same shape (residual connections)

Policy head:    (8, 2, 6, 7) → (8, 84) → (8, 7)      7 logits
Value head:     (8, 1, 6, 7) → (8, 42) → (8, 64) → (8, 1) → (8,)   scalar
```

Total parameters: **377,629**. For comparison:
- AlphaZero for Go: ~13 million parameters (40 res blocks, 256 filters)
- Our network: ~378K parameters (5 res blocks, 64 filters)
- Ratio: ~35× smaller. Appropriate since Connect Four is ~35× simpler than Go in board size.

## Part 7: Illegal Move Masking

The network outputs logits for all 7 columns — including columns that might be full. We can't play in a full column, so we mask those:

```python
def predict(self, game):
    self.eval()
    nn_input = game.get_nn_input()
    device = next(self.parameters()).device
    state_tensor = torch.FloatTensor(nn_input).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = self(state_tensor)

    policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
    mask = game.get_legal_moves_mask()
    policy *= mask                    # zero out illegal moves
    total = policy.sum()
    if total > 0:
        policy /= total              # renormalize to sum to 1

    return policy, value.cpu().item()
```

```
Network output: [0.15, 0.12, 0.18, 0.25, 0.10, 0.12, 0.08]
Legal mask:     [1,    1,    1,    0,    1,    1,    1   ]    (col 3 full)
After masking:  [0.15, 0.12, 0.18, 0.00, 0.10, 0.12, 0.08]
After renorm:   [0.20, 0.16, 0.24, 0.00, 0.13, 0.16, 0.11]  (sums to 1)
```

The network might think column 3 is a great move, but if it's full, that probability gets redistributed to the legal moves.

## Part 8: The Untrained Network

Before any training, the network's weights are random. What does it output?

```
Policy: [0.141, 0.147, 0.135, 0.162, 0.135, 0.138, 0.142]
Value:  0.101
```

**Policy is near-uniform** (~0.14 each) — the network has no idea which moves are good. All columns look equally promising. This is correct — a random network should have no preference.

**Value is near-zero** (~0.1) — the network has no idea who's winning. This is also correct — a random assessment should be "I don't know" ≈ 0.

As training progresses, the policy will become peaky (strongly preferring good moves) and the value will become more extreme (confidently predicting wins and losses).

## Part 9: Why a Shared Backbone Works

The policy and value heads share the same ResNet backbone. This works because understanding the board position is useful for BOTH tasks:

- **For policy:** "There's a diagonal threat building on the right" → "I should play there to extend it or block it"
- **For value:** "There's a diagonal threat building on the right" → "Whoever owns it is probably ahead"

The backbone learns one representation that serves both purposes. This is more parameter-efficient than having two separate networks, and the shared features often generalize better because they're trained from two different supervision signals simultaneously.

In our A2C implementation (RL project), shared backbones caused competing gradients. AlphaZero avoids this because both heads have well-defined, complementary targets — the policy head learns from MCTS search results and the value head learns from game outcomes. They're not pulling the backbone in opposite directions.

## Summary

| Component | Shape Change | What It Does |
|-----------|-------------|--------------|
| Input | `(B, 3, 6, 7)` | Board as three binary channels |
| ConvBlock | `(B, 3, 6, 7) → (B, 64, 6, 7)` | Initial feature extraction |
| 5× ResBlock | `(B, 64, 6, 7) → (B, 64, 6, 7)` | Deep feature processing with skip connections |
| Policy Conv | `(B, 64, 6, 7) → (B, 2, 6, 7)` | Compress for move prediction |
| Policy FC | `(B, 84) → (B, 7)` | Output move logits |
| Value Conv | `(B, 64, 6, 7) → (B, 1, 6, 7)` | Compress for position evaluation |
| Value FC | `(B, 42) → (B, 64) → (B, 1)` | Output scalar value in [-1, +1] |
| Illegal mask | Applied to policy | Zero out full columns, renormalize |

## What's Next

In [Chapter 4](04_training.md), we bring everything together: the game engine generates positions, MCTS (guided by the network) produces training targets, and the network trains on those targets. This is the AlphaZero self-play loop — the self-improvement cycle that takes the AI from random play to expert play.