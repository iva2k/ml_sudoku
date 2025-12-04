#!/usr/bin/env python3
# dqn_cnn4.py

"""
Deep Q-Network Solver (DQN) for Sudoku.
Hybrid CNN-Attention model. It uses specialized convolutions for Sudoku-specific
perception and then employs self-attention for global, logical reasoning.

1. Architecture Rationale

This model aims to combine the best of both CNNs and Transformers for Sudoku:

* **CNN for Perception (SudokuConstraintConv)**: The initial layers use convolutions with
  kernels explicitly shaped for Sudoku's geometry (1x9 for rows, 9x1 for columns, 3x3 for
  boxes). This provides a powerful and efficient inductive bias, allowing the model to
  "see" the basic constraints of the game without having to learn them from scratch.

* **Attention for Reasoning (GlobalReasoningBlock)**: After the initial perception, the
  model switches to a Transformer-style self-attention mechanism. This is the core of its
  logical capability. It allows every cell to look at every other cell on the board and
  dynamically decide which ones are most important for updating its own state. This is
  critical for solving complex, non-local patterns (like "X-wings" or "naked pairs")
  that a standard CNN cannot handle.

* **Iterative Refinement**: The model stacks multiple `GlobalReasoningBlock`s. Each block
  represents one step of logical deduction. By passing the board state through these
  blocks sequentially, the model can perform chained reasoning, where the conclusion of one
  step becomes the premise for the next. For example:
  1. *Step 1*: Cell A realizes it must be a 5.
  2. *Step 2*: Cell B, in the same row as A, sees that A is now a 5 and concludes it can no
     longer be a 5 itself.

2. Comparison to Other Models

* **vs. Pure CNN (dqn_cnn1, dqn_cnn3)**: Pure CNNs struggle because their local kernels
  (e.g., 3x3) are a poor match for Sudoku's global rules. They lack an effective mechanism
  for long-range information propagation. This model solves that by adding attention.

* **vs. Pure Transformer (dqn_transformer)**: A pure Transformer treats the grid as a
  generic sequence of 81 tokens. It must learn all spatial relationships (what a "row" or
  "box" is) from scratch using only positional encodings. This hybrid model is potentially
  more data-efficient because the CNN front-end hard-codes the geometric priors, freeing
  up the attention mechanism to focus purely on logical deduction.

"""

# import torch
import torch
import torch.nn as nn

from dqn_cnn2 import SudokuConstraintConv
from dqn_cnn3 import ReasoningBlock


class GlobalReasoningBlock(nn.Module):
    """
    A reasoning block that combines local (per-cell) logic with a global
    self-attention mechanism for information propagation across the entire grid.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Local reasoning part (from dqn_cnn3)
        self.local_reasoning = ReasoningBlock(d_model)

        # Global attention part
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x is of shape (B, C, H, W)."""
        b, c, h, w = x.shape
        residual = x

        # 1. Local Reasoning (per-cell MLP)
        x = self.local_reasoning(x)

        # 2. Global Attention (information propagation)
        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.view(b, c, h * w).permute(0, 2, 1)
        x_seq = self.norm1(x_seq)

        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.norm2(x_seq + attn_out)

        # Reshape back to image format: (B, H*W, C) -> (B, C, H, W)
        x = x_seq.permute(0, 2, 1).view(b, c, h, w)

        return x + residual


class DQNSolverCNN4(nn.Module):
    """
    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.
    This version uses a hybrid CNN-Attention architecture.
    """

    def __init__(self, _input_shape, _output_size, device=None):
        super().__init__()
        self.device = device
        d_model = 128  # Dimension for reasoning features

        # 1. Perception: Extract geometric constraints
        # Input: 10 channels (one-hot digits)
        # Output: 48 features per Row/Col/Box -> 144 total
        self.constraint_conv = SudokuConstraintConv(10, 48)

        # 2. Reducer: Project the concatenated perception features into the reasoning dimension
        self.reduce = nn.Sequential(
            nn.Conv2d(48 * 3, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # 3. Global Reasoning: A stack of blocks that combine local logic and global attention.
        # This allows the model to perform iterative, chained logical deductions.
        self.reasoning = nn.Sequential(
            GlobalReasoningBlock(d_model, n_heads=4),
            GlobalReasoningBlock(d_model, n_heads=4),
            GlobalReasoningBlock(d_model, n_heads=4),
            GlobalReasoningBlock(d_model, n_heads=4),
        )

        # 4. Output Head: Project each of the 81 cell embeddings to 9 possible digit scores.
        self.fc = nn.Linear(d_model, 9)

    def forward(self, x):
        """Forward pass."""
        b, _c, _h, _w = x.shape

        # Stage 1: Perceive local Sudoku constraints (rows, cols, boxes)
        x = self.constraint_conv(x)

        # Stage 2: Project into the main reasoning space
        x = self.reduce(x)

        # Stage 3: Perform iterative global reasoning
        x = self.reasoning(x)

        # Stage 4: Predict Q-values
        # x is (B, D_Model, 9, 9). We want (B, 81, 9) for cell-centric Q-values.
        # Permute to (B, 9, 9, D_Model) -> Reshape to (B, 81, D_Model)
        x = x.permute(0, 2, 3, 1).reshape(b, 81, -1)
        x = self.fc(x)  # (B, 81, 9)

        # Flatten to standard RL action space: (B, 729)
        return x.view(b, -1)
