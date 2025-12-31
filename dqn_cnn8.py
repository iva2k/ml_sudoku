#!/usr/bin/env python3
# dqn_cnn8.py

"""
Deep Q-Network Solver (DQN) for Sudoku - CNN8 (Gated Recurrent Axial Transformer).

Improvements over CNN7:
1.  **Gated Recurrence (ConvGRU)**:
    - CNN7 used a simple residual update (x = x + attention). This causes state drift
      and instability over many steps (exploding/vanishing signals), leading to
      degradation on simple puzzles and training plateaus.
    - CNN8 uses a Convolutional GRU to update the state. This allows the model to
      explicitly "forget" or "keep" information. Crucially, it allows the model to
      lock in a solution for a cell and stop updating it, preventing corruption of
      simple/solved areas while reasoning about complex ones.

2.  **Disentangled Attention**:
    - CNN7 summed Row, Col, and Box attention.
    - CNN8 concatenates them and lets the GRU/Mixer decide which signal is relevant.

Pre-training:

--- Test Performance by Difficulty ---
  Blanks |   Solved |   Unsolved |   Solve Rate
-------- + -------- + ---------- + ------------
       3 |        5 |          0 |       100.0%
       4 |        5 |          0 |       100.0%
       5 |        5 |          0 |       100.0%
       6 |        5 |          0 |       100.0%
       7 |        5 |          0 |       100.0%
       8 |        5 |          0 |       100.0%
       9 |        4 |          1 |        80.0%
      10 |        4 |          1 |        80.0%
      11 |        4 |          1 |        80.0%
      12 |        5 |          0 |       100.0%
      13 |        5 |          0 |       100.0%
      14 |        5 |          0 |       100.0%
      15 |        4 |          1 |        80.0%
      16 |        4 |          1 |        80.0%
      17 |        4 |          1 |        80.0%
      18 |        3 |          2 |        60.0%
      19 |        1 |          4 |        20.0%
      20 |        2 |          3 |        40.0%
      21 |        0 |          5 |         0.0%
      22 |        1 |          4 |        20.0%
      23 |        1 |          4 |        20.0%
      24 |        2 |          3 |        40.0%
      25 |        2 |          3 |        40.0%
      26 |        2 |          3 |        40.0%
      27 |        0 |          5 |         0.0%
      28 |        1 |          4 |        20.0%
      29 |        1 |          4 |        20.0%
      30 |        0 |          5 |         0.0%
      31 |        1 |          4 |        20.0%
      32 |        0 |          5 |         0.0%
      ...|
      55 |        0 |          5 |         0.0%
Final Capability Score: 8.120

"""

import torch
import torch.nn as nn

from dqn_cnn2 import SudokuConstraintConv


class AxialSudokuAttention(nn.Module):
    """
    Computes Multi-Head Attention along a specific Sudoku dimension (Row, Col, or Box).
    (Copied from CNN7 to ensure standalone stability)
    """

    def __init__(
        self, d_model: int, mode: str, num_heads: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        # Self-Attention Layer
        # batch_first=True expects inputs of shape (Batch, SeqLen, Embedding)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, 9, 9)
        Output: (B, C, 9, 9) with attention applied along the specified dimension.
        """
        b, c, h, w = x.shape
        # We need to reshape 'x' such that the target dimension becomes the sequence length (L=9).
        # The other spatial dimension becomes part of the batch size for parallel processing.

        if self.mode == "row":
            # Rows: We want to attend across columns (dim 3).
            # (B, C, 9, 9) -> (B, 9, C, 9) -> (B*9, 9, C)
            # Batch items = B * 9 (Num Rows), Seq Len = 9 (Num Cols)
            x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, 9, 9, C)
            x_reshaped = x_perm.view(b * 9, 9, c)

            attn_out, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)

            # Reshape back
            # (B*9, 9, C) -> (B, 9, 9, C) -> (B, C, 9, 9)
            out = attn_out.view(b, 9, 9, c).permute(0, 3, 1, 2)

        elif self.mode == "col":
            # Cols: We want to attend across rows (dim 2).
            # (B, C, 9, 9) -> (B, 9, 9, C) -> permute to (B, 9, 9, C) where 2nd dim is col
            # Ideally: treat each column as an independent sequence.
            # (B, C, 9, 9) -> (B, 9, C, 9). We want (B * 9, 9, C) where 9 is the row sequence.

            # Transpose to put Col dim before Row dim: (B, C, 9w, 9h)
            # Actually easier: Permute to (B, W, H, C)
            x_perm = x.permute(0, 3, 2, 1).contiguous()  # (B, 9, 9, C)
            x_reshaped = x_perm.view(b * 9, 9, c)

            attn_out, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)

            # Reshape back: (B*9, 9, C) -> (B, 9w, 9h, C) -> permute back
            out = attn_out.view(b, 9, 9, c).permute(0, 3, 2, 1)

        elif self.mode == "box":
            # Boxes: We want to attend within each 3x3 block.
            # (B, C, 9, 9) -> break into 3x3 blocks.
            # View as (B, C, 3, 3, 3, 3) -> (B, 3, 3, 9, C)
            # effectively: (B, BoxesH, BoxesW, SeqLen, C)

            # 1. Unfold to local blocks? Or view?
            # View is cleaner for non-overlapping blocks.
            x_view = x.view(b, c, 3, 3, 3, 3)
            # Permute to put box dims together and spatial dims together
            # Target: (B, 3, 3, 3, 3, C) -> flatten first 3 dims to batch, next 2 to seq
            # (B, C, box_h, inner_h, box_w, inner_w)
            # Permute -> (B, box_h, box_w, inner_h, inner_w, C)
            x_perm = x_view.permute(0, 2, 4, 3, 5, 1).contiguous()

            # Check shape: (B, 3, 3, 3, 3, C)
            # Collapse B, box_h, box_w -> New Batch Size = B * 9
            # Collapse inner_h, inner_w -> Seq Len = 9
            x_reshaped = x_perm.view(b * 9, 9, c)

            attn_out, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)

            # Back to (B, 3, 3, 3, 3, C)
            out_view = attn_out.view(b, 3, 3, 3, 3, c)
            # Permute back to (B, C, box_h, inner_h, box_w, inner_w)
            # Current: (B, bh, bw, ih, iw, C) -> target (B, C, bh, ih, bw, iw)
            out_perm = out_view.permute(0, 5, 1, 3, 2, 4).contiguous()

            out = out_perm.view(b, c, 9, 9)

        else:
            raise ValueError(f"Unknown Axial Mode: {self.mode}")

        return out


class ConvGRUCell(nn.Module):
    """
    A Convolutional GRU Cell for 1x1 spatial updates (per-cell gating).
    This acts as the memory manager, deciding what to keep and what to update.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 1x1 convolution is equivalent to a dense layer shared across all pixels
        # We use it here to implement the GRU gates efficiently.
        # Input to gates is concatenation of (Input_Features, Hidden_State)
        self.conv_z = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.conv_r = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)
        self.conv_h = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x, h):
        """Forward pass."""
        # x: Input features (from Attention)
        # h: Previous hidden state

        combined = torch.cat([x, h], dim=1)

        z = torch.sigmoid(self.conv_z(combined))  # Update gate
        r = torch.sigmoid(self.conv_r(combined))  # Reset gate

        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_reset))  # Candidate hidden

        # GRU Update Rule
        h_next = (1 - z) * h + z * h_tilde
        return h_next


class ReasoningLayer(nn.Module):
    """
    Computes the logical features for the current step using Axial Attention.
    Does NOT update the state; just extracts information for the GRU.
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()

        # 1. Attention Branches
        self.attn_row = AxialSudokuAttention(d_model, "row", num_heads)
        self.attn_col = AxialSudokuAttention(d_model, "col", num_heads)
        self.attn_box = AxialSudokuAttention(d_model, "box", num_heads)

        self.norm = nn.GroupNorm(32, d_model)

        # Mixer to combine attention outputs into a single feature map
        # Input: Row + Col + Box + Original State (x0)
        self.mixer = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )

        # FFN (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
        )

    def forward(self, x, x0):
        # Normalize before attention (Pre-Norm)
        x_norm = self.norm(x)

        # Compute Attention
        out_row = self.attn_row(x_norm)
        out_col = self.attn_col(x_norm)
        out_box = self.attn_box(x_norm)

        # Mix with initial constraints (x0) to anchor reasoning
        combined = torch.cat([out_row, out_col, out_box, x0], dim=1)
        features = self.mixer(combined)

        # Apply FFN
        features = features + self.ffn(features)

        return features


class DQNSolverCNN8(nn.Module):
    """
    CNN8: Gated Recurrent Axial Transformer.
    """

    def __init__(
        self, _input_shape, _output_size, device=None, reasoning_steps: int = 16
    ):
        super().__init__()
        self.device = device
        self.reasoning_steps = reasoning_steps

        d_model = 128

        # 1. Perception
        self.constraint_conv = SudokuConstraintConv(10, 48)
        self.embedding = nn.Sequential(
            nn.Conv2d(48 * 3, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )

        # Positional Embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, 9, 9) * 0.02)

        # 2. Reasoning Modules
        self.reasoning_layer = ReasoningLayer(d_model, num_heads=4)
        self.gru = ConvGRUCell(d_model, d_model)

        # Final Norm
        self.norm_final = nn.GroupNorm(32, d_model)

        # 3. Output Head
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 9, kernel_size=1),
        )

    def forward(self, x):
        """
        Forward pass with Gated Recurrence.
        """
        b = x.shape[0]

        # 1. Embed
        x = self.constraint_conv(x)
        x = self.embedding(x)
        x = x + self.pos_embedding
        x0 = x  # Anchor

        # 2. Recurrent Reasoning with GRU
        for _ in range(self.reasoning_steps):
            # Extract logical features from current state
            features = self.reasoning_layer(x, x0)
            # Update state using GRU (Gated update)
            x = self.gru(features, x)

        x = self.norm_final(x)

        # 3. Output
        x = self.head(x)
        return x.permute(0, 2, 3, 1).reshape(b, -1)
