#!/usr/bin/env python3
# dqn_cnn7.py

"""
Deep Q-Network Solver (DQN) for Sudoku - CNN7 (Recurrent Axial Transformer).

Architecture:
1.  **Input Embedding**: SudokuConstraintConv (from CNN2) to extract initial geometric features.
2.  **Recurrent Backbone**: A stack of `SudokuTransformerBlock` layers that run significantly deeper
    by re-using the same weights (Recurrent Neural Network style).
    - Unlike CNN6 (ACT), this uses a FIXED number of steps during training to avoid the complexity
      and instability of dynamic halting.
    - We use 32 steps by default to mimic "deep thinking".
3.  **Axial Attention**: The core reasoning mechanism.
    - Instead of global O(N^2) attention, we run 3 parallel attention branches:
      - Row Attention (1x9)
      - Column Attention (9x1)
      - Box Attention (3x3)
    - This enforces Sudoku constraints strictly and is much more efficient (O(3 * N * 9^2) vs O(N * 81^2)).

This model aims to solve "Super-Hard" puzzles by allowing deep logical propagation while remaining
training-efficient on GPUs due to parallelizable attention heads (within each step).

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
       9 |        5 |          0 |       100.0%
      10 |        5 |          0 |       100.0%
      11 |        3 |          2 |        60.0%
      12 |        5 |          0 |       100.0%
      13 |        5 |          0 |       100.0%
      14 |        3 |          2 |        60.0%
      15 |        4 |          1 |        80.0%
      16 |        3 |          2 |        60.0%
      17 |        5 |          0 |       100.0%
      18 |        2 |          3 |        40.0%
      19 |        1 |          4 |        20.0%
      20 |        2 |          3 |        40.0%
      21 |        0 |          5 |         0.0%
      22 |        0 |          5 |         0.0%
      23 |        2 |          3 |        40.0%
      24 |        0 |          5 |         0.0%
      25 |        1 |          4 |        20.0%
      26 |        1 |          4 |        20.0%
      27 |        3 |          2 |        60.0%
      28 |        0 |          5 |         0.0%
      ...|
      55 |        0 |          5 |         0.0%
Final Capability Score: 10.090

Post-Training (17k episodes):

  Blanks |   Solved |   Unsolved |   Solve Rate
-------- + -------- + ---------- + ------------
       3 |      208 |          1 |        99.5%
       4 |      198 |          1 |        99.5%
       5 |      192 |          2 |        99.0%
       6 |      206 |          1 |        99.5%
       7 |      189 |          8 |        95.9%
       8 |      202 |          7 |        96.7%
       9 |      170 |         19 |        89.9%
      10 |      177 |         24 |        88.1%
      11 |      181 |         27 |        87.0%
      12 |      167 |         35 |        82.7%
      13 |      183 |         57 |        76.2%
      14 |      140 |         57 |        71.1%
      15 |      147 |         78 |        65.3%
      16 |      107 |         93 |        53.5%
      17 |      120 |         94 |        56.1%
      18 |      111 |        105 |        51.4%
      19 |       66 |        124 |        34.7%
      20 |       67 |        119 |        36.0%
      21 |       51 |        153 |        25.0%
      22 |       42 |        141 |        23.0%
      23 |       42 |        184 |        18.6%
      24 |       31 |        197 |        13.6%
      25 |       13 |        183 |         6.6%
      26 |       13 |        184 |         6.6%
      27 |       16 |        203 |         7.3%
      28 |        3 |        190 |         1.6%
      29 |        5 |        218 |         2.2%
      30 |        3 |        214 |         1.4%
      31 |        2 |        214 |         0.9%
Final Capability Score: 0.334

"""

import torch
import torch.nn as nn

from dqn_cnn2 import SudokuConstraintConv


class AxialSudokuAttention(nn.Module):
    """
    Computes Multi-Head Attention along a specific Sudoku dimension (Row, Col, or Box).
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


class SudokuTransformerBlock(nn.Module):
    """
    One block of reasoning that processes Rows, Columns, and Boxes in parallel.
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()

        # 1. Attention Branches
        self.attn_row = AxialSudokuAttention(d_model, "row", num_heads)
        self.attn_col = AxialSudokuAttention(d_model, "col", num_heads)
        self.attn_box = AxialSudokuAttention(d_model, "box", num_heads)

        # Mixer / Projection for the parallel branches
        self.mixer = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=1, bias=False),
        )

        # Norms
        self.norm1 = nn.GroupNorm(32, d_model)
        self.norm2 = nn.GroupNorm(32, d_model)

        # 2. Feed Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model * 4, d_model, kernel_size=1, bias=False),
        )

    def forward(self, x, x0=None):
        """Forward pass."""

        # 1. Parallel Attention
        residual = x
        x_norm = self.norm1(x)
        
        if x0 is None:
            x0 = x

        # Run all 3 branches
        out_row = self.attn_row(x_norm)
        out_col = self.attn_col(x_norm)
        out_box = self.attn_box(x_norm)

        # Concatenate and Mix
        combined = torch.cat([out_row, out_col, out_box, x0], dim=1)
        x_attn = self.mixer(combined)

        x = residual + x_attn

        # 2. FFN
        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)

        x = residual + x_ffn

        return x


class DQNSolverCNN7(nn.Module):
    """
    CNN7: Fixed-step Recurrent Axial Transformer.
    """

    def __init__(
        self, _input_shape, _output_size, device=None, reasoning_steps: int = 16
    ):
        super().__init__()
        self.device = device
        self.reasoning_steps = reasoning_steps

        d_model = 128  # Slightly smaller than CNN6 (192) to keep FFNs fast,
        # but attention is more powerful.

        # 1. Perception (reuse efficient convs)
        self.constraint_conv = SudokuConstraintConv(10, 48)
        self.embedding = nn.Sequential(
            nn.Conv2d(48 * 3, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )
        
        # Positional Embeddings (Learnable)
        # Helps the model distinguish identical features in different locations (e.g. empty board)
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, 9, 9) * 0.02)

        # 2. Recurrent Transformer Block
        self.transformer = SudokuTransformerBlock(d_model, num_heads=4)
        
        # Final Norm (Critical for Pre-Norm architectures)
        self.norm_final = nn.GroupNorm(32, d_model)

        # 3. Output Head
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 9, kernel_size=1),  # Output 9 logits per cell
        )

    def forward(self, x):
        """
        Forward pass with fixed recurrence.
        """
        b = x.shape[0]

        # 1. Embed
        x = self.constraint_conv(x)
        x = self.embedding(x)
        x = x + self.pos_embedding
        x0 = x

        # 2. Recurrent Reasoning
        # We implicitly unroll this loop during training (BPTT).
        # Since 'self.transformer' is the same module, weights are tied.
        for _ in range(self.reasoning_steps):
            x = self.transformer(x, x0)

        x = self.norm_final(x)

        # 3. Output
        x = self.head(x)  # (B, 9, 9, 9) -> (B, Digits, Rows, Cols)
        
        # Permute to (B, Rows, Cols, Digits) and flatten to (B, 729)
        return x.permute(0, 2, 3, 1).reshape(b, -1)


if __name__ == "__main__":
    # Test Verification
    print("Verifying CNN7 Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQNSolverCNN7(None, 729, device=device, reasoning_steps=4).to(device)

    # 1. Check Parameter Count
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")

    # 2. Dummy Forward Pass
    dummy_input = torch.zeros((2, 10, 9, 9), device=device)
    try:
        output = model(dummy_input)
        print(f"Output Shape: {output.shape}")
        assert output.shape == (2, 729)
        print("✅ Forward pass successful.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise e
