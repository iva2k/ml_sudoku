#!/usr/bin/env python3
# dqn_cnn9.py

"""
Deep Q-Network Solver (DQN) for Sudoku - CNN9 (Sequential Gated Recurrent Axial Transformer).

Improvements over CNN8:
1.  **Sequential Attention (Cascaded Reasoning)**:
    - CNN8 ran Row, Col, and Box attention in parallel. This created a latency where
      a constraint found in a Row wasn't visible to the Column attention until the
      NEXT recurrent step.
    - CNN9 runs attention sequentially: Row -> Col -> Box.
    - This allows "intra-step" propagation. A Naked Pair found in a Row immediately
      restricts candidates for the Column attention within the SAME reasoning step.
      This drastically speeds up the detection of complex patterns like Chains and Wings.

2.  **Enhanced Mixer**:
    - Uses a slightly larger kernel or deeper mixing strategy to integrate the
      sequentially refined features before the GRU update.
"""

import torch
import torch.nn as nn

from dqn_cnn2 import SudokuConstraintConv
from dqn_cnn8 import AxialSudokuAttention, ConvGRUCell


class SequentialReasoningLayer(nn.Module):
    """
    Computes logical features using Cascaded Axial Attention.
    Flow: Input -> Row Attn -> Col Attn -> Box Attn -> Mixer -> Output
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()

        # 1. Row Branch
        self.norm_row = nn.GroupNorm(32, d_model)
        self.attn_row = AxialSudokuAttention(d_model, "row", num_heads)

        # 2. Col Branch
        self.norm_col = nn.GroupNorm(32, d_model)
        self.attn_col = AxialSudokuAttention(d_model, "col", num_heads)

        # 3. Box Branch
        self.norm_box = nn.GroupNorm(32, d_model)
        self.attn_box = AxialSudokuAttention(d_model, "box", num_heads)

        # 4. Mixer / FFN
        # We mix the final cascaded result with the anchor state x0
        self.mixer = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.ReLU(inplace=True),
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model * 2, d_model, kernel_size=1),
        )

    def forward(self, x, x0):
        # 1. Row Attention
        # Residual connection: x = x + Attn(Norm(x))
        residual = x
        x_norm = self.norm_row(x)
        x = residual + self.attn_row(x_norm)

        # 2. Column Attention (sees Row updates immediately)
        residual = x
        x_norm = self.norm_col(x)
        x = residual + self.attn_col(x_norm)

        # 3. Box Attention (sees Row and Col updates immediately)
        residual = x
        x_norm = self.norm_box(x)
        x = residual + self.attn_box(x_norm)

        # 4. Mix with Anchor (x0)
        # We concatenate the fully refined 'x' with the original 'x0'
        combined = torch.cat([x, x0], dim=1)
        features = self.mixer(combined)

        # FFN
        features = features + self.ffn(features)

        return features


class DQNSolverCNN9(nn.Module):
    """
    CNN9: Sequential Gated Recurrent Axial Transformer.
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

        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, 9, 9) * 0.02)

        # 2. Reasoning Modules (Sequential)
        self.reasoning_layer = SequentialReasoningLayer(d_model, num_heads=4)
        self.gru = ConvGRUCell(d_model, d_model)

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
            # Extract logical features using Sequential Attention
            features = self.reasoning_layer(x, x0)
            # Update state using GRU
            x = self.gru(features, x)

        x = self.norm_final(x)

        # 3. Output
        x = self.head(x)
        return x.permute(0, 2, 3, 1).reshape(b, -1)
