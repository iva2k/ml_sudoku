#!/usr/bin/env python3
# dqn_cnn2.py

"""
Deep Q-Network Solver (DQN) for Sudoku
CNN-based with kernels aligned to Sudoku rules (row/col/box).
"""

import torch
import torch.nn as nn


class SudokuConstraintConv(nn.Module):
    """
    A custom layer that respects Sudoku geometry.
    It computes features for:
    1. Entire Rows (1x9)
    2. Entire Cols (9x1)
    3. Entire Boxes (3x3 non-overlapping)
    And broadcasts them back to the grid cells.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x9 convolution finds row patterns
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 9))
        # 9x1 convolution finds column patterns
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 1))
        # 3x3 stride 3 convolution finds box patterns (non-overlapping tiling)
        self.box_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=3)

        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass."""
        # x: (B, C, 9, 9)

        # 1. Row Features: (B, Out, 9, 1)
        r = self.row_conv(x)
        # Expand back to (B, Out, 9, 9) by repeating across columns
        r = r.expand(-1, -1, -1, 9)

        # 2. Col Features: (B, Out, 1, 9)
        c = self.col_conv(x)
        # Expand back to (B, Out, 9, 9) by repeating across rows
        c = c.expand(-1, -1, 9, -1)

        # 3. Box Features: (B, Out, 3, 3)
        b = self.box_conv(x)
        # Expand back to (B, Out, 9, 9) by tiling 3x3 blocks
        b = b.repeat_interleave(3, dim=2).repeat_interleave(3, dim=3)

        # Concatenate all features: (B, Out*3, 9, 9)
        out = torch.cat([r, c, b], dim=1)
        return self.relu(self.bn(out))


class DQNSolverCNN2(nn.Module):
    """
    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.
    """

    def __init__(self, _input_shape, output_size, device=None):
        super().__init__()
        self.device = device

        # TODO: (when needed) Implement tying _input_shape to the CNN input shape.

        # 1. First Pass: Extract geometric constraints
        # Input: 10 channels (one-hot digits)
        self.constraint_conv = SudokuConstraintConv(10, 64)

        # 2. Mixing Layer: 1x1 conv to combine row/col/box info per pixel
        # Input channels = 64 * 3 = 192
        self.mix_conv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 3. Second Pass: Deepen reasoning on mixed features
        self.constraint_conv2 = SudokuConstraintConv(128, 64)

        # 4. Final Head
        # Input channels = 64 * 3 = 192
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 9 * 9, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.constraint_conv(x)
        x = self.mix_conv(x)
        x = self.constraint_conv2(x)
        return self.fc(x)
