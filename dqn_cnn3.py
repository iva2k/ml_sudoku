#!/usr/bin/env python3
# dqn_cnn3.py

"""
Deep Q-Network Solver (DQN) for Sudoku.
CNN-based with kernels aligned to Sudoku rules (row/col/box), with reasoning blocks.

1. Analysis of the Current DQNSolver

The current DQNSolver is quite sophisticated. It's not a simple CNN; it has specialized
components designed for Sudoku:

* **SudokuConstraintConv**: This is the model's "perception" layer. It uses specialized
   convolutions (1x9 for rows, 9x1 for columns, 3x3 with stride 3 for boxes) to gather
   information about which numbers are present in each of a cell's three constraining groups.
   It then concatenates these features. This is an excellent design for capturing the basic
   geometric constraints of the game.
* **ReasoningBlock**: This is intended to be the "logic" unit. It uses 1x1 convolutions, which
   act like small neural networks applied independently to the feature set of each of the 81
   cells. The idea is for it to learn logical rules like, "If my row features show a '5' and
   my column features show a '7', then I cannot be a 5 or 7."
* **Overall Flow**: The model perceives constraints, "thinks" about them with reasoning blocks,
   perceives them again, and then makes a decision via a fully connected head.

2. Critique of Its Weaknesses

Despite the thoughtful design, there are fundamental architectural limitations that prevent it
from learning the complex, chained logic required for difficult Sudoku puzzles.

   1. Limited Information Flow (The Core Problem): The ReasoningBlock operates on each cell in
      isolation. It can see the features for its own row, column, and box, but it cannot see
      the features of other cells. A difficult Sudoku requires reasoning like:

      "If the only two possible cells for a '7' in this box are in the same row, then no other
      cell in that entire row (even outside this box) can be a '7'."

The model cannot learn this. A cell's ReasoningBlock has no information about the
possibilities of its neighbors. It's like trying to solve a Sudoku by only looking at one cell
at a time and the numbers already placed in its groups, without considering the empty cells.

   2. Output Representation is Not Cell-Centric: The model flattens everything into a single
      vector of 729 Q-values (row * 81 + col * 9 + digit). This mixes the "where" (cell) and
      "what" (digit) into one large action space. A more natural approach for Sudoku is to
      first decide which cell to fill and then what digit to place in it. The current
      structure makes it harder for the model to reason about the properties of a single cell.

   3. Reliance on Action Masking: The agent isn't being forced to learn the basic rules
      because the generate_legal_mask function does it externally. The model doesn't get
      penalized for suggesting an illegal move (e.g., placing a '5' in a row that already has
      one); that action is simply filtered out. This is a crutch that prevents it from
      learning the most fundamental layer of reasoning.

"""

# import torch
import torch.nn as nn

from dqn_cnn2 import SudokuConstraintConv


class ReasoningBlock(nn.Module):
    """
    A logical processing unit for each cell.
    Uses 1x1 convolutions to act as a per-pixel Dense Neural Network.
    This allows the model to learn complex exclusions like:
    "If Row has 1 AND Col has 2, then I cannot be 1 or 2."
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=channels)

    def forward(self, x):
        """Forward pass."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DQNSolverCNN3(nn.Module):
    """
    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.

    """

    def __init__(self, _input_shape, output_size, device=None):
        super().__init__()
        self.device = device

        # 1. Perception: Extract geometric constraints
        # Input: 10 channels (one-hot digits)
        # Output: 64 features per Row/Col/Box -> 192 total
        self.constraint_conv = SudokuConstraintConv(10, 64)

        # 2. Reasoning: Deep per-pixel logic (1x1 Convolutions)
        # We first reduce dimensions to make reasoning efficient
        self.reduce = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(inplace=True),
        )

        # Stack of reasoning blocks (Thinking time)
        self.reasoning = nn.Sequential(
            ReasoningBlock(128),
            ReasoningBlock(128),
            ReasoningBlock(128),
        )

        # 3. Second Pass: Re-evaluate constraints based on reasoning features
        # This helps propagate complex dependencies
        self.constraint_conv2 = SudokuConstraintConv(128, 64)

        # 4. Final Decision
        # Input channels = 64 * 3 (from conv2) = 192
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 9 * 9, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        """Forward pass."""
        # Stage 1: See constraints
        x = self.constraint_conv(x)

        # Stage 2: Think about exclusions (per cell)
        x = self.reduce(x)
        x = self.reasoning(x)

        # Stage 3: Re-check context
        x = self.constraint_conv2(x)

        # Stage 4: Act
        return self.fc(x)
