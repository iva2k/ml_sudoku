#!/usr/bin/env python3
# dqn_cnn5.py

"""
Deep Q-Network Solver (DQN) for Sudoku.
Hybrid CNN-Attention model with Recurrent Reasoning.

1. Architecture Rationale

This model builds upon the insights from DQNSolverCNN4, addressing a key limitation:
the fixed depth of logical reasoning. While stacking `GlobalReasoningBlock`s allows for
chained deductions, the number of steps is static. However, the logical complexity of
a Sudoku puzzle is not uniform. Some puzzles require more iterative steps of
constraint propagation than others, regardless of the initial number of clues.

This model replaces the fixed stack of reasoning blocks with a single, recurrently
applied `GlobalReasoningBlock`. This has several advantages:

* **Adaptive Computation**: The model can learn to perform a variable number of
  reasoning steps. For simple board states, it might converge quickly. For complex
  states requiring deep logical chains, it can "think" for longer by iterating
  through the reasoning block multiple times.

* **Parameter Efficiency**: Instead of learning N separate reasoning blocks, the model
  learns a single, more powerful, and generalizable reasoning unit that is applied
  repeatedly. This is analogous to the difference between a feed-forward network and
  a recurrent neural network (RNN).

* **Alignment with Human Problem-Solving**: This iterative refinement process more
  closely mimics how humans solve difficult Sudokus: we scan the board, apply a
  rule, update our mental model, and then re-scan the board with this new
  information, repeating the process until a new deduction can be made.

2. Implementation

The core of this model is a `for` loop in the `forward` pass that applies the same
`GlobalReasoningBlock` `n_iterations` times. The number of iterations becomes a
hyperparameter that controls the maximum "thinking time" or logical depth the
model is allowed.

"""

# import torch
import torch.nn as nn

from dqn_cnn2 import SudokuConstraintConv
from dqn_cnn4 import GlobalReasoningBlock


class DQNSolverCNN5(nn.Module):
    """
    The Deep Q-Network with Recurrent Reasoning.
    This version uses a single GlobalReasoningBlock applied iteratively.
    """

    def __init__(self, _input_shape, _output_size, device=None, n_iterations: int = 8):
        super().__init__()
        self.device = device
        self.n_iterations = n_iterations
        d_model = 128  # Dimension for reasoning features

        # 1. Perception: Extract geometric constraints
        self.constraint_conv = SudokuConstraintConv(10, 48)

        # 2. Reducer: Project perception features into the reasoning dimension
        self.reduce = nn.Sequential(
            nn.Conv2d(48 * 3, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # 3. Recurrent Global Reasoning: A single, powerful block applied iteratively.
        self.reasoning_block = GlobalReasoningBlock(d_model, n_heads=4)

        # 4. Output Head: Project each cell's final embedding to 9 digit scores.
        self.fc = nn.Linear(d_model, 9)

    def forward(self, x):
        """Forward pass."""
        b, _c, _h, _w = x.shape

        # Stage 1: Perceive local Sudoku constraints
        x = self.constraint_conv(x)

        # Stage 2: Project into the main reasoning space
        x = self.reduce(x)

        # Stage 3: Perform iterative global reasoning by applying the same block N times.
        for _ in range(self.n_iterations):
            x = self.reasoning_block(x)

        # Stage 4: Predict Q-values
        # Permute to (B, 9, 9, D_Model) -> Reshape to (B, 81, D_Model)
        x = x.permute(0, 2, 3, 1).reshape(b, 81, -1)
        x = self.fc(x)  # (B, 81, 9)

        # Flatten to standard RL action space: (B, 729)
        return x.view(b, -1)
