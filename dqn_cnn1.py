#!/usr/bin/env python3
# dqn_cnn1.py

"""
    Deep Q-Network Solver (DQN) for Sudoku - CNN-based with 3x3 kernels.

    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.

    1. Analysis of the DQNSolver1 Architecture

    DQNSolver1 is a classic Deep Convolutional Neural Network (CNN) architecture, similar to what
    one might use for image classification tasks. It's built using standard, well-understood
    components:

       * **Initial Convolution**: It starts with a Conv2d layer that takes the 10-channel one-hot
          encoded input and transforms it into 128 feature maps. The kernel size is 3x3 with
          padding, which is a standard choice for preserving the 9x9 spatial dimensions while
          extracting local features.
       * **Residual Blocks**: The core of the network consists of three ResidualBlocks. Each block
         contains two 3x3 convolutional layers. The "residual connection" (adding the input of the
         block to its output) is a powerful technique borrowed from ResNet architectures. It helps
         in training deeper networks by allowing gradients to flow more easily, preventing the
         "vanishing gradient" problem. This allows the network to build up more complex feature
         representations layer by layer.
       * **Final Convolution and Flattening**: After the residual blocks, another convolutional
         layer processes the features, which are then flattened from a 2D grid of features
         (128 x 9 x 9) into a single long vector.
       * **Fully Connected Head**: This flattened vector is passed through two dense (Linear)
         layers, which ultimately produce the final 729 Q-values, one for each possible action
         (cell + digit).

    In essence, DQNSolver1 treats the Sudoku grid as a small 9x9 image with 10 channels and uses
    a standard deep learning approach to find patterns in it.

    2. Critique of Its Weaknesses

    While this architecture is powerful for general-purpose image analysis, it has significant
    weaknesses when applied to the specific, logical structure of Sudoku. Its main drawback is a
    mismatch between the network's inductive bias and the problem's domain logic.

       1. Lack of Sudoku-Specific Inductive Bias: The network's primary tool is the 3x3
          convolution. A 3x3 kernel is a local feature detector. It's designed to find patterns
          among a cell and its 8 immediate neighbors. However, Sudoku rules are not local in this
          way. A cell is constrained by 20 other cells spread across its entire row, its entire
          column, and its 3x3 box.

          * For a 3x3 kernel at cell (4,4) to "see" the value at cell (4,8) at the end of the row,
            the information must be passed through multiple convolutional layers. This is a very
            indirect and inefficient way to learn the fundamental rule "no two same numbers in a
            row."
          * The model has no built-in understanding of rows, columns, or non-overlapping boxes. It
            must learn these geometric concepts from scratch, which is a difficult task for a
            generic CNN.

       2. **Inefficient Information Aggregation**: Because the architecture relies on stacking
          local kernels, it struggles with long-range dependencies. The ResidualBlocks create a
          deep network, which in theory gives it a larger "receptive field" (the area of the input
          that can influence a single output feature). However, it's still a brute-force way to
          achieve the global information sharing that Sudoku requires. The network has to expend a
          lot of its capacity just to learn how to gather information from the correct cells,
          leaving less capacity for actual logical reasoning.

       3. **"Where" and "What" are Conflated**: Similar to the other CNN-based models, the final
          output is a single flat vector of 729 actions. The network must learn a complex mapping
          from its grid-based features to this unstructured action space. This makes it difficult
          to learn cell-specific properties versus digit-specific properties, as the two concepts
          are entangled in the final output layer.

    In summary, DQNSolver1 is a powerful but generic tool. It's like giving a carpenter a high-end
    sledgehammer for a task that requires a precision screwdriver. It might eventually get the job
    done for very simple puzzles by sheer force (i.e., memorizing patterns), but it lacks the
    architectural finesse to learn the explicit, rule-based, long-range reasoning needed for
    advanced Sudoku. The DQNSolver (with SudokuConstraintConv) and the DQNSolverTransformer are
    significant improvements because their architectures are explicitly designed to respect the
    row, column, and box geometry of the game.

"""

# import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Forward pass through the residual block."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DQNSolverCNN1(nn.Module):
    """
    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.
    """

    def __init__(self, _input_shape, output_size, device=None):
        super().__init__()
        self.device = device

        # TODO: (when needed) Implement tying _input_shape to the CNN input shape.

        # Calculate the size after convolutional layers (9x9 grid remains 9x9 with padding=1, k=3)
        # 128 channels * 9 rows * 9 cols = 10368
        self.flattened_size = 128 * 9 * 9

        self.net = nn.Sequential(
            # Initial convolution to get to the desired channel dimension
            nn.Conv2d(10, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Stack of residual blocks to deepen the network effectively
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Flatten(),

            # Fully Connected Layers (DNN head)
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_size)  # Final output is 729 Q-values
        )

    def forward(self, x):
        """
        Forward pass through the CNN and then the fully connected layers.
        Input x: (batch_size, 10, 9, 9) for one-hot encoded state.
        """
        # Input is already a one-hot tensor
        x = x.to(self.device)

        q_values = self.net(x)
        return q_values
