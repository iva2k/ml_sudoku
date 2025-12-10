#!/usr/bin/env python3
# dqn_cnn6.py

"""
Deep Q-Network Solver (DQN) for Sudoku.
Hybrid CNN-Attention model with true Adaptive Computation Time (ACT).

1. Architecture Rationale

This model directly addresses the core observation that logical complexity, not the
number of clues, determines puzzle difficulty. It builds on the recurrent structure
of DQNSolverCNN5 but introduces a dynamic halting mechanism inspired by the
"Adaptive Computation Time" (ACT) paper.

* **True Adaptive Computation**: Unlike DQNSolverCNN5's fixed number of iterations,
  this model learns *when to stop thinking*. For each input puzzle, it can
  dynamically decide to perform more reasoning steps if the logical path is deep,
  or halt early if the state is simple.

* **How it Works**:
  1. **ACTReasoningBlock**: A modified `GlobalReasoningBlock` that, in addition to
     updating the board state, outputs a single scalar value called the "halting
     logit". A sigmoid function converts this into a "halting probability" for
     the current step.
  2. **Dynamic `while` loop**: The `forward` pass contains a `while` loop that
     iteratively applies the reasoning block. The loop continues as long as the
     cumulative sum of halting probabilities is less than a threshold (e.g., 0.99)
     and a maximum number of steps has not been exceeded.
  3. **Weighted State Averaging**: The final output state is a weighted average of
     the intermediate states produced at each step. The weights are the halting
     probabilities, so steps where the model was "more sure" about halting
     contribute more to the final answer.
  4. **Ponder Cost**: To train this behavior, a "ponder cost" is added to the
     main RL loss function. This cost is simply the number of steps the model
     took. It penalizes the model for thinking for too long, forcing it to
     balance accuracy with computational efficiency. The main training loop in
     `rl_sudoku.py` will need to be modified to accommodate this.

This architecture is the most sophisticated yet, as it allows the model's
computational depth to adapt to the logical depth of the problem itself.

"""

import torch
import torch.nn as nn

from dqn_cnn2 import SudokuConstraintConv
from dqn_cnn4 import GlobalReasoningBlock


class ACTReasoningBlock(nn.Module):
    """
    A reasoning block that includes a "halting neuron" for Adaptive Computation Time.
    It outputs both the next state and a scalar halting probability.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.reasoning = GlobalReasoningBlock(d_model, n_heads)

        # The "halting neuron" is a linear layer that maps the global average
        # of cell features to a single logit.
        self.halting_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            - next_state (torch.Tensor): The updated feature map.
            - halt_prob (torch.Tensor): A scalar tensor (B, 1) with the probability of
              halting at this step.
        """
        next_state = self.reasoning(x)

        # Global Average Pooling to get a single feature vector per puzzle
        global_features = next_state.mean(dim=[2, 3])  # (B, D_Model)
        halt_prob = self.halting_gate(global_features)  # (B, 1)

        return next_state, halt_prob


class DQNSolverCNN6(nn.Module):
    """
    The Deep Q-Network with Adaptive Computation Time (ACT).
    """

    def __init__(
        self,
        _input_shape,
        _output_size,
        device=None,
        max_steps: int = 16,
        halt_threshold: float = 0.99,
    ):
        super().__init__()
        self.device = device
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        d_model = 128

        # 1. Perception: Extract geometric constraints
        self.constraint_conv = SudokuConstraintConv(10, 48)

        # 2. Reducer: Project perception features into the reasoning dimension
        self.reduce = nn.Sequential(
            nn.Conv2d(48 * 3, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # 3. Recurrent Global Reasoning with ACT
        self.reasoning_block = ACTReasoningBlock(d_model, n_heads=4)

        # 4. Output Head: Project each cell's final embedding to 9 digit scores.
        self.fc = nn.Linear(d_model, 9)

    def forward(self, x):
        """Forward pass."""
        b, _c, _h, _w = x.shape
        x = self.reduce(self.constraint_conv(x))

        # --- ACT Loop ---
        halt_accum = torch.zeros((b, 1), device=self.device)
        step_counter = 0
        ponder_cost = torch.zeros(b, device=self.device)
        state_sum = torch.zeros_like(x)

        while (halt_accum.max() < self.halt_threshold) and (
            step_counter < self.max_steps
        ):
            x, halt_prob = self.reasoning_block(x)

            # Calculate how much of the "halting budget" is left
            remainder = 1.0 - halt_accum
            # The probability for this step is capped by the remainder
            step_prob = torch.min(halt_prob, remainder)

            halt_accum += step_prob
            # Only increment ponder cost for samples that are still running.
            is_running_mask = (halt_accum < 1.0).float().squeeze(-1)
            ponder_cost += is_running_mask
            state_sum += x * step_prob.view(b, 1, 1, 1)  # Weight state by its prob
            step_counter += 1

        # Handle any remaining probability budget if max_steps was reached
        remainder = 1.0 - halt_accum
        state_sum += x * remainder.view(b, 1, 1, 1)
        ponder_cost += remainder.squeeze(-1)

        # --- Output ---
        final_state = state_sum.permute(0, 2, 3, 1).reshape(b, 81, -1)
        q_values = self.fc(final_state).view(b, -1)

        # Return ponder_cost for the loss function
        return q_values, ponder_cost
