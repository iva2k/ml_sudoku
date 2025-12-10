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

    def __init__(self, d_model: int, n_heads: int, halting_bias: int | None = -1.0):
        super().__init__()
        self.reasoning = GlobalReasoningBlock(d_model, n_heads)

        # The "halting neuron" is a linear layer that maps the global average
        # of cell features to a single logit.
        self.halting_gate = nn.Sequential(
            # Input is d_model * 2 because we concatenate mean and std dev
            nn.Linear(d_model * 2, d_model // 4),  # Hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        # Initialize the bias of the final linear layer to encourage a starting probability
        # around 0.27 (sigmoid(-1.0)). This starts the model with ~4 steps,
        # allowing bidirectional learning.
        if halting_bias is not None:
            nn.init.constant_(self.halting_gate[2].bias, halting_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            - next_state (torch.Tensor): The updated feature map.
            - halt_prob (torch.Tensor): A scalar tensor (B, 1) with the probability of
              halting at this step.
        """
        # --- CRITICAL FIX: Provide richer statistics to the halting gate ---
        # Instead of just the mean, we concatenate mean and standard deviation
        # to give the gate a better signal to differentiate between states.
        mean_features = x.mean(dim=[2, 3])
        std_features = x.std(dim=[2, 3])
        global_features = torch.cat([mean_features, std_features], dim=1)

        halt_prob = self.halting_gate(global_features)  # (B, 1)

        next_state = self.reasoning(x)
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
        halting_bias: int | None = -1.0,
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
            # Use GroupNorm to prevent batch-wise homogenization
            nn.GroupNorm(num_groups=32, num_channels=d_model),
            nn.ReLU(inplace=True),
        )

        # 3. Recurrent Global Reasoning with ACT
        self.reasoning_block = ACTReasoningBlock(d_model, n_heads=4, halting_bias=halting_bias)

        # 4. Output Head: Project each cell's final embedding to 9 digit scores.
        self.fc = nn.Linear(d_model, 9)

    def set_reasoning_grad(self, requires_grad: bool):
        """Enable or disable gradients for the main reasoning components."""
        for param in self.constraint_conv.parameters():
            param.requires_grad = requires_grad
        for param in self.reduce.parameters():
            param.requires_grad = requires_grad
        for param in self.reasoning_block.reasoning.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """Forward pass."""
        b, _c, _h, _w = x.shape
        x = self.reduce(self.constraint_conv(x))

        # --- ACT Loop (Sample-Independent) ---
        halt_accum = torch.zeros((b, 1), device=self.device)
        ponder_cost = torch.zeros(b, device=self.device)
        state_sum = torch.zeros_like(x)
        # A mask to track which samples are still running
        is_running_mask = torch.ones(b, device=self.device, dtype=torch.bool)

        for _step_counter in range(self.max_steps):
            # Only update states for running samples
            x_running = x[is_running_mask]
            if x_running.numel() == 0:
                break  # All samples have halted

            # Apply the reasoning block only to the running samples.
            # The halt probability is correctly based on the *current* state `x_running`.
            x_next_running, halt_prob_running = self.reasoning_block(x_running)

            # The probability for this step is capped by the remaining budget.
            # We only update probabilities for the running samples.
            step_prob = torch.min(halt_prob_running, 1.0 - halt_accum[is_running_mask])

            # --- CRITICAL FIX: Map running probabilities back to the full batch ---
            # Create a zero tensor for the whole batch and fill in the probabilities
            # for the running samples at their correct indices.
            step_prob_full = torch.zeros_like(halt_accum)
            step_prob_full[is_running_mask] = step_prob

            halt_accum += step_prob_full
            ponder_cost += is_running_mask.float()
            # The state_sum is a weighted average of the *next* states.
            # The state sum must be weighted by the state *before* the reasoning step,
            # not after. This correctly links the halt probability to the state that produced it.
            state_sum[is_running_mask] += x[is_running_mask] * step_prob.view(-1, 1, 1, 1)

            # To avoid the in-place error, create a new tensor for the next iteration's state.
            x_next = x.clone()
            x_next[is_running_mask] = x_next_running
            x = x_next

            # Update the mask for the next iteration
            is_running_mask = halt_accum.squeeze(-1) < self.halt_threshold

        # Differentiable Ponder Cost Calculation (Required for the ACT Method)
        # The remainder is the "leftover" probability budget.
        remainder = 1.0 - halt_accum
        # The final state sum includes the state weighted by the remainder.
        state_sum += x * remainder.view(b, 1, 1, 1)

        # The total ponder cost is the number of steps taken (N) plus the remainder (R_N).
        ponder_cost += remainder.squeeze(-1)

        # --- Output ---
        final_state = state_sum.permute(0, 2, 3, 1).reshape(b, 81, -1)
        q_values = self.fc(final_state).view(b, -1)

        # Return ponder_cost for the loss function
        return q_values, ponder_cost
