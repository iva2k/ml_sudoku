#!/usr/bin/env python3
# rl_sudoku.py

"""Sudoku RL agent using PyTorch.

Implemented in 3 phases as a tutorial,
## Phase 1: Create the environment and code carcass

We define the Environment, the Network, and the Replay Buffer
 - all imports
 - class SudokuEnv - custom gymnasium environment
 - class ReplayBuffer, Transition data structure
 - class DQNSolver
 - get_action(), optimize_model()
 - main(), parse_args()

## Phase 2: Implement Core Logic / Functioning Environment
 - Reward Shaping logic
 - Action Masking logic

## Phase 3: Generalization and Training
 - Procedural content generation for the environment
"""

import argparse
from collections import deque, namedtuple
import random
from typing import Any, Optional
import time
from datetime import timedelta

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# TODO: (when needed) Adjust these during Phase 3
# Default Hyperparameters & Epsilon-greedy params (all can be changed from command line):
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.99995
TARGET_UPDATE = 50  # Frequency (in episodes) to update the target network
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
LR = 0.0001
MAX_EPISODES = 50000
WEIGHT_DECAY = 0.01

# Large Negative reward to suppress known illegal actions
ILLEGAL_ACTION_VALUE = -1e10


def _generate_initial_grid1() -> np.ndarray:
    """Generates a complete, solved Sudoku grid using a randomized backtracking algorithm."""
    grid = np.zeros((9, 9), dtype=np.int32)
    nums = list(range(1, 10))

    def find_empty(g):
        for r in range(9):
            for c in range(9):
                if g[r, c] == 0:
                    return (r, c)
        return None

    def is_valid(g, num, pos):
        r, c = pos
        # Check row
        if num in g[r, :]:
            return False
        # Check column
        if num in g[:, c]:
            return False
        # Check 3x3 box
        box_r, box_c = 3 * (r // 3), 3 * (c // 3)
        if num in g[box_r:box_r + 3, box_c:box_c + 3]:
            return False
        return True

    def solve(g):
        find = find_empty(g)
        if not find:
            return True  # Solved
        else:
            row, col = find

        random.shuffle(nums)  # Randomize numbers to try
        for num in nums:
            if is_valid(g, num, (row, col)):
                g[row, col] = num
                if solve(g):
                    return True
                g[row, col] = 0  # Backtrack
        return False

    solve(grid)
    return grid.astype(np.int32)


def _generate_initial_grid() -> np.ndarray:
    """Generates a complete, solved Sudoku grid using constraint-preserving transformations."""
    def _shuffle_digits(grid: np.ndarray) -> np.ndarray:
        """Permutes the digits (1-9) across the entire grid."""
        mapping = list(range(1, 10))
        random.shuffle(mapping)  # e.g., [5, 8, 1, 4, 3, 7, 2, 9, 6]
        shuffled_grid = np.zeros_like(grid)
        for old_val in range(1, 10):
            new_val = mapping[old_val - 1]
            shuffled_grid[grid == old_val] = new_val
        return shuffled_grid

    def _swap_major_rows(grid: np.ndarray) -> np.ndarray:
        """Permutes the 3 major row blocks (rows 0-2, 3-5, 6-8)."""
        row_blocks = np.split(
            grid, 3, axis=0)  # Split into 3 arrays of shape (3, 9)
        random.shuffle(row_blocks)
        return np.concatenate(row_blocks, axis=0)

    def _swap_major_cols(grid: np.ndarray) -> np.ndarray:
        """Permutes the 3 major column blocks (cols 0-2, 3-5, 6-8)."""
        col_blocks = np.split(
            grid, 3, axis=1)  # Split into 3 arrays of shape (9, 3)
        random.shuffle(col_blocks)
        return np.concatenate(col_blocks, axis=1)

    def _shuffle_minor_rows(grid: np.ndarray) -> np.ndarray:
        """Shuffles rows within each of the three 3x3 row blocks."""
        new_grid = grid.copy()
        for i in range(3):
            # Indices for the current row block (e.g., 0, 1, 2)
            indices = list(range(i * 3, (i + 1) * 3))
            random.shuffle(indices)
            new_grid[i * 3: (i + 1) * 3, :] = grid[indices, :]
        return new_grid

    def _shuffle_minor_cols(grid: np.ndarray) -> np.ndarray:
        """Shuffles columns within each of the three 3x3 column blocks."""
        new_grid = grid.copy()
        for i in range(3):
            # Indices for the current column block (e.g., 0, 1, 2)
            indices = list(range(i * 3, (i + 1) * 3))
            random.shuffle(indices)
            new_grid[:, i * 3: (i + 1) * 3] = grid[:, indices]
        return new_grid

    base = 3
    side = base * base

    # 1. Create Base Grid (Canonical Pattern)
    # This formula is guaranteed to produce a solved grid
    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side
    nums = np.array(list(range(1, side + 1)))  # [1, 2, ..., 9]

    # Initial canonical solved grid
    grid = np.array([[nums[pattern(r, c)] for c in range(side)]
                    for r in range(side)], dtype=np.int32)

    # 2. Apply Permutations (for variety)
    # This order is safe and preserves all Sudoku rules:
    grid = _shuffle_digits(grid)
    grid = _swap_major_rows(grid)
    grid = _swap_major_cols(grid)
    grid = _shuffle_minor_rows(grid)
    grid = _shuffle_minor_cols(grid)

    return grid


def _clue_grid(solved_grid: np.ndarray, num_clues: int = 30) -> np.ndarray:
    """Removes numbers from a solved grid to create a puzzle."""
    clue_grid = solved_grid.copy()
    # Create a list of all 81 indices (r, c)
    all_indices = [(r, c) for r in range(9) for c in range(9)]

    # Shuffle the indices and remove all but the first 'num_clues'
    random.shuffle(all_indices)

    # Determine how many cells to keep (the clues)
    cells_to_keep = set(all_indices[:num_clues])

    for r in range(9):
        for c in range(9):
            if (r, c) not in cells_to_keep:
                clue_grid[r, c] = 0
    return clue_grid


def action_encode(row, col, digit):
    """Encode action from row, col, digit."""
    return row * 81 + col * 9 + (digit - 1)


def action_decode(action):
    """Decode action to row, col, digit."""
    row = (action // 81) % 9
    col = (action // 9) % 9
    digit = (action % 9) + 1  # Digits are 1-9
    return row, col, digit


def generate_legal_mask(grid):
    """
    Generates a boolean mask of size 729 marking legal actions. True indicates a legal action: 
    placing any digit (1-9) in an empty cell (value 0).
    """
    mask = np.zeros(9 * 9 * 9, dtype=bool)
    for r in range(9):
        for c in range(9):
            # An action is only legal if the target cell is currently empty
            if grid[r, c] == 0:
                for d in range(1, 10):
                    action_idx = action_encode(r, c, d)
                    mask[action_idx] = True
    return mask


# State/action transition data structure for the Replay Buffer
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class SudokuEnv(gym.Env):
    """
    A custom Gymnasium environment for the 9x9 Sudoku puzzle.
    State: The current 9x9 grid.
    Action: (row, col, digit) to place. Total 729 actions.
    """

    def __init__(
        self,
        puzzle_str: Optional[str] = None,
        sol_str: Optional[str] = None,
        reward_shaping: bool = False,
        fixed_puzzle: bool = False
    ):
        super().__init__()

        # 9 rows, 9 columns, digits 1-9.
        # Action space is discrete, but we'll map 0-728 to (row, col, digit)
        self.action_space = gym.spaces.Discrete(9 * 9 * 9)
        # Observation space is 9x9 grid, values 0 (empty) to 9
        self.observation_space = gym.spaces.Box(
            low=0, high=9, shape=(9, 9), dtype=np.int32)

        # RL options
        self.reward_shaping = reward_shaping
        self.fixed_puzzle = fixed_puzzle

        # Initial puzzle (0 for empty cells)
        self.default_puzzle, self.default_solution = self._parse_puzzle(puzzle_str, sol_str)
        self.initial_puzzle = self.default_puzzle.copy()
        self.solution_grid = self.default_solution.copy()
        self.current_grid = self.default_puzzle.copy()
        self.violation_count = 0
        print(
            f"Sudoku Environment Initialized. "
            f"Shaping: {self.reward_shaping}, "
            f"Generation: {not self.fixed_puzzle}"
        )

    def _parse_puzzle(self, puzzle_str: Optional[str], sol_str: Optional[str]):
        """Converts an 81-char string (0s for empty) into a 9x9 numpy array."""
        if puzzle_str is None and sol_str is None:
            # Default easy puzzle for starting development
            puzzle_str = "000260701680070090190004500820100040004602900050003028009300074040050036703018000"  # Puzzle
            sol_str = "435269781682571493197834562826195347374682915951743628519326874248957136763418259"  # Solution

        # Ensure the string is 81 characters long and contains only digits
        if len(puzzle_str) != 81 or not puzzle_str.isdigit() or "0" not in puzzle_str:
            raise ValueError(f"Puzzle string must be 81 digits (0-9), must contain zeroes to be a puzzle, {len(puzzle_str)} provided.")
        if len(sol_str) != 81 or not sol_str.isdigit() or "0" in sol_str:
            raise ValueError(f"Solution string must be 81 digits (1-9), no 0's, {len(sol_str)} provided.")

        grid = np.array([int(c) for c in puzzle_str]).reshape((9, 9))
        sol_grid = np.array([int(c) for c in sol_str]).reshape((9, 9))
        return grid, sol_grid

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment to a new (or default) puzzle state."""
        super().reset(seed=seed, options=options)

        if self.fixed_puzzle:
            # Use the default puzzle
            self.initial_puzzle = self.default_puzzle.copy()
            self.solution_grid = self.default_solution.copy()
            # The default_puzzle and default_solution should have been set during __init__
        else:
            # Generate a new, random, solvable puzzle
            self.solution_grid = _generate_initial_grid()
            # Keep between 25 and 35 clues for a mix of difficulties
            num_clues = random.randint(25, 35)
            self.initial_puzzle = _clue_grid(self.solution_grid, num_clues=num_clues)

        self.current_grid = self.initial_puzzle.copy()

        if self.reward_shaping:
            # Initialize violation count for the start of the episode
            self.violation_count = self._get_violation_count(self.current_grid)

        observation = self.current_grid.astype(np.float32)
        info = {}
        return observation, info

    def step(self, action):
        """
        Applies the action (0-728) to the grid and returns next_state, reward, done, info.
        """
        # Map the single integer action back to (row, col, digit)
        row, col, digit = action_decode(action)

        reward = 0.0
        terminated = False
        truncated = False

        # 1. An action is illegal if it overwrites an existing number.
        # if self.initial_puzzle[row, col] != 0:
        #     reward = -10.0  # Penalty for overwriting a fixed cell
        # el
        if self.current_grid[row, col] != 0:
            # Penalty for wasting a step, but action masking should prevent this exploitation
            reward = -10.0
        # 2. Check if the chosen digit matches the ground-truth solution.
        else:
            if self.solution_grid[row, col] == digit:
                # Correct move: place the digit and give a positive reward.
                self.current_grid[row, col] = digit
                reward = 5.0

                # 3. Check if the puzzle is fully solved.
                # This happens if all cells are filled AND they match the solution.
                if np.all(self.current_grid != 0) and np.array_equal(self.current_grid, self.solution_grid):
                    reward += 100.0  # Large reward for solving
                    terminated = True
            else:
                # TODO: We still place the digit to let the agent see the consequences of its mistake.
                # Incorrect move: place the digit but give a penalty.
                # The problem with placing wrong digit is in the dead-end path.
                # The agent is penalized for overwriting placed digits, so there is no way out of the dead-end state.
                # self.current_grid[row, col] = digit

                reward = -5.0  # Penalty for an invalid move (violates rules)

        # State is returned as float32 for PyTorch compatibility
        observation = self.current_grid.astype(np.float32)
        info = {"current_reward": reward}

        return observation, reward, terminated, truncated, info

    def _get_violation_count(self, grid):
        """Calculates the total number of rule violations (duplicates) in the entire grid."""
        violations = 0

        # Helper to calculate violations in a single 1D array (row, col, or flattened box)
        def check_violations_1d(arr):
            # Only consider digits 1-9
            counts = np.bincount(arr[arr > 0], minlength=10)[1:]
            # Each digit count > 1 contributes (count - 1) violations
            return np.sum(counts[counts > 1] - 1)

        # Check all rows
        for i in range(9):
            violations += check_violations_1d(grid[i, :])

        # Check all columns
        for i in range(9):
            violations += check_violations_1d(grid[:, i])

        # Check all 3x3 boxes
        for i in range(3):
            for j in range(3):
                box = grid[i*3:(i+1)*3, j*3:(j+1)*3].flatten()
                violations += check_violations_1d(box)

        return violations

    def _is_valid_placement(self, grid, r, c, d):
        """
        Checks if placing digit 'd' at (r, c) is a valid move according to Sudoku rules.
        A temporary grid check is not needed here as we rely on the `step` function 
        to handle placing the digit only after this check passes.
        """
        # print(f"Checking placement at {r}, {c} with digit {d}")
        # print(f"Placement {r}, {c}: {'  ' * (c + 9 * r)}{d}")

        # Check if 'd' is already in the same row or column
        if d in grid[r, :] or d in grid[:, c]:
            return False

        # Check 3x3 subgrid
        start_r, start_c = 3 * (r // 3), 3 * (c // 3)
        box = grid[start_r:start_r + 3, start_c:start_c + 3]
        if d in box:
            return False

        # print(f"Placement {r}, {c}: {d} is valid")
        return True

    def _check_group(self, group):
        """Helper to check if a 9-element group (row, col, or box) is solved."""
        if len(group) != 9:
            return False
        # Check for 9 unique non-zero digits
        return len(np.unique(group)) == 9 and np.all(group > 0)

    def _is_fully_solved(self, grid):
        """Checks if the grid is full and all rows, columns, and 3x3 boxes are valid."""
        # 1. Check if there are any empty cells (0s) left.
        if np.any(grid == 0):
            return False

        # 2. Check all rows and columns for uniqueness (must contain 1-9)
        for i in range(9):
            if not self._check_group(grid[i, :]):  # Row check
                return False
            if not self._check_group(grid[:, i]):  # Column check
                return False

        # 3. Check all 3x3 boxes
        for i in range(3):
            for j in range(3):
                # Flatten the 3x3 box into a 9-element group
                box = grid[i*3:(i+1)*3, j*3:(j+1)*3].flatten()
                if not self._check_group(box):
                    return False

        return True

    def render(self):
        """Prints the current state of the Sudoku grid."""
        print(self.format_grid_to_string(self.current_grid))

    @staticmethod
    def format_grid_to_string(grid: np.ndarray) -> str:
        """Formats a 9x9 grid for pretty printing, replacing 0s with spaces."""
        s = ""
        for i in range(9):
            if i > 0 and i % 3 == 0:
                s += "------+-------+------\n"

            row_str = []
            for j in range(9):
                digit = grid[i, j]
                row_str.append(str(digit) if digit != 0 else " ")

            s += " ".join(row_str[0:3]) + " | " + " ".join(row_str[3:6]
                                                           ) + " | " + " ".join(row_str[6:9]) + "\n"
        return s

    def close(self):
        """Clean up resources."""
        pass


class ReplayBuffer:
    """Stores experience tuples to stabilize training."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Returns a random batch of transitions."""
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNSolver(nn.Module):
    """
    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.
    """

    def __init__(self, input_shape, output_size, device=None):
        super().__init__()
        self.device = device

        # Use CNN to capture local structure (rows, columns, 3x3 boxes)
        self.conv = nn.Sequential(
            # 1. First Conv: 1 input channel (the grid), 64 output channels
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # 2. Second Conv
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            # 3. Third Conv
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate the size after convolutional layers (9x9 grid remains 9x9 with padding=1, k=3)
        # 128 channels * 9 rows * 9 cols = 10368
        self.flattened_size = 128 * 9 * 9

        # Fully Connected Layers (DNN head)
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)  # Final output is 729 Q-values
        )

    def forward(self, x):
        """
        Forward pass through the CNN and then the fully connected layers.
        Input x: (batch_size, 9, 9). Needs to be reshaped to (batch_size, 1, 9, 9)
        """
        # Ensure input is float and has the channel dimension
        x = x.float().unsqueeze(1)

        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        q_values = self.fc(x)
        return q_values


def get_action(
    state,
    policy_net,
    action_space,
    epsilon,
    eps_end,
    eps_decay,
    use_masking: bool = False
):
    """
    Implements the epsilon-greedy policy.
    Returns the chosen action and the new epsilon value after decay.
    """
    # Use max to ensure we don't go below EPS_END
    current_epsilon = max(eps_end, epsilon)

    # Epsilon decay
    new_epsilon = epsilon * eps_decay

    mask = None
    if use_masking:
        mask = generate_legal_mask(state)

    if random.random() < current_epsilon:
        # Explore: Choose a random action
        if use_masking:
            # Sample only from legal actions
            legal_actions = np.where(mask)[0]
            if len(legal_actions) == 0:
                # Should not happen in a solvable puzzle, but return a safe action (first one)
                action = 0
            else:
                action = random.choice(legal_actions)
        else:
            action = action_space.sample()
    else:
        # Exploit: Choose the action with the highest predicted Q-value
        with torch.no_grad():
            # Add batch dimension and get Q-values
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=policy_net.device).unsqueeze(0)
            q_values = policy_net(state_tensor)

            if use_masking:
                # Apply mask: set Q-values of illegal actions to a very small number
                # Create a mask with 0 for legal actions and a large negative value for illegal ones
                mask_tensor = torch.from_numpy(mask).to(policy_net.device)
                additive_mask = torch.where(mask_tensor,
                                            torch.tensor(
                                                0.0, device=policy_net.device),
                                            torch.tensor(ILLEGAL_ACTION_VALUE,
                                                         device=policy_net.device)
                                            ).unsqueeze(0)
                masked_q_values = q_values + additive_mask
                action = masked_q_values.argmax().item()
            else:
                # Use argmax to get the best action index
                action = q_values.argmax().item()

    return action, new_epsilon


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    memory,
    batch_size,
    gamma,
    use_masking: bool = False
):
    """
    Performs one step of optimization on the Policy Network.

    BELLMAN LOSS AND BACKPROP
    """
    transitions = memory.sample(batch_size)
    if not transitions:
        return

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Convert batch of transitions to tensors
    non_final_mask = torch.tensor(
        tuple(map(lambda d: not d, batch.done)), dtype=torch.bool)

    device = policy_net.device
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.tensor(
        batch.action, device=device).unsqueeze(1)  # [B, 1]
    reward_batch = torch.tensor(
        batch.reward, dtype=torch.float32, device=device)

    # Compute Q(s_t, a) - the Q-values for the actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) = max_a Q(s_{t+1}, a) for non-terminal next states
    next_state_values = torch.zeros(batch_size)
    non_final_next_states = torch.stack(
        [s for s, done in zip(batch.next_state, batch.done) if not done]).to(device)
    if non_final_next_states.numel() > 0:
        with torch.no_grad():
            target_q_values = target_net(non_final_next_states)

            if use_masking:
                # Generate and apply mask to target Q-values
                next_states_mask_np = np.stack(
                    [generate_legal_mask(s.numpy()) for s in non_final_next_states])
                next_states_mask_tensor = torch.from_numpy(
                    next_states_mask_np).to(device)

                # Create a mask with 0 for legal actions and a large negative value for illegal ones
                additive_mask = torch.where(next_states_mask_tensor, torch.tensor(
                    0.0, device=device), torch.tensor(ILLEGAL_ACTION_VALUE, device=device))
                masked_q_values = target_q_values + additive_mask

                # Take the max over the masked Q-values
                next_state_values[non_final_mask] = masked_q_values.max(1)[
                    0].detach()
            else:
                # Use standard max Q-value
                next_state_values[non_final_mask] = target_q_values.max(1)[
                    0].detach()

    # Compute the expected Q values (target)
    expected_state_action_values = (
        next_state_values.to(device) * gamma) + reward_batch

    # Compute Huber loss (a robust form of MSE) - less sensitive to outliers
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping (optional but recommended for stability)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Q-Learning Sudoku Solver")
    # Training arguments
    parser.add_argument('--episodes', type=int,
                        default=MAX_EPISODES, help='Number of episodes to train.')
    parser.add_argument('--puzzle', type=str, default=None,
                        help='Initial Sudoku puzzle string (81 chars, 0 for empty).')
    parser.add_argument('--reward_shaping', action='store_true',
                        help='Enable reward shaping (progress-based rewards).')
    parser.add_argument('--masking', action='store_true',
                        help='Enable action masking (only choose empty cells).')
    parser.add_argument('--fixed_puzzle', action='store_true',
                        help='Use only given puzzle for training.')

    # Hyperparameter arguments
    parser.add_argument('--lr', type=float, default=LR,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help='Discount factor for future rewards.')
    parser.add_argument('--batch_size', type=int,
                        default=BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--memory_capacity', type=int,
                        default=MEMORY_CAPACITY, help='Capacity of the replay buffer.')
    parser.add_argument('--target_update', type=int, default=TARGET_UPDATE,
                        help='Frequency (in episodes) to update the target network.')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay for the AdamW optimizer.')
    # Epsilon-greedy arguments
    parser.add_argument('--eps_start', type=float, default=EPS_START,
                        help='Starting value of epsilon for exploration.')
    parser.add_argument('--eps_end', type=float,
                        default=EPS_END, help='Minimum value of epsilon.')
    parser.add_argument('--eps_decay', type=float,
                        default=EPS_DECAY, help='Decay rate for epsilon.')

    parser.add_argument('--log_episodes', type=int,
                        default=20, help='Log info once every N episodes.')
    args = parser.parse_args()

    # Add device to args
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    return args, parser


def main():
    """
    Main training loop.
    """

    args, _parser = parse_args()

    # 1. Initialize Environment, Networks, and Optimizer
    env = SudokuEnv(puzzle_str=args.puzzle,
                    reward_shaping=args.reward_shaping,
                    fixed_puzzle=args.fixed_puzzle)

    input_shape = env.observation_space.shape
    output_size = env.action_space.n  # 729

    policy_net = DQNSolver(input_shape, output_size,
                           args.device).to(args.device)
    target_net = DQNSolver(input_shape, output_size,
                           args.device).to(args.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is always in evaluation mode

    optimizer = optim.AdamW(policy_net.parameters(),
                            lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    memory = ReplayBuffer(args.memory_capacity)

    # Training loop
    steps_done = 0
    current_epsilon = args.eps_start
    best_reward = -float('inf')
    solved_count = 0

    print(f"Starting Training for {args.episodes} episodes...")
    start_time = time.time()

    for i_episode in range(1, args.episodes + 1):
        # 2. Reset the environment and get initial state
        state, _ = env.reset()

        episode_reward = 0
        episode_solved = False

        # 3. Run the episode
        for _t in range(81):  # Max 81 steps (cells) per episode
            action, current_epsilon = get_action(
                state,
                policy_net,
                env.action_space,
                current_epsilon,
                args.eps_end,
                args.eps_decay,
                args.masking
            )

            # 4. Take action in environment
            observation, reward, terminated, truncated, _info = env.step(
                action)
            next_state = observation
            done = terminated or truncated

            # 5. Store the transition in the replay memory
            # Convert numpy arrays to tensors for storage
            state_tensor = torch.from_numpy(state).float()  # Stored on CPU
            next_state_tensor = torch.from_numpy(
                next_state).float()  # Stored on CPU
            memory.push(state_tensor, action, reward, next_state_tensor, done)

            # Move to the next state
            state = next_state
            episode_reward += reward

            # 6. Perform optimization step
            optimize_model(policy_net, target_net, optimizer,
                           memory, args.batch_size, args.gamma, args.masking)
            steps_done += 1

            if terminated:
                episode_solved = True
                solved_count += 1
                break
            if truncated:
                # Typically not used in Sudoku, but good practice
                break

        # 7. Update the target network periodically
        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 8. Logging and Reporting
        if best_reward == -float('inf') or episode_reward > best_reward:
            # if best_reward != -float('inf'):
            print(
                f"Episode {i_episode}: New Best Reward: {episode_reward:.2f}")
            best_reward = episode_reward
            # TODO: (when needed)) Save model checkpoint here

        if episode_solved or i_episode % args.log_episodes == 0:
            if not env.fixed_puzzle:
                print(f"Episode {i_episode}: New puzzle:")
                print(SudokuEnv.format_grid_to_string(env.initial_puzzle))
            print(
                f"Episode: {i_episode}/{args.episodes} (Solved: {solved_count}), "
                f"Steps: {steps_done}, "
                f"Total Reward: {episode_reward:.2f}, "
                f"Epsilon: {max(args.eps_end, current_epsilon):.4f}"
            )

    end_time = time.time()
    training_duration = end_time - start_time
    duration_str = str(timedelta(seconds=training_duration))

    print(
        f"\nTraining Complete. "
        f"Final Best Reward: {best_reward:.2f} over {args.episodes} episodes. "
        f"Total Solved: {solved_count}. "
        f"Total time: {duration_str}"
    )

    # --- Run a final test episode and display the solved grid ---
    print("\n--- Running Final Test Episode ---")
    state, _ = env.reset()
    print("Initial Puzzle:")
    print(SudokuEnv.format_grid_to_string(env.initial_puzzle))

    for _t in range(81):  # Max 81 steps
        # In test mode, always exploit (no exploration)
        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=args.device).unsqueeze(0)
            q_values = policy_net(state_tensor)  # [1, 729]

            if args.masking:
                mask = generate_legal_mask(state)
                mask_tensor = torch.from_numpy(mask).to(args.device)
                additive_mask = torch.where(mask_tensor,
                                            torch.tensor(
                                                0.0, device=args.device),
                                            torch.tensor(
                                                ILLEGAL_ACTION_VALUE, device=args.device)
                                            ).unsqueeze(0)
                masked_q_values = q_values + additive_mask
                action = masked_q_values.argmax().item()
            else:
                action = q_values.argmax().item()

        observation, _, terminated, truncated, _ = env.step(action)
        state = observation
        if terminated or truncated:
            break

    print("\nFinal Grid:")
    env.render()

    print("\nDelta (Agent's moves):")
    delta_grid = env.current_grid - env.initial_puzzle
    print(SudokuEnv.format_grid_to_string(delta_grid))


if __name__ == "__main__":
    main()
    # TODO: (when needed) Save trained model to file, load from file, continue training, test model.
