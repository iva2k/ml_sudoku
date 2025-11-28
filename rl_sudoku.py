#!/usr/bin/env python3
# rl_sudoku.py
# pylint:disable=too-many-lines

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
import ctypes
import math
import platform
import random
from typing import Any, List, Optional, Tuple
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
EPS_END = 0.05  # A slightly higher floor can encourage exploration on harder puzzles
EPS_DECAY = 0.9997  # Faster decay to encourage exploitation sooner
TARGET_UPDATE = 10  # Frequency (in episodes) to update the target network
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128  # Larger batch size can stabilize training
LR = 0.00025  # Slightly higher learning rate
MAX_EPISODES = 50000
WEIGHT_DECAY = 0.01

# Large Negative reward to suppress known illegal actions
ILLEGAL_ACTION_VALUE = -1e10


def _generate_initial_grid1() -> np.ndarray:
    """Generates a complete, solved Sudoku grid using a randomized backtracking algorithm."""
    grid = np.zeros((9, 9), dtype=np.int32)
    nums = list(range(1, 10))

    def find_blank(g):
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
        find = find_blank(g)
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


def check_move_validity(grid: np.ndarray, r: int, c: int, num: int) -> bool:
    """
    Fast Numba-friendly style check for Sudoku validity.
    Returns True if placing 'num' at (r, c) does not violate row/col/box rules.
    """
    # Check row
    if num in grid[r, :]:
        return False
    # Check column
    if num in grid[:, c]:
        return False
    # Check 3x3 box
    box_r, box_c = 3 * (r // 3), 3 * (c // 3)
    if num in grid[box_r:box_r + 3, box_c:box_c + 3]:
        return False
    return True


def generate_legal_mask(grid: np.ndarray) -> np.ndarray:
    """
    Optimized STRICT MASKING (CPU version). Generates a boolean mask of size 729.
    An action is legal (True) if it places a valid digit in an empty cell.
    This version uses Python sets for fast lookups and is highly efficient on the CPU.
    """
    mask = np.zeros(9 * 9 * 9, dtype=bool)

    # 1. Pre-compute sets of used numbers for all rows, cols, and boxes for O(1) lookups.
    rows = [set(grid[r, :]) - {0} for r in range(9)]
    cols = [set(grid[:, c]) - {0} for c in range(9)]
    boxes = [[set() for _ in range(3)] for _ in range(3)]
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                boxes[r // 3][c // 3].add(grid[r, c])

    all_digits = set(range(1, 10))

    # 2. Iterate only through empty cells to find valid moves.
    empty_cells = np.argwhere(grid == 0)
    for r, c in empty_cells:
        # Find used digits for this specific cell using fast set unions
        used_digits = rows[r] | cols[c] | boxes[r // 3][c // 3]
        valid_digits = all_digits - used_digits

        for d in valid_digits:
            action_idx = action_encode(r, c, d)
            mask[action_idx] = True

    return mask


def generate_legal_mask_gpu(grid_tensor: torch.Tensor) -> torch.Tensor:
    """
    Experimental GPU-accelerated version of generate_legal_mask.
    This function performs the same logic using parallel tensor operations on the GPU.
    """
    _device = grid_tensor.device

    # 1. Create a (9, 9, 9) tensor where `used_digits[r, c, d-1]` is true if digit `d` is used
    # in the row, column, or box corresponding to cell (r, c).

    # One-hot encode the grid: (9, 9, 10) -> (9, 9, 9) for digits 1-9
    one_hot = torch.nn.functional.one_hot(
        grid_tensor.long(), num_classes=10)[:, :, 1:].bool()

    # Row and column used digits: (9, 9) booleans for each digit
    row_used = one_hot.any(dim=1, keepdim=True).expand(-1, 9, -1)
    col_used = one_hot.any(dim=0, keepdim=True).expand(9, -1, -1)

    # Box used digits: Reshape and check
    box_used = one_hot.view(3, 3, 3, 3, 9).any(dim=(1, 3), keepdim=True)
    box_used = box_used.expand(3, 3, 3, 3, 9).reshape(9, 9, 9)

    # Combine all used digits. `used_mask[r, c, d-1]` is true if d is NOT a valid digit for (r,c)
    used_mask = row_used | col_used | box_used

    # 2. Create a mask for empty cells. `empty_mask[r, c]` is true if the cell is empty.
    empty_mask = grid_tensor == 0

    # 3. The final legal mask is where a cell is empty AND the digit is NOT used.
    # We expand empty_mask to match the shape of used_mask.
    legal_mask_3d = empty_mask.unsqueeze(2) & ~used_mask
    return legal_mask_3d.view(-1)  # Flatten to 729


def state_to_one_hot(grid: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts a 9x9 grid of 0-9 values to a 10x9x9 one-hot encoded tensor on the given device."""
    # Create a 10x9x9 tensor of zeros
    one_hot = torch.zeros((10, 9, 9), dtype=torch.float32, device=device)
    # Use advanced indexing to set the '1's.
    # For each cell (r, c), the value grid[r, c] determines the channel.
    rows, cols = np.indices((9, 9))
    one_hot[grid, rows, cols] = 1
    # The result is already on the correct device
    return one_hot


# State/action transition data structure for the Replay Buffer
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done', 'episode'))


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
        fixed_puzzle: bool = False,
        block_wrong_moves: bool = False,
    ):
        super().__init__()

        # 9 rows, 9 columns, digits 1-9.
        # Action space is discrete, but we'll map 0-728 to (row, col, digit)
        self.action_space = gym.spaces.Discrete(9 * 9 * 9)
        # Observation space is 9x9 grid, values 0 (blank) to 9
        self.observation_space = gym.spaces.Box(
            low=0, high=9, shape=(9, 9), dtype=np.int32)

        # RL options
        self.reward_shaping = reward_shaping
        self.fixed_puzzle = fixed_puzzle
        self.block_wrong_moves = block_wrong_moves

        # Initial puzzle (0 for blank cells)
        self.default_puzzle, self.default_solution = self._parse_puzzle(
            puzzle_str, sol_str)
        self.initial_puzzle = self.default_puzzle.copy()
        self.solution_grid = self.default_solution.copy()
        self.current_grid = self.default_puzzle.copy()
        self.violation_count = 0
        self.rewarded_rows = set()
        self.rewarded_cols = set()
        self.rewarded_boxes = set()
        self.episode_stats = {}
        print(
            f"Sudoku Environment Initialized. "
            f"Puzzle Source: {'Fixed' if self.fixed_puzzle else 'Generated'}, "
            # f"Reward Shaping: {'Enabled' if self.reward_shaping else 'Disabled'}"
            f"Reward Shaping: {self.reward_shaping}"
        )

    def _parse_puzzle(
        self,
        puzzle_str: Optional[str],
        sol_str: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts an 81-char string (0 for blank) into a 9x9 numpy array."""
        if puzzle_str is None and sol_str is None:
            # Default easy puzzle for starting development
            puzzle_str, sol_str = (
                "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
                "435269781682571493197834562826195347374682915951743628519326874248957136763418259"
            )

        # Ensure the string is 81 characters long and contains only digits
        if len(puzzle_str) != 81 or not puzzle_str.isdigit() or "0" not in puzzle_str:
            raise ValueError(
                f"Puzzle string must be 81 digits (0-9), must contain zeroes to be a puzzle, "
                f"{len(puzzle_str)} provided."
            )
        if len(sol_str) != 81 or not sol_str.isdigit() or "0" in sol_str:
            raise ValueError(
                f"Solution string must be 81 digits (1-9), no 0's, {len(sol_str)} provided.")

        grid = np.array([int(c) for c in puzzle_str]).reshape((9, 9))
        sol_grid = np.array([int(c) for c in sol_str]).reshape((9, 9))
        return grid, sol_grid

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None
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

            num_clues = random.randint(25, 55)  # Default value
            num_clues = options.get(
                "num_clues", num_clues) if options else num_clues
            self.initial_puzzle = _clue_grid(self.solution_grid, num_clues)

        self.current_grid = self.initial_puzzle.copy()

        if self.reward_shaping:
            # Initialize violation count for the start of the episode
            self.violation_count = self._get_violation_count(self.current_grid)

        # Initialize per-episode statistics
        self.episode_stats = {
            "blank_cells_start": np.count_nonzero(self.initial_puzzle == 0),
            "correct_moves": 0,
            "completed_rows": 0,
            "completed_cols": 0,
            "completed_boxes": 0,
        }
        # Track completed groups to avoid giving rewards multiple times
        self.rewarded_rows = set()
        self.rewarded_cols = set()
        self.rewarded_boxes = set()

        observation = self.current_grid.astype(np.float32)
        return observation, self.episode_stats

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
                # Correct move: place the digit
                self.current_grid[row, col] = digit
                reward = 10.0  # Base reward for a correct number
                self.episode_stats["correct_moves"] += 1
                # print(f"  -> Correct guess! Placed {digit} at ({row}, {col}).")

                # Check for and reward group completions (row, col, box)
                reward += self._check_and_reward_group_completion(
                    row, col, 50.0)
                # 3. Check if the puzzle is fully solved.
                # This happens if all cells are filled AND they match the solution.
                if (np.all(self.current_grid != 0)
                        and np.array_equal(self.current_grid, self.solution_grid)):
                    reward += 100.0  # Large reward for solving
                    terminated = True
            elif self._is_valid_placement(self.current_grid, row, col, digit):
                # Correct move: place the digit and give a positive reward
                self.current_grid[row, col] = digit
                reward = 5.0
            else:
                # Incorrect move: place the digit but give a penalty.
                if not self.block_wrong_moves:
                    self.current_grid[row, col] = digit

                reward = -5.0  # Penalty for an invalid move (violates rules)

        # State is returned as float32 for PyTorch compatibility
        observation = self.current_grid.astype(np.float32)
        info = {"current_reward": reward}

        return observation, reward, terminated, truncated, info

    def _check_and_reward_group_completion(self, r, c, reward_per_group=50.0):
        """
        Checks if the row, column, or box containing the new move (r, c) is now complete
        and correct. If so, provides a reward and logs it.
        """
        reward = 0.0

        # 1. Check Row Completion
        if r not in self.rewarded_rows:
            row_slice = self.current_grid[r, :]
            # Check if the row is full (no zeros)
            if np.all(row_slice > 0):
                # Check if it matches the solution
                if np.array_equal(row_slice, self.solution_grid[r, :]):
                    # print(f"  -> Milestone! Row {r} completed correctly.")
                    reward += reward_per_group
                    self.rewarded_rows.add(r)
                    self.episode_stats["completed_rows"] += 1

        # 2. Check Column Completion
        if c not in self.rewarded_cols:
            col_slice = self.current_grid[:, c]
            if np.all(col_slice > 0):
                if np.array_equal(col_slice, self.solution_grid[:, c]):
                    # print(f"  -> Milestone! Column {c} completed correctly.")
                    reward += reward_per_group
                    self.rewarded_cols.add(c)
                    self.episode_stats["completed_cols"] += 1

        # 3. Check Box Completion
        box_r, box_c = 3 * (r // 3), 3 * (c // 3)
        box_idx = (box_r // 3, box_c // 3)
        if box_idx not in self.rewarded_boxes:
            box_slice = self.current_grid[box_r:box_r+3, box_c:box_c+3]
            if np.all(box_slice > 0):
                if np.array_equal(box_slice, self.solution_grid[box_r:box_r+3, box_c:box_c+3]):
                    # print(f"  -> Milestone! Box {box_idx} completed correctly.")
                    reward += reward_per_group
                    self.rewarded_boxes.add(box_idx)
                    self.episode_stats["completed_boxes"] += 1
        return reward

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
        # 1. Check if there are any blank cells (0) left.
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


class DifficultyHistogram:
    """Collects and logs statistics on puzzle solve rates by difficulty."""

    def __init__(self):
        self.solved_by_difficulty = {}
        self.unsolved_by_difficulty = {}

    def update(self, blank_cells: int, is_solved: bool):
        """Updates the histogram with the result of a single puzzle."""
        if blank_cells is None:
            return

        if is_solved:
            self.solved_by_difficulty[blank_cells] = self.solved_by_difficulty.get(
                blank_cells, 0) + 1
        else:
            self.unsolved_by_difficulty[blank_cells] = self.unsolved_by_difficulty.get(
                blank_cells, 0) + 1

    def log(self, title: str):
        """Prints the formatted histogram table to the console."""
        print(f"\n--- {title} ---")
        all_difficulties = sorted(set(self.solved_by_difficulty.keys()) | set(
            self.unsolved_by_difficulty.keys()))
        if not all_difficulties:
            print("No data collected.")
            return

        print(f"{'Blanks':>8} | {'Solved':>8} | {'Unsolved':>10} | {'Solve Rate':>12}")
        print(f"{'-'*8:->8} + {'-'*8:->8} + {'-'*10:->10} + {'-'*12:->12}")
        for blanks in all_difficulties:
            solved = self.solved_by_difficulty.get(blanks, 0)
            unsolved = self.unsolved_by_difficulty.get(blanks, 0)
            total = solved + unsolved
            solve_rate = f"{(solved / total * 100):.1f}%" if total > 0 else "N/A"
            print(f"{blanks:8d} | {solved:8d} | {unsolved:10d} | {solve_rate:>12s}")


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


class DQNSolver1(nn.Module):
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
        self.row_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 9))
        # 9x1 convolution finds column patterns
        self.col_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(9, 1))
        # 3x3 stride 3 convolution finds box patterns (non-overlapping tiling)
        self.box_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=3)

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


class DQNSolver2(nn.Module):
    """
    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.
    """

    def __init__(self, _input_shape, output_size, device=None):
        super().__init__()
        self.device = device

        # 1. First Pass: Extract geometric constraints
        # Input: 10 channels (one-hot digits)
        self.constraint_conv = SudokuConstraintConv(10, 64)

        # 2. Mixing Layer: 1x1 conv to combine row/col/box info per pixel
        # Input channels = 64 * 3 = 192
        self.mix_conv = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 3. Second Pass: Deepen reasoning on mixed features
        self.constraint_conv2 = SudokuConstraintConv(128, 64)

        # 4. Final Head
        # Input channels = 64 * 3 = 192
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 9 * 9, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        """Forward pass."""
        x = self.constraint_conv(x)
        x = self.mix_conv(x)
        x = self.constraint_conv2(x)
        return self.fc(x)


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
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """Forward pass."""
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DQNSolver(nn.Module):
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
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Stack of reasoning blocks (Thinking time)
        self.reasoning = nn.Sequential(
            ReasoningBlock(128),
            ReasoningBlock(128),
            ReasoningBlock(128)
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
            nn.Linear(1024, output_size)
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
    Returns the chosen action (or None if no actions are legal) and the new epsilon value.
    """
    # Use max to ensure we don't go below EPS_END
    current_epsilon = max(eps_end, epsilon)

    # Epsilon decay
    new_epsilon = max(eps_end, epsilon * eps_decay)

    mask = None
    if use_masking:
        # Here state is always numpy
        mask = generate_legal_mask(state)

    if random.random() < current_epsilon:
        # Explore: Choose a random action
        if use_masking:
            # Sample only from legal actions
            legal_actions = np.nonzero(mask)[0]
            if len(legal_actions) == 0:
                # No legal moves are possible (board is full). Signal to terminate.
                return None, new_epsilon
            action = random.choice(legal_actions)
        else:
            action = action_space.sample()
    else:
        # Exploit: Choose the action with the highest predicted Q-value
        with torch.no_grad():
            # Add batch dimension and get Q-values
            one_hot_state = state_to_one_hot(
                state, policy_net.device).unsqueeze(0)
            q_values = policy_net(one_hot_state)

            if use_masking:
                # Apply mask: set Q-values of illegal actions to a very small number
                # Create a mask with 0 for legal actions and a large negative value for illegal ones
                mask_t = torch.from_numpy(mask).to(
                    policy_net.device).unsqueeze(0)
                masked_q = q_values.clone()
                masked_q[~mask_t] = ILLEGAL_ACTION_VALUE
                action = masked_q.argmax().item()
            else:
                # Use argmax to get the best action index
                action = q_values.argmax(dim=1).item()

    return action, new_epsilon


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    transitions: List[Transition],
    gamma,
    use_masking: bool = False,
):
    """
    Performs one step of optimization on the Policy Network.

    BELLMAN LOSS AND BACKPROP

    :param transitions: List of transitions to train on.
    """

    if not transitions:
        return

    # Determine the actual batch size from the transitions list
    batch_size = len(transitions)

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Convert batch of transitions to tensors
    non_final_mask = torch.tensor(
        tuple(not d for d in batch.done), dtype=torch.bool)

    device = policy_net.device
    # Convert all states in the batch to one-hot encoding
    state_batch = torch.stack(
        [state_to_one_hot(s.numpy(), device) for s in batch.state]
    )
    action_batch = torch.tensor(
        batch.action, device=device).unsqueeze(1)  # [B, 1]
    reward_batch = torch.tensor(
        batch.reward, dtype=torch.float32, device=device)

    # Compute Q(s_t, a) - the Q-values for the actions taken
    q_values = policy_net(state_batch)
    state_action_values = q_values.gather(1, action_batch)

    # Compute V(s_{t+1}) = max_a Q(s_{t+1}, a) for non-terminal next states
    next_state_values = torch.zeros(batch_size, device=device)

    # Only proceed if there are non-final states
    if non_final_mask.any():
        non_final_next_states_np = [s for s, done in zip(
            batch.next_state, batch.done) if not done]
        non_final_next_states_t = torch.stack(
            [state_to_one_hot(s.numpy(), device) for s in non_final_next_states_np])

        with torch.no_grad():
            target_q_values = target_net(non_final_next_states_t)

            if use_masking:
                # Generate and apply mask to target Q-values
                # Using CPU version here for simplicity as it involves a list comprehension
                masks_np = np.stack([generate_legal_mask(s.cpu().numpy())
                                    for s in non_final_next_states_np])
                masks_t = torch.from_numpy(masks_np).to(device)

                target_q_values[~masks_t] = ILLEGAL_ACTION_VALUE
                # Take the max over the masked Q-values
                next_state_values[non_final_mask] = target_q_values.max(1)[
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
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Q-Learning Sudoku Solver")
    # Training arguments
    parser.add_argument('--episodes', type=int,
                        default=MAX_EPISODES, help='Number of episodes to train.')
    parser.add_argument('--puzzle', type=str, default=None,
                        help='Initial Sudoku puzzle string (81 chars, 0 for blank).')
    parser.add_argument('--reward_shaping', action='store_true',
                        help='Enable reward shaping (progress-based rewards).')
    parser.add_argument('--masking', action='store_true',
                        help='Enable action masking (only choose blank cells).')
    parser.add_argument('--fixed_puzzle', action='store_true',
                        help='Use only given puzzle for training.')
    parser.add_argument('--block_wrong_moves', action='store_true',
                        help='Prevent agent from making wrong moves.')

    # Hyperparameter arguments
    parser.add_argument('--lr', type=float, default=LR,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help='Discount factor for future rewards.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training.')
    parser.add_argument('--memory_capacity', type=int,
                        default=MEMORY_CAPACITY, help='Capacity of the replay buffer.')
    parser.add_argument('--target_update', type=int, default=TARGET_UPDATE,
                        help='Frequency (in episodes) to update the target network.')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay for the AdamW optimizer.')
    # Epsilon-greedy arguments
    parser.add_argument('--eps_start', type=float, default=EPS_START,
                        help='Starting value of epsilon for exploration.')
    parser.add_argument('--eps_end', type=float, default=EPS_END,
                        help='Minimum value of epsilon.')
    parser.add_argument('--eps_decay', type=float, default=EPS_DECAY,
                        help='Decay rate for epsilon.')

    parser.add_argument('--log_episodes', type=int, default=10,
                        help='Log info once every N episodes.')

    # Testing arguments
    parser.add_argument('--test_games', type=int, default=10,
                        help='Number of games to test after training.')
    parser.add_argument('--test_difficulty_min', type=int, default=6,
                        help='Min blank cells for test puzzles.')
    parser.add_argument('--test_difficulty_max', type=int, default=61,
                        help='Max blank cells for test puzzles.')
    parser.add_argument('--show_boards', action='store_true',
                        help='Show test puzzles and solutions.')

    # Model persistence
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save the trained model.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model.')

    args = parser.parse_args()

    return args, parser


def prevent_sleep():
    """
    Prevents the system from entering sleep or turning off the display.
    Returns a function to restore the original state.
    """
    # TODO: (when needed) This belongs in a shared lib "os_utils.py"

    p = platform.system()
    if p == "Windows":
        # pylint:disable=invalid-name
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        # pylint:enable=invalid-name

        try:
            # Prevent sleep
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
            print("Windows sleep prevention activated.")
            # Return a function to restore previous state

            def restore_sleep():
                try:
                    ctypes.windll.kernel32.SetThreadExecutionState(
                        ES_CONTINUOUS)
                finally:
                    print("Windows sleep prevention deactivated.")
            return restore_sleep
        except AttributeError:
            print("Could not prevent sleep: Failed to call Windows API.")

    # TODO: (when needed) Implement for other OS's
    else:
        print(
            f"WARNING: Sleep prevention not implemented for this platform {p}.")

    return lambda: None  # Return a no-op function for other systems


CURRICULUM_LEVELS = [
    {"name": "Super Easy", "clues": (
        78, 80), "solve_rate_threshold": 0.9, "eval_window": 50},
    {"name": "Easy", "clues": (
        50, 78), "solve_rate_threshold": 0.7, "eval_window": 100},
    {"name": "Medium", "clues": (
        40, 55), "solve_rate_threshold": 0.5, "eval_window": 200},
    {"name": "Hard", "clues": (
        25, 45), "solve_rate_threshold": None, "eval_window": None},  # Final level
]


def get_curriculum_puzzle_clues(curriculum_level: int) -> int:
    """Select puzzle difficulty based on the current curriculum level."""
    level_info = CURRICULUM_LEVELS[curriculum_level]
    min_clues, max_clues = level_info["clues"]
    num_clues = random.randint(min_clues, max_clues)
    return num_clues


def check_curriculum_progress(
    current_level: int,
    recent_solves: deque
) -> int:
    """Check if the agent is ready to advance to the next curriculum level."""
    if current_level >= len(CURRICULUM_LEVELS) - 1:
        return current_level  # Already at the highest level

    level_info = CURRICULUM_LEVELS[current_level]
    eval_window = level_info["eval_window"]
    threshold = level_info["solve_rate_threshold"]

    if len(recent_solves) >= eval_window:
        solve_rate = sum(recent_solves) / len(recent_solves)
        if solve_rate >= threshold:
            print(
                f"\n*** Curriculum Level Up! "
                f"Passed '{level_info['name']}' "
                f"with solve rate {solve_rate:.2f} >= {threshold:.2f} ***"
            )
            return current_level + 1
    return current_level


def train(args, env, policy_net, target_net, optimizer, memory) -> int:
    """Main training loop."""

    # Add device to args
    if "device" not in args or not args.device:
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Training loop
    epoch_steps_done = 0
    best_reward = -float('inf')
    solved_count = 0
    # Track minimum clues for a solved puzzle (lower is harder)
    min_clues_solved = 99

    # Metadata for training state
    total_episodes_trained = args.start_episode - 1
    current_epsilon = args.current_epsilon
    curriculum_level = args.curriculum_level
    recent_solves = deque(
        maxlen=CURRICULUM_LEVELS[curriculum_level].get("eval_window"))

    histogram = DifficultyHistogram()

    print(
        f"Starting Training "
        f"at episode {args.start_episode} "
        f"for {args.episodes} episodes, "
        f"Curriculum Level: {CURRICULUM_LEVELS[curriculum_level]['name']}"
    )
    start_time = time.time()

    for i_episode in range(args.start_episode, args.start_episode + args.episodes):
        final_episode = i_episode == args.start_episode + args.episodes - 1

        # 1. Adaptive Curriculum Learning
        curriculum_level = check_curriculum_progress(
            curriculum_level, recent_solves)
        if curriculum_level < len(CURRICULUM_LEVELS) - 1 and len(recent_solves) == recent_solves.maxlen:
            # Reset window for new level
            recent_solves = deque(
                maxlen=CURRICULUM_LEVELS[curriculum_level].get("eval_window"))

        # 2. Reset the environment and get initial state
        num_clues = get_curriculum_puzzle_clues(curriculum_level)
        reset_options = {"num_clues": num_clues}
        state, _info = env.reset(options=reset_options)

        episode_steps = 0
        episode_reward = 0
        episode_solved = False
        episode_transitions = []  # Store transitions for this episode

        # 3. Run the episode
        for _step in range(81):  # Max 81 steps (cells) per episode
            action, current_epsilon = get_action(
                state,
                policy_net,
                env.action_space,
                current_epsilon,
                args.eps_end,
                args.eps_decay,
                args.masking
            )

            if action is None:
                # No legal moves were available, terminate the episode
                # print("No legal moves left. Ending episode.")
                break

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
            transition = Transition(
                state_tensor, action, reward, next_state_tensor, done, i_episode)
            memory.push(*transition)
            # Also store for potential end-of-episode training
            episode_transitions.append(transition)

            # 6. Move to the next state
            state = next_state
            episode_reward += reward

            # 7. Perform optimization step
            transitions = memory.sample(args.batch_size)
            optimize_model(
                policy_net, target_net, optimizer,
                transitions,
                args.gamma, args.masking
            )
            epoch_steps_done += 1
            episode_steps += 1

            if terminated:
                episode_solved = True
                solved_count += 1
                if min_clues_solved > num_clues:
                    min_clues_solved = num_clues
                break
            if truncated:
                # Typically not used in Sudoku, but good practice
                break

        recent_solves.append(1 if episode_solved else 0)
        total_episodes_trained += 1

        # Update difficulty histograms
        blank_cells = env.episode_stats.get(
            'blank_cells_start', 81 - num_clues)
        histogram.update(blank_cells, episode_solved)

        # 8. Hindsight Experience Replay (HER): After an episode finishes (win or lose),
        # perform an extra training pass on its full trajectory. This provides immediate
        # feedback on the outcome.
        if episode_transitions:
            # For successful episodes, this reinforces the good moves.
            # For failed episodes, this reinforces the penalties for bad moves.
            optimize_model(
                policy_net, target_net, optimizer,
                episode_transitions,
                args.gamma, args.masking
            )

        # 9. Update the target network periodically
        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 10. Logging and Reporting
        best_reward_str = ""
        if best_reward == -float('inf') or episode_reward > best_reward:
            # if best_reward != -float('inf'):
            best_reward_str = " (New Best Reward)"
            best_reward = episode_reward

        if episode_solved or best_reward_str or i_episode % args.log_episodes == 0:
            # if not env.fixed_puzzle:
            #     print(f"Episode {i_episode}: New puzzle:")
            #     print(SudokuEnv.format_grid_to_string(env.initial_puzzle))

            stats = env.episode_stats
            solved_ratio = f"{stats['correct_moves']:2d}/{stats['blank_cells_start']:2d}"
            groups_completed = f"R:{stats['completed_rows']}" \
                f"/C:{stats['completed_cols']}" \
                f"/B:{stats['completed_boxes']}"

            print(
                f"Episode {i_episode:6d}: "
                f"Level: {CURRICULUM_LEVELS[curriculum_level]['name']}, "
                f"Steps: {episode_steps:3d}, "
                f"Epoch Steps: {epoch_steps_done:6d}, "
                f"Epsilon: {max(args.eps_end, current_epsilon):.4f}, "
                f"Cells: {solved_ratio}, Groups: {groups_completed}, "
                f"({'    Solved' if episode_solved else 'NOT Solved'}), "
                f"Total Reward: {episode_reward: 8.2f}{best_reward_str}, "
            )

        # 11. Save the model periodically or at the end of training
        if args.save_model and (best_reward_str or final_episode or i_episode % 100 == 0):
            model_str = 'final model' if final_episode else 'model checkpoint'
            print(f"Saving {model_str} to {args.save_model}...")
            torch.save({
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_episodes_trained': total_episodes_trained,
                'curriculum_level': curriculum_level,
                'current_epsilon': current_epsilon,
            }, args.save_model)

    end_time = time.time()
    training_duration = end_time - start_time
    duration_str = str(timedelta(seconds=training_duration))
    time_per_step = training_duration / epoch_steps_done
    time_per_step_str = str(timedelta(seconds=time_per_step))

    difficulty_summary = f" (Hardest level {81-min_clues_solved} cell(s))" \
        if min_clues_solved != 99 else ""
    print("\n" + "\n  ".join([
        f"Training Complete {'='*60}",
        f"Final Best Reward: {best_reward:.2f} over {total_episodes_trained} total episodes.",
        f"Total Solved: {solved_count}{difficulty_summary}",
        f"Total time: {duration_str} ({time_per_step_str} per step)",
    ]))
    histogram.log("Training Performance by Difficulty (Blank Cells)")

    return 0


def run_test_episode(args, env, policy_net, initial_state, show_boards=True):
    """Runs a single episode in test mode (exploitation only) and returns the result."""
    state = initial_state
    episode_steps = 0
    episode_reward = 0

    for _step in range(81):  # Max 81 steps
        with torch.no_grad():
            one_hot_state = state_to_one_hot(state, args.device).unsqueeze(0)
            q_values = policy_net(one_hot_state)

            if args.masking:
                # During testing, state is a numpy array, so we use the CPU version.
                mask = generate_legal_mask(state)
                if not np.any(mask):
                    break  # No legal moves left
                mask_tensor = torch.from_numpy(mask).to(args.device)
                additive_mask = torch.where(
                    mask_tensor,
                    torch.tensor(0.0, device=args.device),
                    torch.tensor(ILLEGAL_ACTION_VALUE, device=args.device)
                ).unsqueeze(0)
                action = (q_values + additive_mask).argmax().item()
            else:
                action = q_values.argmax().item()

        observation, reward, terminated, truncated, _info = env.step(action)
        state = observation
        episode_reward += reward
        episode_steps += 1
        if terminated or truncated:
            break

    if show_boards:
        initial_board = SudokuEnv.format_grid_to_string(
            env.initial_puzzle).split('\n')
        final_board = SudokuEnv.format_grid_to_string(
            env.current_grid).split('\n')
        delta_grid = env.current_grid - env.initial_puzzle
        delta_board = SudokuEnv.format_grid_to_string(delta_grid).split('\n')

        # Print 3 boards horizontally, with some gap
        gap = "    "
        headings = ["Initial", "Final", "Moves"]
        print(gap.join([f"{h:21s}" for h in headings]))

        lines = len(initial_board)
        print("\n".join([
            initial_board[i] + gap + final_board[i] + gap + delta_board[i]
            for i in range(lines)
        ]))

    is_solved = np.array_equal(env.current_grid, env.solution_grid)
    return is_solved, episode_reward, episode_steps


def log_test_result(env, i_game, num_generated_games, steps, final_reward, is_solved):
    """Helper function to log the results of a single test game."""
    stats = env.episode_stats
    solved_ratio = f"{stats['correct_moves']:2d}/{stats['blank_cells_start']:2d}"
    groups_completed = f"R:{stats['completed_rows']}" \
        f"/C:{stats['completed_cols']}" \
        f"/B:{stats['completed_boxes']}"
    print(
        f"  Game  {i_game:6d} of {num_generated_games:6d}: "
        f"Steps: {steps:3d}, "
        # f"Epoch Steps: {epoch_steps_done:6d}, "
        # f"Epsilon: {max(args.eps_end, current_epsilon):.4f}, "
        f"Cells: {solved_ratio}, Groups: {groups_completed}, "
        f"({'    Solved' if is_solved else 'NOT Solved'}), "
        f"Reward: {final_reward: 8.2f}, "
    )


def test(args, env, policy_net) -> int:
    """Tests the trained agent on a set of puzzles."""
    print(f"\n--- Running Test Phase ({args.test_games} games) ---")
    solved_count = 0
    total_reward = 0

    histogram = DifficultyHistogram()
    num_generated_games = args.test_games if not args.puzzle else args.test_games - 1

    # 1. Test on the fixed puzzle first, if provided
    if args.puzzle:
        print("\n1. Testing on the provided fixed puzzle...")
        # Force reset to the fixed puzzle
        env.fixed_puzzle = True
        state, _ = env.reset()
        env.fixed_puzzle = args.fixed_puzzle  # Revert to original setting

        is_solved, final_reward, steps = run_test_episode(
            args, env, policy_net, state, show_boards=args.show_boards)
        if is_solved:
            solved_count += 1
        total_reward += final_reward

        log_test_result(env, 0, num_generated_games,
                        steps, final_reward, is_solved)
        histogram.update(env.episode_stats.get('blank_cells_start'), is_solved)

    # 2. Test on procedurally generated puzzles
    if num_generated_games > 0:
        print(
            f"\n2. Testing on {num_generated_games} generated puzzles "
            f"(Difficulty: {args.test_difficulty_min}-{args.test_difficulty_max})..."
        )
        for i_game in range(1, num_generated_games+1):
            num_clues = 81 - math.floor(0.5 + args.test_difficulty_min + (i_game - 1) * (
                args.test_difficulty_max - args.test_difficulty_min) / num_generated_games)
            state, _ = env.reset(options={"num_clues": num_clues})
            is_solved, final_reward, steps = run_test_episode(
                args, env, policy_net, state, show_boards=args.show_boards)
            if is_solved:
                solved_count += 1
            total_reward += final_reward

            log_test_result(env, i_game, num_generated_games,
                            steps, final_reward, is_solved)
            histogram.update(env.episode_stats.get(
                'blank_cells_start'), is_solved)

    # 3. Report final statistics
    solve_rate = (solved_count / args.test_games) * \
        100 if args.test_games > 0 else 0
    avg_reward = total_reward / args.test_games if args.test_games > 0 else 0

    print("\n" + "\n  ".join([
        f"Test Phase Complete {'='*56}",
        f"Puzzles Solved: {solved_count} / {args.test_games} ({solve_rate:.1f}%)",
        f"Average Reward: {avg_reward:.2f}",
    ]))
    histogram.log("Test Performance by Difficulty (Blank Cells)")

    return 0


def main() -> int:
    """Main function."""

    args, _parser = parse_args()

    # Initialize Environment, Networks, Optimizer, and Memory
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SudokuEnv(
        puzzle_str=args.puzzle,
        reward_shaping=args.reward_shaping,
        fixed_puzzle=args.fixed_puzzle,
        block_wrong_moves=args.block_wrong_moves,
    )
    policy_net = DQNSolver(env.observation_space.shape,
                           env.action_space.n, args.device).to(args.device)
    target_net = DQNSolver(env.observation_space.shape,
                           env.action_space.n, args.device).to(args.device)
    optimizer = optim.AdamW(policy_net.parameters(),
                            lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    memory = ReplayBuffer(args.memory_capacity)

    # Load a pre-trained model if specified
    if args.load_model:
        try:
            print(f"Loading model and metadata from {args.load_model}...")
            checkpoint = torch.load(args.load_model, map_location=args.device)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load metadata to continue training state
            total_episodes_trained = checkpoint.get(
                'total_episodes_trained', 0)
            args.start_episode = total_episodes_trained + 1
            args.curriculum_level = checkpoint.get('curriculum_level', 0)
            args.current_epsilon = checkpoint.get(
                'current_epsilon', args.eps_start)

            # TODO: (when needed) Improve how it logs. This log assumes training, but args.episodes = 0 will skip training
            print(f"Resuming training from episode {args.start_episode}.")
            print(
                f"Loaded curriculum level: {CURRICULUM_LEVELS[args.curriculum_level]['name']}.")
            print(f"Loaded epsilon: {args.current_epsilon:.4f}.")

        except FileNotFoundError:
            print(
                f"Error: Model file not found at {args.load_model}. Starting from scratch.")
            args.start_episode = 1
            args.curriculum_level = 0
            args.current_epsilon = args.eps_start
    else:
        args.start_episode = 1
        args.curriculum_level = 0
        args.current_epsilon = args.eps_start

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is always in evaluation mode

    # Prevent the computer from sleeping during training
    restore_sleep_function = prevent_sleep()

    rc = 0
    try:
        # Run training if episodes are specified
        if args.episodes > 0:
            # train is responsible for saving to args.save_model (checkpoint and final)
            rc = train(args, env, policy_net, target_net, optimizer, memory)
        if not rc:
            # Run the test phase if specified
            if args.test_games > 0:
                rc = test(args, env, policy_net)
    except Exception as e:
        # traceback.print_exc()
        print(f"\nError {e} occurred during training or testing. Exiting.")
        rc = 1
    finally:
        # This will run whether the training completes successfully or fails
        restore_sleep_function()
    return rc


if __name__ == "__main__":
    exit(main())
