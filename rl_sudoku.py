#!/usr/bin/env python3
# rl_sudoku.py
# pylint:disable=too-many-lines,broad-exception-caught

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
from datetime import timedelta
import math
import platform
import random
from typing import Any, List, Optional, Tuple
import time
import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_cnn1 import DQNSolverCNN1
from dqn_cnn2 import DQNSolverCNN2
from dqn_cnn3 import DQNSolverCNN3
from dqn_cnn4 import DQNSolverCNN4
from dqn_cnn5 import DQNSolverCNN5
from dqn_cnn6 import DQNSolverCNN6
from dqn_transformer import DQNSolver as DQNSolverTransformer

from sudoku import (
    format_grid_to_string,
    # format_grid_to_strings,
    arr_to_str,
    generate_solved_sudoku,
    get_unique_sudoku,
    print_grids,
    str_to_arr,
)

# TODO: (when needed) Adjust these during Phase 3
# Default Hyperparameters & Epsilon-greedy params (all can be changed from command line):
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05  # A slightly higher floor can encourage exploration on harder puzzles
EPS_DECAY = 0.9997  # Faster decay to encourage exploitation sooner
TARGET_UPDATE = 10  # Frequency (in episodes) to update the target network
MEMORY_CAPACITY = 100000 # Increased for better PER performance
BATCH_SIZE = 128  # Larger batch size can stabilize training
LR = 0.00025  # Slightly higher learning rate
MAX_EPISODES = 50000
PONDER_PENALTY = 0.1  # Increased penalty for each "thinking" step in ACT models
PONDER_PENALTY_START = 0.0  # Start at 0 to allow "deep thinking" learning before optimizing for speed
PONDER_PENALTY_ANNEAL_EPISODES = 5000
WEIGHT_DECAY = 0.01

# Large Negative reward to suppress known illegal actions
ILLEGAL_ACTION_VALUE = -1e10

# Dafault test settings:
TEST_GAMES_REPEAT = 5
TEST_DIFFICULTY_MIN = 3
TEST_DIFFICULTY_MAX = 61

EXPLORE_MASKING_GPU = True  # Use GPU for exploration masking

def _puzzle_worker(
    queue: mp.Queue,
    stop_event: Event,
    min_clues_shared: Synchronized,
    max_clues_shared: Synchronized,
):
    """Worker process to generate valid Sudoku puzzles. Handles Ctrl-C gracefully."""
    try:
        while not stop_event.is_set():
            if queue.qsize() > 50:  # Don't overfill the queue
                time.sleep(0.1)
                continue

            # Read the current difficulty from shared memory
            min_clues = min_clues_shared.value
            max_clues = max_clues_shared.value

            solution = generate_solved_sudoku()
            num_clues = random.randint(min_clues, max_clues)
            puzzle = get_unique_sudoku(solution, num_clues)
            if puzzle is not None:
                # Serialize numpy arrays to strings to minimize memory use in pipe
                puzzle_str = arr_to_str(puzzle)
                solution_str = arr_to_str(solution)
                queue.put((puzzle_str, solution_str, num_clues))
    except KeyboardInterrupt:
        pass  # Exit gracefully on Ctrl-C


class PuzzleGenerator:
    """Manages a pool of worker processes to generate puzzles asynchronously."""

    def __init__(
        self, num_workers: int, initial_min_clues: int = 1, initial_max_clues: int = 35
    ):
        self.num_workers = num_workers
        # Use shared memory for dynamic difficulty updates
        self.min_clues: Synchronized = mp.Value("i", initial_min_clues)
        self.max_clues: Synchronized = mp.Value("i", initial_max_clues)
        self.puzzle_queue = mp.Queue(maxsize=100)
        self.stop_event: Event = mp.Event()
        self.workers: List[mp.Process] = []

    def start(self):
        """Starts the worker processes."""
        print(f"Starting {self.num_workers} puzzle generator workers...")
        for _ in range(self.num_workers):
            p = mp.Process(
                target=_puzzle_worker,
                args=(
                    self.puzzle_queue,
                    self.stop_event,
                    self.min_clues,
                    self.max_clues,
                ),
            )
            p.start()
            self.workers.append(p)

    def stop(self):
        """Stops all worker processes."""
        print("Stopping puzzle generator workers...")
        self.stop_event.set()
        # Clear the queue to unblock workers if they are waiting
        while not self.puzzle_queue.empty():
            try:
                self.puzzle_queue.get_nowait()
            except Exception:
                break
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()  # Force terminate if join fails
        print("Puzzle workers stopped.")

    def get_puzzle(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Retrieves a pre-generated puzzle from the queue."""
        # This will block until a puzzle is available
        puzzle_str, solution_str, num_clues = self.puzzle_queue.get()
        # Deserialize strings back to numpy arrays
        return str_to_arr(puzzle_str), str_to_arr(solution_str), num_clues

    def set_difficulty(self, min_clues: int, max_clues: int):
        """Atomically updates the difficulty range for the workers."""
        with self.min_clues.get_lock(), self.max_clues.get_lock():
            self.min_clues.value = min_clues
            self.max_clues.value = max_clues


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
    if num in grid[box_r : box_r + 3, box_c : box_c + 3]:
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
    Vectorized, GPU-accelerated version of generate_legal_mask.
    Accepts a batch of grids (B, 9, 9) and returns a batch of masks (B, 729).
    """
    is_batch = grid_tensor.dim() == 3
    if not is_batch:
        # Add batch dimension if not present
        grid_tensor = grid_tensor.unsqueeze(0)

    b_size = grid_tensor.shape[0]
    _device = grid_tensor.device

    # One-hot encode the grid batch: (B, 9, 9, 10) -> (B, 9, 9, 9) for digits 1-9
    one_hot = nn.functional.one_hot(grid_tensor.long(), num_classes=10)[
        :, :, :, 1:
    ].bool()

    # Row and column used digits: (B, 9, 9, 9)
    row_used = one_hot.any(dim=2, keepdim=True).expand(-1, -1, 9, -1)
    col_used = one_hot.any(dim=1, keepdim=True).expand(-1, 9, -1, -1)

    # Box used digits: Reshape and check
    box_used = one_hot.view(b_size, 3, 3, 3, 3, 9).any(dim=(2, 4), keepdim=True)
    box_used = box_used.expand(b_size, 3, 3, 3, 3, 9).reshape(b_size, 9, 9, 9)

    used_mask = row_used | col_used | box_used

    empty_mask = (grid_tensor == 0).unsqueeze(3)  # (B, 9, 9, 1)
    legal_mask_4d = empty_mask & ~used_mask

    final_mask = legal_mask_4d.view(b_size, -1)  # Flatten to (B, 729)
    return final_mask if is_batch else final_mask.squeeze(0)


def state_to_one_hot(grid_tensor: torch.Tensor) -> torch.Tensor:
    """
    Vectorized. Converts a batch of grid tensors (B, 9, 9) or a single grid tensor (9,9)
    to a one-hot encoded tensor (B, 10, 9, 9) on the given device.
    Assumes input tensor is already on the correct device.
    """
    is_batch = grid_tensor.dim() == 3
    if not is_batch:
        grid_tensor = grid_tensor.unsqueeze(0)  # Add batch dimension

    one_hot = (
        nn.functional.one_hot(grid_tensor.long(), num_classes=10)
        .permute(0, 3, 1, 2)
        .float()
    )

    return one_hot if is_batch else one_hot.squeeze(0)


# State/action transition data structure for the Replay Buffer
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "episode")
)


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
        puzzle_generator: Optional[PuzzleGenerator] = None,
    ):
        super().__init__()

        # 9 rows, 9 columns, digits 1-9.
        # Action space is discrete, but we'll map 0-728 to (row, col, digit)
        self.action_space = gym.spaces.Discrete(9 * 9 * 9)
        # Observation space is 9x9 grid, values 0 (blank) to 9
        self.observation_space = gym.spaces.Box(
            low=0, high=9, shape=(9, 9), dtype=np.int32
        )

        # RL options
        self.reward_shaping = reward_shaping
        self.fixed_puzzle = fixed_puzzle
        self.puzzle_generator = puzzle_generator

        # Initial puzzle (0 for blank cells)
        self.default_puzzle, self.default_solution = self._parse_puzzle(
            puzzle_str, sol_str
        )
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
        self, puzzle_str: Optional[str], sol_str: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Converts an 81-char string (0 for blank) into a 9x9 numpy array."""
        if puzzle_str is None and sol_str is None:
            # Default easy puzzle for starting development
            puzzle_str, sol_str = (
                "000260701680070090190004500820100040004602900050003028009300074040050036703018000",
                "435269781682571493197834562826195347374682915951743628519326874248957136763418259",
            )

        # Ensure the string is 81 characters long and contains only digits
        if len(puzzle_str) != 81 or not puzzle_str.isdigit() or "0" not in puzzle_str:
            raise ValueError(
                f"Puzzle string must be 81 digits (0-9), must contain zeroes to be a puzzle, "
                f"{len(puzzle_str)} provided."
            )
        if len(sol_str) != 81 or not sol_str.isdigit() or "0" in sol_str:
            raise ValueError(
                f"Solution string must be 81 digits (1-9), no 0's, {len(sol_str)} provided."
            )

        grid = np.array([int(c) for c in puzzle_str]).reshape((9, 9))
        sol_grid = np.array([int(c) for c in sol_str]).reshape((9, 9))
        return grid, sol_grid

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment to a new (or default) puzzle state."""
        super().reset(seed=seed, options=options)

        num_clues = 0  # Initialize to a default value
        if self.fixed_puzzle:
            # Use the default puzzle
            self.initial_puzzle = self.default_puzzle.copy()
            self.solution_grid = self.default_solution.copy()
            num_clues = np.count_nonzero(self.initial_puzzle != 0)
            # The default_puzzle and default_solution should have been set during __init__
        elif options and "num_clues" in options and self.puzzle_generator:
            # Synchronously generate a puzzle with a specific number of clues for testing.
            # This bypasses the async queue to ensure the test sweep is accurate.
            num_clues = options["num_clues"]
            solution = generate_solved_sudoku()
            puzzle = get_unique_sudoku(solution, num_clues)
            self.initial_puzzle, self.solution_grid = puzzle, solution
        elif self.puzzle_generator:
            # Get a pre-validated puzzle from the async generator
            self.initial_puzzle, self.solution_grid, num_clues = (
                self.puzzle_generator.get_puzzle()
            )
        else:
            # Generate a new, random, solvable puzzle
            self.solution_grid = generate_solved_sudoku()

            num_clues = random.randint(25, 55)  # Default value
            num_clues = options.get("num_clues", num_clues) if options else num_clues
            self.initial_puzzle = get_unique_sudoku(self.solution_grid, num_clues)

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

        observation = self.current_grid  # Return as int32
        return observation, num_clues, self.episode_stats

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
                reward += self._check_and_reward_group_completion(row, col, 50.0)
                # 3. Check if the puzzle is fully solved.
                # This happens if all cells are filled AND they match the solution.
                if np.all(self.current_grid != 0) and np.array_equal(
                    self.current_grid, self.solution_grid
                ):
                    reward += 100.0  # Large reward for solving
                    terminated = True
            else:
                # Incorrect move. This path is a dead end as it deviates from the unique solution.
                # We penalize the agent and terminate the episode immediately to avoid
                # training on a "poisoned" board state that has no solution.
                # Penalty for an invalid move (violates rules or solution path)
                reward = -5.0
                truncated = True

        # State is returned as float32 for PyTorch compatibility
        observation = self.current_grid  # Return as int32
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
            box_slice = self.current_grid[box_r : box_r + 3, box_c : box_c + 3]
            if np.all(box_slice > 0) and np.array_equal(
                box_slice, self.solution_grid[box_r : box_r + 3, box_c : box_c + 3]
            ):
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
                box = grid[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3].flatten()
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
        box = grid[start_r : start_r + 3, start_c : start_c + 3]
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
                box = grid[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3].flatten()
                if not self._check_group(box):
                    return False

        return True

    def render(self):
        """Prints the current state of the Sudoku grid."""
        print(format_grid_to_string(self.current_grid))

    def close(self):
        """Clean up resources."""


class SumTree:
    """
    A binary tree data structure where the value of a parent node is the sum of its
    children. It is used for efficient sampling from a probability distribution.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree is 1-indexed, size is 2*capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # Data is stored in the second half of the tree array
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0  # Next index to write to
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Update the tree upwards from a leaf node."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node."""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]

    def add(self, p: float, data: Any):
        """Store priority and sample."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float):
        """Update priority of a sample."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get a sample from the tree."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """A Replay Buffer that samples transitions based on their TD-error (priority)."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.epsilon = 0.01  # Small constant to ensure non-zero priority
        self.max_priority = 1.0

    def push(self, *args):
        """Adds a new transition with maximum priority to ensure it gets sampled."""
        # New transitions are given max priority to guarantee they are trained on at least once
        priority = self.max_priority
        self.tree.add(priority, Transition(*args))

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample a batch, returning transitions, indices, and IS weights."""
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(p)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # Normalize for stability
        return batch, np.array(indices), is_weights

    def __len__(self):
        return self.tree.n_entries

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        for idx, p in zip(indices, priorities):
            priority = (p + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


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
            self.solved_by_difficulty[blank_cells] = (
                self.solved_by_difficulty.get(blank_cells, 0) + 1
            )
        else:
            self.unsolved_by_difficulty[blank_cells] = (
                self.unsolved_by_difficulty.get(blank_cells, 0) + 1
            )

    def log(self, title: str):
        """Prints the formatted histogram table to the console."""
        print(f"\n--- {title} ---")
        all_difficulties = sorted(
            set(self.solved_by_difficulty.keys())
            | set(self.unsolved_by_difficulty.keys())
        )
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

    def get_capability_score(self, use_best_100: bool = False) -> float:
        """
        Calculates a single numeric score representing the solver's capability.
        - The integer part is the max difficulty (blanks) solved with 100% accuracy. If
          `use_best_100` is False, this is the highest level in an unbroken chain of
          100% solved difficulties starting from the easiest.
        - The fractional part is a weighted average of solve rates for harder puzzles.
        """
        all_difficulties = sorted(
            set(self.solved_by_difficulty.keys())
            | set(self.unsolved_by_difficulty.keys())
        )
        if not all_difficulties:
            return 0.0

        # 1. Find the integer part (max_100_diff)
        max_100_diff = 0
        if not use_best_100:
            # Strict mode: Find highest difficulty in an unbroken 100% streak from the start
            for blanks in all_difficulties:
                if self.unsolved_by_difficulty.get(blanks, 0) == 0:
                    max_100_diff = blanks  # This level is 100%, continue
                else:
                    break  # Streak is broken, stop here
        else:
            # Default mode: Find the absolute max difficulty with 100% solve rate
            for blanks in all_difficulties:
                if self.unsolved_by_difficulty.get(blanks, 0) == 0:
                    max_100_diff = max(max_100_diff, blanks)

        # 2. Calculate the fractional part for difficulties > max_100_diff
        weighted_solve_rate_sum = 0.0
        total_weight = 0.0
        higher_difficulties = [d for d in all_difficulties if d > max_100_diff]

        if higher_difficulties:
            for blanks in higher_difficulties:
                solved = self.solved_by_difficulty.get(blanks, 0)
                unsolved = self.unsolved_by_difficulty.get(blanks, 0)
                total = solved + unsolved
                if total > 0:
                    solve_rate = solved / total
                    weight = blanks  # Weight by difficulty
                    weighted_solve_rate_sum += solve_rate * weight
                    total_weight += weight

        fractional_part = (
            (weighted_solve_rate_sum / total_weight) if total_weight > 0 else 0.0
        )

        return float(max_100_diff) + fractional_part


def get_action(
    state,
    policy_net,
    action_space,
    epsilon,
    eps_end,
    eps_decay,
    use_masking: bool = False,
):
    """
    Implements the epsilon-greedy policy.
    Returns the chosen action (or None if no actions are legal) and the new epsilon value.
    """
    # Use max to ensure we don't go below EPS_END
    current_epsilon = max(eps_end, epsilon)

    # Epsilon decay
    new_epsilon = max(eps_end, epsilon * eps_decay)

    if random.random() < current_epsilon:
        # Explore: Choose a random action
        if use_masking:
            # Sample only from legal actions using the faster GPU version
            if EXPLORE_MASKING_GPU:  # Only use GPU masking if available:
                state_t = torch.from_numpy(state).to(policy_net.device)
                mask_t = generate_legal_mask_gpu(state_t)
                # .squeeze() can create a 0-dim tensor if there's only one legal move.
                # np.atleast_1d() ensures it's always an array.
                legal_actions = np.atleast_1d(
                    torch.nonzero(mask_t, as_tuple=False).squeeze().cpu().numpy()
                )
            else:
                mask = generate_legal_mask(state)
                legal_actions = np.atleast_1d(np.nonzero(mask)[0])
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
            state_t = torch.from_numpy(state).to(policy_net.device)
            one_hot_state = state_to_one_hot(state_t)

            # Handle models that return multiple values (like ACT ponder cost)
            output = policy_net(one_hot_state.unsqueeze(0))
            if isinstance(output, tuple):
                q_values = output[0]
            else:
                q_values = output

            if use_masking:
                # Apply mask: set Q-values of illegal actions to a very small number
                # Generate mask on GPU
                mask_t = generate_legal_mask_gpu(state_t)

                masked_q = q_values.clone()
                # Unsqueeze to add batch dim
                masked_q[~mask_t.unsqueeze(0)] = ILLEGAL_ACTION_VALUE
                action = masked_q.argmax(dim=1).item()
            else:
                # Use argmax to get the best action index
                action = q_values.argmax(dim=1).item()

    return action, new_epsilon


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    memory,
    batch_data,
    gamma,
    ponder_penalty,
    use_masking: bool = False,
):
    """
    Performs one step of optimization on the Policy Network.
    Handles both standard and prioritized replay.
    Returns (min, mean, max)) ponder steps.
    """
    is_per = isinstance(memory, PrioritizedReplayBuffer)
    if is_per:
        transitions, indices, is_weights = batch_data
        is_weights_t = torch.tensor(
            is_weights, dtype=torch.float32, device=policy_net.device
        )
    else:
        transitions = batch_data

    if not transitions:
        return (0.0, 0.0, 0.0)

    # Determine the actual batch size from the transitions list
    batch_size = len(transitions)

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # Convert batch of transitions to tensors
    non_final_mask = torch.tensor(tuple(not d for d in batch.done), dtype=torch.bool)

    device = policy_net.device
    # Vectorized conversion of all states in the batch to one-hot encoding
    # Directly stack tensors from the buffer
    state_batch_tensors = torch.stack(batch.state).to(device)
    state_batch = state_to_one_hot(state_batch_tensors)
    action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)  # [B, 1]
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

    # Compute Q(s_t, a) - the Q-values for the actions taken
    # Handle models that return multiple values (like ACT ponder cost)
    output = policy_net(state_batch)
    if isinstance(output, tuple):
        q_values, ponder_cost = output
    else:
        q_values = output
        ponder_cost = None

    state_action_values = q_values.gather(1, action_batch)

    # Compute V(s_{t+1}) = max_a Q(s_{t+1}, a) for non-terminal next states
    next_state_values = torch.zeros(batch_size, device=device)

    # Only proceed if there are non-final states
    if non_final_mask.any():
        # Stack all next_states and filter on the GPU using the non_final_mask.
        # This avoids a slow Python loop and keeps the pipeline on the GPU.
        all_next_states = torch.stack(batch.next_state).to(device)
        non_final_next_states_gpu = all_next_states[non_final_mask]
        non_final_next_states_t = state_to_one_hot(non_final_next_states_gpu)

        with torch.no_grad():
            # Handle models that return multiple values
            target_output = target_net(non_final_next_states_t)
            if isinstance(target_output, tuple):
                target_q_values = target_output[0]
            else:
                target_q_values = target_output

            if use_masking:
                # Generate masks for the entire batch on the GPU
                masks_t = generate_legal_mask_gpu(non_final_next_states_gpu)

                # Avoid in-place modification of target_q_values.
                # Create an additive mask instead of modifying the tensor directly.
                # This prevents corruption of the computation graph.
                additive_mask = torch.where(masks_t, 0.0, ILLEGAL_ACTION_VALUE)
                masked_target_q = target_q_values + additive_mask

                # Take the max over the masked Q-values
                next_state_values[non_final_mask] = masked_target_q.max(1)[0].detach()
            else:
                # Use standard max Q-value
                next_state_values[non_final_mask] = target_q_values.max(1)[0].detach()

    # Compute the expected Q values (target)
    expected_state_action_values = (next_state_values.to(device) * gamma) + reward_batch

    # Compute Huber loss (a robust form of MSE) - less sensitive to outliers
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # For PER, we need per-element losses to update priorities.
    # We only update priorities for batches that were actually sampled from the buffer
    # (i.e., when indices are available). The HER pass will have indices=None.
    if is_per:
        with torch.no_grad():
            td_error = torch.abs(
                state_action_values - expected_state_action_values.unsqueeze(1)
            )
            if indices is not None:
                memory.update_priorities(indices, td_error.cpu().numpy().flatten())

        # Apply importance sampling weights
        loss = (loss * is_weights_t).mean()

    # --- Optimization ---
    optimizer.zero_grad()

    if ponder_cost is not None:
        # Combine losses for ACT models (Single-Pass)
        # This allows the reasoning block to learn to produce features that trigger
        # the halting gate at the appropriate time (Reason-then-Halt).
        ponder_loss = ponder_penalty * ponder_cost.mean()
        total_loss = loss + ponder_loss
        total_loss.backward()
    else:
        # Standard single-pass optimization for non-ACT models
        loss.backward()

    # Gradient clipping (optional but recommended for stability)
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return (
            ponder_cost.min().item(),
            ponder_cost.mean().item(),
            ponder_cost.max().item(),
         ) if ponder_cost is not None else (0.0, 0.0, 0.0)


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
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            print("Windows sleep prevention activated.")
            # Return a function to restore previous state

            def restore_sleep():
                try:
                    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                finally:
                    print("Windows sleep prevention deactivated.")

            return restore_sleep
        except AttributeError:
            print("Could not prevent sleep: Failed to call Windows API.")

    # TODO: (when needed) Implement for other OS's
    else:
        print(f"WARNING: Sleep prevention not implemented for this platform {p}.")

    return lambda: None  # Return a no-op function for other systems


# Websudoku levels: Easy=45, Medium=52, Hard=54, Evil=55.
# Theoretical limit 65 (min 17 clues) - for 66 (16 clues) unique solutions do not exist.
# Past 45 random clues tend to lead to multiple solutions often
CURRICULUM_LEVELS = [
    {
        "name": "Super Easy",
        "clues": (78, 80),
        "solve_rate_threshold": 0.9,
        "eval_window": 50,
    },
    {
        "name": "Easy",
        "clues": (50, 78),
        "solve_rate_threshold": 0.7,
        "eval_window": 100,
    },
    {
        "name": "Medium",
        "clues": (40, 55),
        "solve_rate_threshold": 0.5,
        "eval_window": 200,
    },
    {
        "name": "Hard",
        "clues": (25, 45),
        "solve_rate_threshold": None,
        "eval_window": None,
    },
    {
        "name": "Expert",
        "clues": (19, 27),
        "solve_rate_threshold": None,
        "eval_window": None,
    },  # Final level
]


def get_curriculum_puzzle_clues(curriculum_level: int) -> int:
    """Select puzzle difficulty based on the current curriculum level."""
    level_info = CURRICULUM_LEVELS[curriculum_level]
    min_clues, max_clues = level_info["clues"]
    num_clues = random.randint(min_clues, max_clues)
    return num_clues


def check_curriculum_progress(current_level: int, recent_solves: deque) -> int:
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
                f"Passed level {current_level} '{level_info['name']}' "
                f"with solve rate {solve_rate:.2f} >= {threshold:.2f} ***"
            )
            return current_level + 1
    return current_level


def train(args, env, policy_net, target_net, optimizer, memory) -> int:
    """Main training loop."""

    # Add device to args
    if "device" not in args or not args.device:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Training loop
    epoch_steps_done = 0
    best_reward = -float("inf")
    solved_count = 0
    # Track minimum clues for a solved puzzle (lower is harder)
    min_clues_solved = 99

    # Metadata for training state
    total_episodes_trained = args.start_episode - 1
    current_epsilon = args.current_epsilon
    curriculum_level = args.curriculum_level
    recent_solves = deque(maxlen=CURRICULUM_LEVELS[curriculum_level].get("eval_window"))

    histogram = DifficultyHistogram()

    print(
        f"Starting Training "
        f"at episode {args.start_episode} "
        f"for {args.episodes} episodes, "
        f"Curriculum Level: {curriculum_level} '{CURRICULUM_LEVELS[curriculum_level]['name']}'"
    )
    start_time = time.time()

    try:
        for i_episode in range(args.start_episode, args.start_episode + args.episodes):
            final_episode = i_episode == args.start_episode + args.episodes - 1

            # 1. Adaptive Curriculum Learning
            curriculum_level = check_curriculum_progress(
                curriculum_level, recent_solves
            )
            if (
                curriculum_level < len(CURRICULUM_LEVELS) - 1
                and len(recent_solves) == recent_solves.maxlen
            ):
                # Reset window for new level
                recent_solves = deque(
                    maxlen=CURRICULUM_LEVELS[curriculum_level].get("eval_window")
                )

            # Update puzzle generator difficulty based on curriculum
            if env.puzzle_generator:
                min_c, max_c = CURRICULUM_LEVELS[curriculum_level]["clues"]
                env.puzzle_generator.set_difficulty(min_c, max_c)

            # Ponder Cost Annealing: Calculate current penalty for this episode
            progress = min(
                1.0,
                (
                    total_episodes_trained / args.ponder_penalty_anneal_episodes
                    if args.ponder_penalty_anneal_episodes > 0
                    else 1.0
                ),
            )
            current_ponder_penalty = args.ponder_penalty_start + progress * (
                args.ponder_penalty - args.ponder_penalty_start
            )

            # 2. Reset the environment and get initial state
            state, num_clues, _info = env.reset()

            episode_steps = 0
            episode_reward = 0
            episode_solved = False
            episode_transitions = []  # Store transitions for this episode
            ponder_steps = (0.0, 0.0, 0.0)

            # 3. Run the episode
            for _step in range(81):  # Max 81 steps (cells) per episode
                action, current_epsilon = get_action(
                    state,
                    policy_net,
                    env.action_space,
                    current_epsilon,
                    args.eps_end,
                    args.eps_decay,
                    args.masking,
                )

                if action is None:
                    # No legal moves were available, terminate the episode
                    # print("No legal moves left. Ending episode.")
                    break

                # 4. Take action in environment
                observation, reward, terminated, truncated, _info = env.step(action)
                next_state = observation
                done = terminated or truncated

                # 5. Store the transition in the replay memory
                # Store state as integer tensors on CPU
                state_t = torch.from_numpy(state).int()
                next_state_t = torch.from_numpy(next_state).int()
                transition = Transition(
                    state_t, action, reward, next_state_t, done, i_episode
                )
                memory.push(*transition)
                # Also store for potential end-of-episode training
                episode_transitions.append(transition)

                # 6. Move to the next state
                state = next_state
                episode_reward += reward

                # 7. Perform optimization step
                if len(memory) >= args.batch_size:
                    batch_data = memory.sample(args.batch_size)
                    ponder_steps = optimize_model(
                        policy_net,
                        target_net,
                        optimizer,
                        memory,
                        batch_data,
                        args.gamma,
                        current_ponder_penalty,
                        args.masking,
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
            blank_cells = env.episode_stats.get("blank_cells_start", 81 - num_clues)
            histogram.update(blank_cells, episode_solved)

            # 8. Hindsight Experience Replay (HER): After an episode finishes (win or lose),
            # perform an extra training pass on its full trajectory. This provides immediate
            # feedback on the outcome.
            if episode_transitions:
                # For PER, we need to construct a valid batch_data tuple.
                # Since this is an out-of-band training pass (not sampled from the buffer),
                # we can use an importance sampling weight of 1.0 for all transitions.
                # The indices are not used for priority updates in this case, so they can be None.
                if isinstance(memory, PrioritizedReplayBuffer):
                    batch_data = (
                        episode_transitions,
                        None,
                        np.ones(len(episode_transitions)),
                    )
                else:
                    batch_data = episode_transitions

                # For successful episodes, this reinforces the good moves.
                # For failed episodes, this reinforces the penalties for bad moves.
                # Log the ponder steps from the more representative HER pass
                _her_ponder_steps = optimize_model(
                    policy_net,
                    target_net,
                    optimizer,
                    memory,
                    batch_data,
                    args.gamma,
                    current_ponder_penalty,
                    args.masking,
                )

            # 9. Update the target network periodically
            if i_episode % args.target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # 10. Logging and Reporting
            best_reward_str = ""
            if best_reward == -float("inf") or episode_reward > best_reward:
                # if best_reward != -float('inf'):
                best_reward_str = " (New Best Reward)"
                best_reward = episode_reward

            if episode_solved or best_reward_str or i_episode % args.log_episodes == 0:
                # if not env.fixed_puzzle:
                #     print(f"Episode {i_episode}: New puzzle:")
                #     print(format_grid_to_string(env.initial_puzzle))

                stats = env.episode_stats
                solved_ratio = (
                    f"{stats['correct_moves']:2d}/{stats['blank_cells_start']:2d}"
                )
                groups_completed = (
                    f"R:{stats['completed_rows']}"
                    f"/C:{stats['completed_cols']}"
                    f"/B:{stats['completed_boxes']}"
                )
                min_ponder, mean_ponder, max_ponder = ponder_steps
                print(
                    f"Episode {i_episode:6d}: "
                    # f"Level: {CURRICULUM_LEVELS[curriculum_level]['name']}, "
                    f"Level: {curriculum_level:2d}/{len(CURRICULUM_LEVELS):2d}, "
                    f"Steps: {episode_steps:3d}, "
                    f"Epoch Steps: {epoch_steps_done:6d}, "
                    f"Epsilon: {max(args.eps_end, current_epsilon):.4f}, "
                    f"Ponder Penalty: {current_ponder_penalty:.4f}, "
                    f"Ponder: ({min_ponder:.2f},{mean_ponder:.2f},{max_ponder:.2f}) "
                    f"Cells: {solved_ratio}, Groups: {groups_completed}, "
                    f"({'    Solved' if episode_solved else 'NOT Solved'}), "
                    f"Total Reward: {episode_reward: 8.2f}{best_reward_str}, "
                )

            # 11. Save the model periodically or at the end of training
            if args.save_model and (
                best_reward_str or final_episode or i_episode % 100 == 0
            ):
                model_str = "final model" if final_episode else "model checkpoint"
                print(f"Saving {model_str} to {args.save_model}...")
                torch.save(
                    {
                        "model_state_dict": policy_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "total_episodes_trained": total_episodes_trained,
                        "curriculum_level": curriculum_level,
                        "current_epsilon": current_epsilon,
                    },
                    args.save_model,
                )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
        if args.save_model:
            print(f"Saving final model state to {args.save_model}...")
            torch.save(
                {
                    "model_state_dict": policy_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_episodes_trained": total_episodes_trained,
                    "curriculum_level": curriculum_level,
                    "current_epsilon": current_epsilon,
                },
                args.save_model,
            )

    end_time = time.time()
    training_duration = end_time - start_time
    duration_str = str(timedelta(seconds=training_duration))
    time_per_step = training_duration / epoch_steps_done
    time_per_step_str = str(timedelta(seconds=time_per_step))

    difficulty_summary = (
        f" (Hardest level {81-min_clues_solved} cell(s))"
        if min_clues_solved != 99
        else ""
    )
    print(
        "\n"
        + "\n  ".join(
            [
                f"Training Complete {'='*60}",
                f"Final Best Reward: {best_reward:.2f} "
                f"over {total_episodes_trained} total episodes.",
                f"Total Solved: {solved_count}{difficulty_summary}",
                f"Total time: {duration_str} ({time_per_step_str} per step)",
            ]
        )
    )
    histogram.log("Training Performance by Difficulty")
    capability_score = histogram.get_capability_score()
    print(f"Final Capability Score: {capability_score:.3f}")

    return 0


def run_test_episode(args, env, policy_net, initial_state, show_boards=True):
    """Runs a single episode in test mode (exploitation only) and returns the result."""
    state = initial_state
    episode_steps = 0
    episode_reward = 0

    for _step in range(81):  # Max 81 steps
        with torch.no_grad():
            state_t = torch.from_numpy(state).to(args.device)
            one_hot_state = state_to_one_hot(state_t)
            output = policy_net(one_hot_state.unsqueeze(0))
            if isinstance(output, tuple):
                q_values = output[0]
            else:
                q_values = output

            if args.masking:
                # Use the GPU-accelerated version for masking.
                mask_t = generate_legal_mask_gpu(state_t)
                if not mask_t.any():
                    break  # No legal moves left

                additive_mask = torch.where(
                    mask_t, 0.0, ILLEGAL_ACTION_VALUE
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
        # Print 3 boards horizontally, with some gap
        print_grids(
            [
                env.initial_puzzle,
                env.current_grid,
                env.current_grid - env.initial_puzzle,
            ],
            ["Initial", "Final", "Moves"],
        )

    is_solved = np.array_equal(env.current_grid, env.solution_grid)
    return is_solved, episode_reward, episode_steps


def log_test_result(env, i_game, num_generated_games, steps, final_reward, is_solved):
    """Helper function to log the results of a single test game."""
    stats = env.episode_stats
    solved_ratio = f"{stats['correct_moves']:2d}/{stats['blank_cells_start']:2d}"
    groups_completed = (
        f"R:{stats['completed_rows']}"
        f"/C:{stats['completed_cols']}"
        f"/B:{stats['completed_boxes']}"
    )
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
    total_steps = 0
    total_reward = 0

    histogram = DifficultyHistogram()
    start_time = time.time()

    num_generated_games = args.test_games if not args.puzzle else args.test_games - 1

    # 1. Test on the fixed puzzle first, if provided
    if args.puzzle:
        print("\n1. Testing on the provided fixed puzzle...")
        # Force reset to the fixed puzzle
        env.fixed_puzzle = True
        state, num_clues, _ = env.reset()
        env.fixed_puzzle = args.fixed_puzzle  # Revert to original setting

        is_solved, final_reward, steps = run_test_episode(
            args, env, policy_net, state, show_boards=args.show_boards
        )
        if is_solved:
            solved_count += 1
        total_reward += final_reward
        total_steps += steps

        log_test_result(env, 0, num_generated_games, steps, final_reward, is_solved)
        histogram.update(env.episode_stats.get("blank_cells_start"), is_solved)

    # 2. Test on procedurally generated puzzles
    if num_generated_games > 0:
        print(
            f"\n2. Testing on {num_generated_games} generated puzzles "
            f"(Difficulty: {args.test_difficulty_min}-{args.test_difficulty_max})..."
        )
        diff_slope = (
            1 + args.test_difficulty_max - args.test_difficulty_min
        ) / num_generated_games
        for i_game in range(1, num_generated_games + 1):
            difficulty = args.test_difficulty_min + math.floor(
                (i_game - 1) * diff_slope
            )
            num_clues = 81 - difficulty
            state, num_clues, _ = env.reset(options={"num_clues": num_clues})
            is_solved, final_reward, steps = run_test_episode(
                args, env, policy_net, state, show_boards=args.show_boards
            )
            if is_solved:
                solved_count += 1
            total_reward += final_reward
            total_steps += steps

            log_test_result(
                env, i_game, num_generated_games, steps, final_reward, is_solved
            )
            histogram.update(env.episode_stats.get("blank_cells_start"), is_solved)

    # 3. Report final statistics
    end_time = time.time()
    test_duration = end_time - start_time
    duration_str = str(timedelta(seconds=test_duration))
    time_per_step_str = "N/A"
    if total_steps > 0:
        time_per_step = test_duration / total_steps
        time_per_step_str = str(timedelta(seconds=time_per_step))

    solve_rate = (solved_count / args.test_games) * 100 if args.test_games > 0 else 0
    avg_reward = total_reward / args.test_games if args.test_games > 0 else 0

    print(
        "\n"
        + "\n  ".join(
            [
                f"Test Phase Complete {'='*56}",
                f"Puzzles Solved: {solved_count} / {args.test_games} ({solve_rate:.1f}%)",
                f"Average Reward: {avg_reward:.2f}",
                f"Total time: {duration_str} ({time_per_step_str} per step)",
            ]
        )
    )
    histogram.log("Test Performance by Difficulty")
    capability_score = histogram.get_capability_score()
    print(f"Final Capability Score: {capability_score:.3f}")

    return 0


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Deep Q-Learning Sudoku Solver")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn6",
        choices=["cnn1", "cnn2", "cnn3", "cnn4", "cnn5", "cnn6", "transformer1"],
        help="Model architecture to use.",
    )
    # Training arguments
    parser.add_argument(
        "--episodes",
        type=int,
        default=MAX_EPISODES,
        help="Number of episodes to train.",
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        default=None,
        help="Initial Sudoku puzzle string (81 chars, 0 for blank).",
    )
    parser.add_argument(
        "--reward_shaping",
        action="store_true",
        help="Enable reward shaping (progress-based rewards).",
    )
    parser.add_argument(
        "--masking",
        action="store_true",
        help="Enable action masking (only choose blank cells).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for training reproducibility."
    )
    parser.add_argument(
        "--fixed_puzzle",
        action="store_true",
        help="Use only given puzzle for training.",
    )

    # Hyperparameter arguments
    parser.add_argument(
        "--lr", type=float, default=LR, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--gamma", type=float, default=GAMMA, help="Discount factor for future rewards."
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training."
    )
    parser.add_argument(
        "--memory_capacity",
        type=int,
        default=MEMORY_CAPACITY,
        help="Capacity of the replay buffer.",
    )
    parser.add_argument(
        "--target_update",
        type=int,
        default=TARGET_UPDATE,
        help="Frequency (in episodes) to update the target network.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay for the AdamW optimizer.",
    )
    parser.add_argument(
        "--ponder_penalty",
        type=float,
        default=PONDER_PENALTY,
        help="Final penalty for each computation step in ACT models (e.g., cnn6).",
    )
    parser.add_argument(
        "--ponder_penalty_start",
        type=float,
        default=PONDER_PENALTY_START,
        help="Starting ponder penalty for annealing.",
    )
    parser.add_argument(
        "--ponder_penalty_anneal_episodes",
        type=int,
        default=PONDER_PENALTY_ANNEAL_EPISODES,
        help="Number of episodes to anneal the ponder penalty over.",
    )

    # Epsilon-greedy arguments
    parser.add_argument(
        "--eps_start",
        type=float,
        default=EPS_START,
        help="Starting value of epsilon for exploration.",
    )
    parser.add_argument(
        "--eps_end", type=float, default=EPS_END, help="Minimum value of epsilon."
    )
    parser.add_argument(
        "--eps_decay", type=float, default=EPS_DECAY, help="Decay rate for epsilon."
    )

    parser.add_argument(
        "--log_episodes", type=int, default=10, help="Log info once every N episodes."
    )

    # Testing arguments
    default_test_games = (
        TEST_DIFFICULTY_MAX - TEST_DIFFICULTY_MIN + 1
    ) * TEST_GAMES_REPEAT
    parser.add_argument(
        "--test_games",
        type=int,
        default=default_test_games,
        help="Number of games to test after training.",
    )
    parser.add_argument(
        "--test_difficulty_min",
        type=int,
        default=TEST_DIFFICULTY_MIN,
        help="Min blank cells for test puzzles.",
    )
    parser.add_argument(
        "--test_difficulty_max",
        type=int,
        default=TEST_DIFFICULTY_MAX,
        help="Max blank cells for test puzzles.",
    )
    parser.add_argument(
        "--show_boards", action="store_true", help="Show test puzzles and solutions."
    )
    parser.add_argument(
        "--test_seed",
        type=int,
        default=42,
        help="Random seed for test reproducibility.",
    )

    # Model persistence
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the trained model."
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to load a pre-trained model."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 2),
        help="Number of CPU workers for puzzle generation.",
    )

    args = parser.parse_args()

    return args, parser


def main() -> int:
    """Main function."""

    args, _parser = parse_args()

    # Set random seeds for training reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Using random seed for training: {args.seed}")

    # Initialize Environment, Networks, Optimizer, and Memory
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model
    solver = None
    if args.model == "cnn1":
        solver = DQNSolverCNN1
    elif args.model == "cnn2":
        solver = DQNSolverCNN2
    elif args.model == "cnn3":
        solver = DQNSolverCNN3
    elif args.model == "cnn4":
        solver = DQNSolverCNN4
    elif args.model == "cnn5":
        solver = DQNSolverCNN5
    elif args.model == "cnn6":
        solver = DQNSolverCNN6
    elif args.model == "transformer1":
        solver = DQNSolverTransformer
    else:
        print(f"Invalid model specified: {args.model}, exiting.")
        return 1
    print(f"Using model {args.model}.")

    # Load a pre-trained model if specified to determine starting curriculum level
    checkpoint = None
    if args.load_model:
        try:
            print(f"Loading metadata from {args.load_model}...")
            checkpoint = torch.load(args.load_model, map_location=args.device)
            total_episodes_trained = checkpoint.get("total_episodes_trained", 0)
            args.start_episode = total_episodes_trained + 1
            args.curriculum_level = checkpoint.get("curriculum_level", 0)
            args.current_epsilon = checkpoint.get("current_epsilon", args.eps_start)
            print(f"Will resume training from episode {args.start_episode}.")
            print(
                f"Starting curriculum level: {args.curriculum_level} '{CURRICULUM_LEVELS[args.curriculum_level]['name']}'."
            )
        except FileNotFoundError:
            print(
                f'Warning: Model file not found at "{args.load_model}". Starting from scratch.'
            )
            args.start_episode = 1
            args.curriculum_level = 0
            args.current_epsilon = args.eps_start
        except Exception as e:
            print(
                f'Warning: {e} reading file "{args.load_model}". Starting from scratch.'
            )
            args.start_episode = 1
            args.curriculum_level = 0
            args.current_epsilon = args.eps_start
    else:
        args.start_episode = 1
        args.curriculum_level = 0
        args.current_epsilon = args.eps_start

    # Initialize the puzzle generator only if needed for training.
    # The test loop generates its own specific puzzles synchronously.
    puzzle_generator = None
    if not args.fixed_puzzle and args.episodes > 0:
        initial_min_clues, initial_max_clues = CURRICULUM_LEVELS[args.curriculum_level][
            "clues"
        ]
        puzzle_generator = PuzzleGenerator(
            num_workers=args.workers,
            initial_min_clues=initial_min_clues,
            initial_max_clues=initial_max_clues,
        )

    env = SudokuEnv(
        puzzle_str=args.puzzle,
        reward_shaping=args.reward_shaping,
        fixed_puzzle=args.fixed_puzzle,
        puzzle_generator=puzzle_generator,
    )

    policy_net = solver(
        env.observation_space.shape, env.action_space.n, args.device
    ).to(args.device)
    target_net = solver(
        env.observation_space.shape, env.action_space.n, args.device
    ).to(args.device)
    optimizer = optim.AdamW(
        policy_net.parameters(),
        lr=args.lr,
        amsgrad=True,
        weight_decay=args.weight_decay,
    )
    memory = PrioritizedReplayBuffer(args.memory_capacity)

    # Load a pre-trained model if specified
    if checkpoint:  # args.load_model
        try:
            print(f"Loading model and optimizer state from {args.load_model}...")
            policy_net.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(
                f"Could not load model weights from {args.load_model}: {e}. Using fresh model."
            )
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
            if puzzle_generator:
                # Start the puzzle generator
                puzzle_generator.start()

            # train is responsible for saving per args.save_model (checkpoint and final)
            rc = train(args, env, policy_net, target_net, optimizer, memory)
        if not rc:
            if puzzle_generator:
                # test() does not use puzzle_generator
                puzzle_generator.stop()
            # Run the test phase if specified
            if args.test_games > 0:
                if args.test_seed is not None:
                    # Set random seeds for training reproducibility
                    random.seed(args.test_seed)
                    np.random.seed(args.test_seed)
                    torch.manual_seed(args.test_seed)
                    print(f"Using random seed for test: {args.test_seed}")

                rc = test(args, env, policy_net)
    except Exception as e:
        # traceback.print_exc()
        print(f"\nError {e} occurred during training or testing. Exiting.")
        rc = 1
    finally:
        # This will run whether the training completes successfully or fails
        restore_sleep_function()
        if puzzle_generator:
            puzzle_generator.stop()
    return rc


if __name__ == "__main__":
    exit(main())
