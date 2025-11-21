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

## Phase 3: ...

"""

import argparse
from collections import deque, namedtuple
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters (Placeholders - adjust these during Phase 3)
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.99995
TARGET_UPDATE = 50 # 1000  # How often to update the target network
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
LR = 0.0001
MAX_EPISODES = 50000
WEIGHT_DECAY=0.01


# Define the state/action transition data structure for the Replay Buffer
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class SudokuEnv(gym.Env):
    """
    A custom Gymnasium environment for the 9x9 Sudoku puzzle.
    State: The current 9x9 grid.
    Action: (row, col, digit) to place. Total 729 actions.
    """

    def __init__(self, puzzle_str=None):
        super().__init__()

        # 9 rows, 9 columns, digits 1-9.
        # Action space is discrete, but we'll map 0-728 to (row, col, digit)
        self.action_space = gym.spaces.Discrete(9 * 9 * 9)
        # Observation space is 9x9 grid, values 0 (empty) to 9
        self.observation_space = gym.spaces.Box(
            low=0, high=9, shape=(9, 9), dtype=np.int32)

        # Initial puzzle (0 for empty cells)
        self.initial_puzzle = self._parse_puzzle(puzzle_str)
        self.current_grid = self.initial_puzzle.copy()

        print("Sudoku Environment Initialized.")

    def _parse_puzzle(self, puzzle_str):
        """Converts an 81-char string (0s for empty) into a 9x9 numpy array."""
        if puzzle_str is None:
            # Default easy puzzle for starting development
            puzzle_str = "000260701680070090190004500820100040004602900050003028009300074040050036703018000"

        # Ensure the string is 81 characters long and contains only digits
        if len(puzzle_str) != 81 or not puzzle_str.isdigit():
            raise ValueError("Puzzle string must be 81 digits (0-9).")

        grid = np.array([int(c) for c in puzzle_str]).reshape((9, 9))
        return grid

    def reset(self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment to the initial puzzle state."""
        super().reset(seed=seed, options=options)
        self.current_grid = self.initial_puzzle.copy()
        observation = self.current_grid.astype(np.float32)
        info = {}
        return observation, info

    def step(self, action):
        """
        Applies the action (0-728) to the grid and returns next_state, reward, done, info.

        # --- PHASE 2: CORE LOGIC IMPLEMENTATION (TODO) ---
        """
        # Map the single integer action back to (row, col, digit)
        row = (action // 81) % 9
        col = (action // 9) % 9
        digit = (action % 9) + 1  # Digits are 1-9

        reward = 0.0
        terminated = False
        truncated = False

        # 1. Check if the cell is already filled (initial puzzle cells are fixed)
        if self.initial_puzzle[row, col] != 0:
            reward = -10.0  # Penalty for overwriting a fixed cell
        else:
            # 2. Tentatively place the digit
            temp_grid = self.current_grid.copy()
            temp_grid[row, col] = digit

            # 3. Check for Sudoku Rule Violations (Simplified placeholder)
            if self._is_valid_placement(temp_grid, row, col, digit):
                self.current_grid[row, col] = digit
                reward = 1.0  # Small positive reward for a valid move

                # 4. Check if Solved
                if np.all(self.current_grid != 0) and self._is_fully_solved(self.current_grid):
                    reward = 100.0  # Large reward for solving
                    terminated = True
            else:
                reward = -5.0  # Penalty for an invalid move (violates rules)

        # State is returned as float32 for PyTorch compatibility
        observation = self.current_grid.astype(np.float32)
        info = {"current_reward": reward}

        return observation, reward, terminated, truncated, info

    def _is_valid_placement(self, grid, r, c, d):
        """Checks if placing digit 'd' at (r, c) is a valid move according to Sudoku rules."""
        # Check if 'd' is already in the same row or column
        # Note: We don't need to exclude the current cell (r, c) because we are checking
        # before the digit is placed, or we are checking a temporary grid.
        if d in grid[r, :] or d in grid[:, c]:
            return False

        # Check 3x3 subgrid
        start_r, start_c = 3 * (r // 3), 3 * (c // 3)
        box = grid[start_r:start_r + 3, start_c:start_c + 3]
        if d in box:
            return False

        return True

    def _is_fully_solved(self, grid):
        """Checks if the grid is full and all rows, columns, and 3x3 boxes are valid."""
        # TODO: Implement full solved check logic here

        # Check all rows and columns
        for i in range(9):
            if not (len(np.unique(grid[i, :])) == 9 and np.all(grid[i, :] > 0)):
                return False  # Row check
            if not (len(np.unique(grid[:, i])) == 9 and np.all(grid[:, i] > 0)):
                return False  # Column check

        # Check all 3x3 boxes
        for i in range(3):
            for j in range(3):
                box = grid[i*3:(i+1)*3, j*3:(j+1)*3]
                if not (len(np.unique(box)) == 9 and np.all(box > 0)):
                    return False

        return True

    def render(self):
        """Prints the current state of the Sudoku grid."""
        print(self.current_grid)

    def close(self):
        pass

# --- PHASE 1: REPLAY BUFFER (CARCASS) ---


class ReplayBuffer:
    """Stores experience tuples to stabilize training."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Returns a random batch of transitions."""
        # TODO: Implement sampling logic
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- PHASE 1: DQN NETWORK (CARCASS) ---


class DQNSolver(nn.Module):
    """
    The Deep Q-Network. Takes the 9x9 grid state and outputs Q-values for 729 actions.

    # --- PHASE 3: NETWORK IMPLEMENTATION (TODO) ---
    """

    def __init__(self, input_shape, output_size):
        super().__init__()
        # Input shape will be (batch_size, 1, 9, 9) if we use unsqueezed input

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


def get_action(state, policy_net, action_space):
    """
    Implements the epsilon-greedy policy.

    # --- PHASE 3: EPSILON-GREEDY (TODO) ---
    """
    global EPS_START #0.24.1, EPS_END, EPS_DECAY
    # Use max to ensure we don't go below EPS_END
    epsilon = max(EPS_END, EPS_START)

    # Epsilon decay
    EPS_START *= EPS_DECAY

    if random.random() < epsilon:
        # Explore: Choose a random action
        return action_space.sample()
    else:
        # Exploit: Choose the action with the highest predicted Q-value
        with torch.no_grad():
            # Add batch dimension and get Q-values
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)
            q_values = policy_net(state_tensor)
            # Use argmax to get the best action index
            return q_values.argmax().item()


def optimize_model(policy_net, target_net, optimizer, memory):
    """
    Performs one step of optimization on the Policy Network.

    # --- PHASE 3: BELLMAN LOSS AND BACKPROP (TODO) ---
    """
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Convert batch of transitions to tensors
    non_final_mask = torch.tensor(
        tuple(map(lambda d: not d, batch.done)), dtype=torch.bool)
    non_final_next_states = torch.stack(
        [s for s, done in zip(batch.next_state, batch.done) if not done])

    state_batch = torch.stack(batch.state)
    action_batch = torch.tensor(batch.action).unsqueeze(1)  # [B, 1]
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32)

    # Compute Q(s_t, a) - the Q-values for the actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) = max_a Q(s_{t+1}, a) for non-terminal next states
    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values (target)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss (a robust form of MSE)
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
    parser.add_argument('--episodes', type=int,
                        default=MAX_EPISODES, help='Number of episodes to train.')
    parser.add_argument('--puzzle', type=str, default=None,
                        help='Initial Sudoku puzzle string (81 chars, 0 for empty).')
    args = parser.parse_args()
    return args, parser


def main():
    """
    # --- PHASE 1: MAIN FUNCTION AND ARGUMENT PARSING (CARCASS) ---
    """

    args, _parser = parse_args()

    # 1. Initialize Environment, Networks, and Optimizer
    env = SudokuEnv(puzzle_str=args.puzzle)

    input_shape = env.observation_space.shape
    output_size = env.action_space.n  # 729

    policy_net = DQNSolver(input_shape, output_size)
    target_net = DQNSolver(input_shape, output_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is always in evaluation mode

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=WEIGHT_DECAY)
    memory = ReplayBuffer(MEMORY_CAPACITY)

    # Training loop
    steps_done = 0
    best_reward = -float('inf')

    print(f"Starting Training for {args.episodes} episodes...")

    for i_episode in range(1, args.episodes + 1):
        # 2. Reset the environment and get initial state
        state, _ = env.reset()

        episode_reward = 0

        # 3. Run the episode
        for t in range(81):  # Max 81 steps (cells) per episode
            action = get_action(state, policy_net, env.action_space)

            # 4. Take action in environment
            observation, reward, terminated, truncated, _info = env.step(action)
            next_state = observation
            done = terminated or truncated

            # 5. Store the transition in the replay memory
            # Convert numpy arrays to tensors for storage
            state_tensor = torch.from_numpy(state).float()
            next_state_tensor = torch.from_numpy(next_state).float()
            memory.push(state_tensor, action, reward, next_state_tensor, done)

            # Move to the next state
            state = next_state
            episode_reward += reward

            # 6. Perform optimization step
            optimize_model(policy_net, target_net, optimizer, memory)
            steps_done += 1

            if done:
                break

        # 7. Update the target network periodically
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 8. Logging and Reporting
        if i_episode % 100 == 0:
            print(
                f"Episode: {i_episode}/{args.episodes}, Steps: {t+1}, Total Reward: {episode_reward:.2f}, Epsilon: {max(EPS_END, EPS_START):.4f}")
            if episode_reward > best_reward:
                best_reward = episode_reward
                print(f"New Best Reward! {best_reward}")
                # TODO: Optional: Save model checkpoint here

    print("Training Complete.")
    # TODO: Optional: Run a final test episode and display the solved grid


if __name__ == "__main__":
    main()
