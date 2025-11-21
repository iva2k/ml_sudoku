#!/usr/bin/env python3

import numpy as np

# Q-learning hyperparameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 0.1
num_episodes = 10000

# Sudoku board
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Convert the board to a numpy array
board = np.array(board)

# Define actions for the agent (filling a number in a cell)
actions = np.arange(1, 10)


def get_empty_cell_indices(board):
    """Get the indices of empty cells (zeros) in the board."""
    return np.argwhere(board == 0)


def is_valid_move(board, number, row, col):
    """Check if a number can be placed in a given cell (row, col)."""
    # Check row and column
    if number in board[row, :] or number in board[:, col]:
        return False

    # Check 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    if number in board[box_row:box_row+3, box_col:box_col+3]:
        return False

    return True


def choose_action(board, row, col):
    """Choose an action (number) for the given state (board and cell position)."""
    if np.random.rand() < exploration_rate:
        # Explore: choose a random action
        return np.random.choice(actions)
    else:
        # Exploit: choose the action with the highest Q-value
        state = tuple(map(tuple, board.copy().tolist()))
        state[row][col] = actions[:, None].tolist()
        return np.argmax(Q[state])


def update_q_value(state, action, reward, next_state):
    """Update the Q-value using the Q-learning update rule."""
    Q[state][action] += learning_rate * (
        reward + discount_factor * np.max(Q[next_state]) - Q[state][action]
    )


# Initialize the Q-table
Q = {}
for row in range(9):
    for col in range(9):
        state = tuple(map(tuple, board.copy().tolist()))
        state[row][col] = actions[:, None].tolist()
        Q[state] = np.zeros(len(actions))


# Run Q-learning
for episode in range(num_episodes):
    current_board = board.copy()
    done = False

    while not done:
        empty_cells = get_empty_cell_indices(current_board)
        if len(empty_cells) == 0:
            done = True
            break

        row, col = empty_cells[np.random.choice(len(empty_cells))]
        action = choose_action(current_board, row, col)
        number = actions[action]

        if is_valid_move(current_board, number, row, col):
            current_board[row, col] = number
        else:
            # Invalid move, penalize the agent
            current_board[row, col] = -1

        # Check if the board is solved
        if np.count_nonzero(current_board == 0) == 0:
            done = True

    # Update Q-values
    for t in range(1, len(empty_cells) + 1):
        state = tuple(map(tuple, board.copy().tolist()))
        state[empty_cells[:t, 0], empty_cells[:t, 1]] = actions[:, None].tolist()
        reward = 1 if t == len(empty_cells) else 0
        next_state = tuple(map(tuple, current_board.copy().tolist()))
        update_q_value(state, action, reward, next_state)

# Solve Sudoku using the learned Q-values
current_board = board.copy()
done = False

while not done:
    empty_cells = get_empty_cell_indices(current_board)
    if len(empty_cells) == 0:
        done = True
        break

    row, col = empty_cells[0]
    state = tuple(map(tuple, current_board.copy().tolist()))
    state[row][col] = actions[:, None].tolist()
    action = np.argmax(Q[state])
    number = actions[action]

    if is_valid_move(current_board, number, row, col):
        current_board[row, col] = number
    else:
        # Invalid move, break the loop
        break

    # Check if the board is solved
    if np.count_nonzero(current_board == 0) == 0:
        done = True

# Print the solved Sudoku board
print(current_board)
