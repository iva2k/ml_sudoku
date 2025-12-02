#!/usr/bin/env python3
# sudoku.py

"""Sudoku puzzle solver and utilities."""

import copy
from datetime import timedelta
import random
from timeit import default_timer as timer
from typing import Callable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

Board = NDArray[np.int_]  # Shape: (9, 9) = np.ndarray((9, 9), dtype=int)
BoardOrStr = Union[Board, str]


# Utility functions
def arr_to_str(board: Board) -> str:
    """Convert a 2D NumPy array to a string."""
    s = ""
    for row in board:
        for c in row:
            s += str(c)
    return s


def str_to_arr(board: str) -> Board:
    """Convert a string to a 2D NumPy array."""
    a = [int(c) for c in board]
    return np.reshape(
        a,
        (
            9,
            9,
        ),
    )


def format_grid_to_strings(grid: Board) -> List[str]:
    """Formats a 9x9 grid for pretty printing, replacing 0s with spaces."""
    s = []
    for r in range(9):
        if r > 0 and r % 3 == 0:
            s.append("------+-------+------")

        # row_str = []
        # for c in range(9):
        #     digit = grid[r, c]
        #     row_str.append(str(digit) if digit != 0 else " ")
        row_str = "".join([str(d) if d != 0 else " " for d in grid[r, :]])

        s.append(
            " ".join(row_str[0:3])
            + " | "
            + " ".join(row_str[3:6])
            + " | "
            + " ".join(row_str[6:9])
        )
    return s


def format_grid_to_string(grid: Board) -> str:
    """Formats a 9x9 grid for pretty printing, replacing 0s with spaces."""
    s = format_grid_to_strings(grid)
    return "\n".join(s)


def print_grid(board: BoardOrStr) -> None:
    """Print Sudoku board."""
    board = str_to_arr(board) if isinstance(board, str) else board
    s = format_grid_to_string(board)
    print(s)


def _is_valid1(g: Board, r, c, n):
    """
    Check to see if placing number 'n' at (row, col) is valid.
    Assumes the cell at (row, col) is currently 0.
    """
    if n in g[r, :]:
        return False
    if n in g[:, c]:
        return False
    box_r, box_c = 3 * (r // 3), 3 * (c // 3)
    if n in g[box_r : box_r + 3, box_c : box_c + 3]:
        return False
    return True


def _is_valid(board: Board, row: int, col: int, n: int) -> bool:
    """
    Optimized check to see if placing number 'n' at (row, col) is valid.
    Assumes the cell at (row, col) is currently 0.
    """
    # Check if 'n' is already in the same row or column
    if np.any(board[row, :] == n) or np.any(board[:, col] == n):
        return False

    # Check 3x3 subgrid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    if np.any(board[start_row : start_row + 3, start_col : start_col + 3] == n):
        return False
    return True


def is_valid(board: BoardOrStr, row: int, col: int, n: int) -> bool:
    """
    Optimized check to see if placing number 'n' at (row, col) is valid.
    Assumes the cell at (row, col) is currently 0.
    """
    board = str_to_arr(board) if isinstance(board, str) else board
    return _is_valid(board, row, col, n)


def count_blanks(board: BoardOrStr):
    """Count the number of blank/empty cells (0s) on the board."""
    # TODO: (now) Optimize
    board = str_to_arr(board) if isinstance(board, str) else board
    count = 0
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                count += 1
    return count


def find_blank(board: Board):
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                return (r, c)
    return None


def find_next_blank(board: Board, row: int = None, col: int = None):
    """
    Finds the next blank/empty cell (0) on the board.
    - If row/col are None, finds the very first empty cell.
    - If row/col are provided, finds the first empty cell after (row, col).
    Returns a (row, col) tuple or False if no empty cells are found.
    """
    board = str_to_arr(board) if isinstance(board, str) else board

    empty_cells = np.argwhere(board == 0)
    if empty_cells.size == 0:
        return False

    if row is None:  # Find the first empty cell on the board
        return tuple(empty_cells[0])

    # Find the first empty cell *after* the given (row, col)
    current_pos_flat = row * 9 + col
    for r, c in empty_cells:
        if r * 9 + c > current_pos_flat:
            return (r, c)
    return False


def all_possible(board: BoardOrStr, row: int, col: int) -> List[int]:
    """
    Return all possible numbers that can be placed at (row, col)
    without violating the Sudoku rules.
    """
    board = str_to_arr(board) if isinstance(board, str) else board
    possibilities = [n for n in range(1, 10) if _is_valid(board, row, col, n)]
    return possibilities


def _count_solutions(board: Board, count_limit: int = 2) -> int:
    """Counts the number of solutions for a board up to a limit using backtracking."""
    count = 0

    def _solve():
        """Custom backtracking solver that counts soultions, limited by count_limit."""
        nonlocal count
        blank_cell = find_next_blank(board)
        if blank_cell is False:
            count += 1
            return count >= count_limit

        row, col = blank_cell
        vals = all_possible(board, row, col)
        for n in vals:
            board[row][col] = n
            if _solve():
                return True
        board[row][col] = 0  # Backtrack
        return False

    _solve()
    return count


def count_solutions(board: BoardOrStr, count_limit: int = 2) -> int:
    """Counts the number of solutions for a board up to a limit using backtracking."""
    board = str_to_arr(board) if isinstance(board, str) else board
    return _count_solutions(board, count_limit)


# Sudoku Solvers


def _solve_brute(board: Board) -> Optional[Board]:
    """Solves the Sudoku puzzle using backtracking."""
    blank_cell = find_next_blank(board)
    if blank_cell is False:
        return board
    row, col = blank_cell
    for n in range(1, 10):
        if _is_valid(board, row, col, n):
            board[row][col] = n
            result = _solve_brute(board)
            if result is not False:
                return result
        board[row][col] = 0  # Backtrack
    return False


def solve_brute(board: BoardOrStr) -> Optional[Board]:
    """Solves the Sudoku puzzle using backtracking."""
    board = str_to_arr(board) if isinstance(board, str) else board
    return _solve_brute(board)


def _solve_eliminator(board: Board) -> Optional[Board]:
    """Solves the Sudoku puzzle using backtracking and eliminator heuristic."""
    row, col = None, None
    while True:
        not_solved = find_next_blank(board)
        if not_solved is False:
            # Solved
            return board
        next_blank_cell = find_next_blank(board, row, col)
        if next_blank_cell is False:
            # Reached the end of board, no elimination possible, do brute-force
            row, col = not_solved
            vals = all_possible(board, row, col)
            for n in vals:
                # brute_board = copy.copy(board)
                brute_board = board
                brute_board[row][col] = n
                result = _solve_eliminator(brute_board)
                if result is not False:
                    return result
                brute_board[row][col] = 0
            return False
        row, col = next_blank_cell
        vals = all_possible(board, row, col)
        if len(vals) == 1:
            board[row][col] = vals[0]
            row, col = None, None

    return False


def solve_eliminator(board: BoardOrStr) -> Optional[Board]:
    """Solves the Sudoku puzzle using backtracking and eliminator heuristic."""
    board = str_to_arr(board) if isinstance(board, str) else board
    return _solve_eliminator(board)


# Sudoku Builder


def generate_solved_sudoku1() -> Board:
    """Generates a complete, solved Sudoku grid using a randomized backtracking algorithm."""
    grid = np.zeros((9, 9), dtype=np.int32)
    nums = list(range(1, 10))

    def solve(g):
        find = find_blank(g)
        if not find:
            return True  # Solved
        else:
            row, col = find

        random.shuffle(nums)  # Randomize numbers to try
        for num in nums:
            if is_valid(g, row, col, num):
                g[row, col] = num
                if solve(g):
                    return True
                g[row, col] = 0  # Backtrack
        return False

    solve(grid)
    return grid.astype(np.int32)


def generate_solved_sudoku() -> Board:
    """Generates a complete, solved Sudoku grid using constraint-preserving transformations."""

    def _shuffle_digits(grid: Board) -> Board:
        """Permutes the digits (1-9) across the entire grid."""
        mapping = list(range(1, 10))
        random.shuffle(mapping)  # e.g., [5, 8, 1, 4, 3, 7, 2, 9, 6]
        shuffled_grid = np.zeros_like(grid)
        for old_val in range(1, 10):
            new_val = mapping[old_val - 1]
            shuffled_grid[grid == old_val] = new_val
        return shuffled_grid

    def _swap_major_rows(grid: Board) -> Board:
        """Permutes the 3 major row blocks (rows 0-2, 3-5, 6-8)."""
        row_blocks = np.split(grid, 3, axis=0)  # Split into 3 arrays of shape (3, 9)
        random.shuffle(row_blocks)
        return np.concatenate(row_blocks, axis=0)

    def _swap_major_cols(grid: Board) -> Board:
        """Permutes the 3 major column blocks (cols 0-2, 3-5, 6-8)."""
        col_blocks = np.split(grid, 3, axis=1)  # Split into 3 arrays of shape (9, 3)
        random.shuffle(col_blocks)
        return np.concatenate(col_blocks, axis=1)

    def _shuffle_minor_rows(grid: Board) -> Board:
        """Shuffles rows within each of the three 3x3 row blocks."""
        new_grid = grid.copy()
        for i in range(3):
            # Indices for the current row block (e.g., 0, 1, 2)
            indices = list(range(i * 3, (i + 1) * 3))
            random.shuffle(indices)
            new_grid[i * 3 : (i + 1) * 3, :] = grid[indices, :]
        return new_grid

    def _shuffle_minor_cols(grid: Board) -> Board:
        """Shuffles columns within each of the three 3x3 column blocks."""
        new_grid = grid.copy()
        for i in range(3):
            # Indices for the current column block (e.g., 0, 1, 2)
            indices = list(range(i * 3, (i + 1) * 3))
            random.shuffle(indices)
            new_grid[:, i * 3 : (i + 1) * 3] = grid[:, indices]
        return new_grid

    base = 3
    side = base * base

    # 1. Create Base Grid (Canonical Pattern)
    # This formula is guaranteed to produce a solved grid
    def pattern(r, c):
        return (base * (r % base) + r // base + c) % side

    nums = np.array(list(range(1, side + 1)))  # [1, 2, ..., 9]

    # Initial canonical solved grid
    grid = np.array(
        [[nums[pattern(r, c)] for c in range(side)] for r in range(side)],
        dtype=np.int32,
    )

    # 2. Apply Permutations (for variety)
    # This order is safe and preserves all Sudoku rules:
    grid = _shuffle_digits(grid)
    grid = _swap_major_rows(grid)
    grid = _swap_major_cols(grid)
    grid = _shuffle_minor_rows(grid)
    grid = _shuffle_minor_cols(grid)

    return grid


def _clue_grid(solved_grid: Board, num_clues: int = 30) -> Board:
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


def get_unique_sudoku(solution: Board, num_clues: int, max_tries: int = 1000) -> Board:
    """
    Generates Sudoku puzzle with 1 provided solution and the given number of clues.
    """
    tries = 0
    while tries < max_tries:
        puzzle = _clue_grid(solution, num_clues)
        # Check for a unique solution
        if count_solutions(puzzle.copy()) == 1:
            return puzzle
        tries += 1
        if tries > 10:
            print(f"DEBUG: slow puzzle generation for {num_clues} clues, try {tries}.")
    raise RuntimeError(
        f"Unable to find a {num_clues} clues puzzle with a 1 solution after {max_tries} tries."
    )


class Sudoku:
    """Sudoku puzzle."""

    def __init__(
        self, quiz: Optional[BoardOrStr] = None, solution: Optional[BoardOrStr] = None
    ):
        self._quiz = quiz
        self._solution = solution

    @property
    def quiz(self) -> BoardOrStr:
        """Puzzle quiz."""
        return self._quiz

    @property
    def solution(self) -> BoardOrStr:
        """Puzzle solution."""
        return self._solution

    def solver(
        self,
        board: BoardOrStr,
        fnc: Optional[Callable[[BoardOrStr], Optional[Board]]] = None,
        num_iter: int = 1000,
    ) -> None:
        """
        Runs Sudoku solver using the specified function and prints the results.
        """
        if num_iter <= 0:
            raise ValueError("num_iter must be a positive integer")
        if not fnc:
            fnc = solve_brute
        name = ""
        if fnc == solve_brute:
            name = "solve_brute"
        elif fnc == solve_eliminator:
            name = "solve_eliminator"

        print()
        print(f"Solver using method {name}")
        print_grid(board)
        print(f"Blanks: {count_blanks(board)}")
        print()
        start_time = timer()
        for _i in range(num_iter):
            q = copy.copy(board)
            result = fnc(q)
        end_time = timer()
        elapsed_time = end_time - start_time
        elapsed_time = elapsed_time / num_iter

        time_str = str(timedelta(seconds=elapsed_time))
        if result is not False:
            print(f"Method {name} solved in {time_str}:")
            print_grid(result)
        else:
            print(f"Method {name} failed to solve")


def main():
    """Main function to run the Sudoku solver."""
    num_iter = 100
    s = Sudoku(
        "000308600302400058005020071586000400000007002090140000403096105001280006070000030",
        "719358624362471958845629371586932417134867592297145863423796185951283746678514239",
    )
    # s = Sudoku('000260701680070090190004500820100040004602900050003028009300074040050036703018000', '435269781682571493197834562826195347374682915951743628519326874248957136763418259')

    # TODO: (when needed) solve_eliminator fails on this sudoku:
    # s = Sudoku("400700000080000029000009150600095000050030080000170004036200000970000040000007005", "495721836187356429263849157618495273754632981329178564536214798971583642842967315")

    s.solver(s.quiz, solve_brute, num_iter)
    s.solver(s.quiz, solve_eliminator, num_iter)
    print()
    print("Known solution:")
    print_grid(s.solution)

    start_time = timer()
    for _i in range(num_iter):
        count = count_solutions(s.quiz)
    end_time = timer()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time / num_iter
    time_str = str(timedelta(seconds=elapsed_time))
    print(f"Number of solutions for quiz: {count}, found in {time_str}")


if __name__ == "__main__":
    main()
