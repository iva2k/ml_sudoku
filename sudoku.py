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


def print_grids(grids: List[BoardOrStr], titles: List[str], gap: str = "    ") -> None:
    """Print multiple Sudoku boards horizontally.
    For example, useful for puzzle, solution and delta.
    """

    grids = [str_to_arr(grid) if isinstance(grid, str) else grid for grid in grids]
    grids_str = [format_grid_to_strings(grid) for grid in grids]

    # for i, (grid, title) in enumerate(zip(grids_str, titles)):

    # Print boards horizontally, with some gap
    headings = titles
    print(gap.join([f"{h:21s}" for h in headings]))

    lines = len(grids_str[0])
    print(
        "\n".join(
            [gap.join([grid_str[i] for grid_str in grids_str]) for i in range(lines)]
        )
    )


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
    board = str_to_arr(board) if isinstance(board, str) else board
    return np.count_nonzero(board == 0)


def find_blank(board: Board):
    """
    Find the first blank/empty cell (0) on the board.
    """
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                return (r, c)
    return None


def find_next_blank(board: Board, row: int = None, col: int = None):
    """
    Find the next blank/empty cell (0) on the board.
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


def get_unique_sudoku(solution: Board, num_clues: int, debug: bool=False) -> Board:
    """
    Generates a Sudoku puzzle with a unique solution by starting with a solved
    grid and recursively removing chunks of cells. This is much more efficient
    than removing cells one-by-one.
    The resulting puzzle will have exactly `num_clues` if successful.
    If not enough blank cells that keep unique solution, the resulting puzzle will
    have more clues than `num_clues`. Rerunning this function or starting from a
    different solution may be able to find a better set of blank cells.
    """
    coords_to_try = list(np.ndindex(9, 9))
    random.shuffle(coords_to_try)
    if num_clues == 81 - 54:
        print(
            f'num_clues={num_clues}, sol="{arr_to_str(solution)}" coords={coords_to_try}'
        )
    return _get_unique_sudoku(solution, num_clues, coords_to_try, debug)


def _get_unique_sudoku(solution: Board, num_clues: int, coords_to_try, debug: bool=False) -> Board:
    """
    Generates a Sudoku puzzle with a unique solution by starting with a solved
    grid and recursively removing chunks of cells. This is much more efficient
    debug and than removing cells one-by-one.
    The resulting puzzle will have exactly `num_clues` if successful.
    If not enough blank cells that keep unique solution, the resulting puzzle will
    have more clues than `num_clues`. Rerunning this function or starting from a
    different solution may be able to find a better set of blank cells.
    """
    puzzle = solution.copy()

    count_sol_count = 0  # Count how many times we called expensive count_solutions()
    keys_at = []

    # This is our pointer in the coords_to_try list
    current_pos = 0
    while current_pos < 81:
        # Stop if we have reached the desired number of clues
        current_blanks = count_blanks(puzzle)
        clues_on_board = 81 - current_blanks
        blanks_to_create = clues_on_board - num_clues
        if blanks_to_create <= 0:
            _ = debug and print(
                f"DEBUG: Generated a puzzle with {clues_on_board} clues / {current_blanks} blanks: "
                f"{count_sol_count} calls to count_solutions(), "
                f"{len(keys_at)} keys at {keys_at}."
            )
            return puzzle

        # 1. Determine chunk size with adaptive stride.
        # Use a large stride early on when key cells are sparse, and reduce it as
        # the puzzle gets denser to avoid excessive `count_solutions` calls.
        if current_pos < 18 or count_sol_count < 3:
            stride = 32
        elif count_sol_count < 5:
            stride = 16
        elif count_sol_count < 10:
            stride = 4
        else:
            stride = 1

        chunk_size_desired = min(stride, blanks_to_create, 81 - current_pos)
        if chunk_size_desired < 15 or current_pos + chunk_size_desired < 42:
            chunk_size = chunk_size_desired
        else:
            # Find largest power-of-2 chunk size for efficient binary search.
            chunk_size = 1 << (chunk_size_desired.bit_length() - 1)

        end_pos = current_pos + chunk_size
        chunk = coords_to_try[current_pos:end_pos]

        if not chunk:
            # No more cells can be removed to meet the num_clues target
            break

        # Store original values before modifying the puzzle
        original_values = {pos: puzzle[pos] for pos in chunk}
        for r, c in chunk:
            puzzle[r, c] = 0

        count_sol_count += 1
        if count_solutions(puzzle.copy(), count_limit=2) == 1:
            # The entire chunk was safe to remove. Commit and gallop forward.
            current_pos = end_pos
            continue

        # 2. Chunk is not safe. Restore it before binary searching.
        for pos, val in original_values.items():
            puzzle[pos] = val

        # 3. Binary search within the chunk to find the safe prefix.
        low, high = 0, len(chunk) - 1
        safe_prefix_len = 0

        while low <= high:
            mid = (low + high) // 2
            # The prefix of the chunk we are testing
            prefix_chunk = chunk[: mid + 1]

            # 3a. Try removing the prefix.
            for r, c in prefix_chunk:
                puzzle[r, c] = 0

            count_sol_count += 1
            if count_solutions(puzzle.copy(), count_limit=2) == 1:
                # This prefix is safe. It's a candidate for our answer.
                safe_prefix_len = mid + 1

                # Early stop - if the other sub-chunk is len 1, we know it has key cell in it.
                if high == mid + 1:
                    # By elimination we found a key cell in the other cell of the chunk.
                    break
                # Try a larger prefix to see if we can do better.
                low = mid + 1
            else:
                # This prefix is not safe. The breaking cell is in it. Try a smaller prefix.
                high = mid - 1
                # Restore only the cells from the unsafe prefix we just tested.
                for pos in prefix_chunk:
                    puzzle[pos] = original_values[pos]
                if debug and count_sol_count > 45:
                    print(
                        f"DEBUG: Struggling, {count_sol_count} calls to count_solutions(), "
                        f"current_pos={current_pos},"
                        f"safe_prefix_len={safe_prefix_len},"
                        f" {len(keys_at)} keys at {keys_at}."
                    )

        # 4. After the search, permanently remove the longest safe prefix found.
        if safe_prefix_len > 0:
            safe_chunk = chunk[:safe_prefix_len]
            for r, c in safe_chunk:
                puzzle[r, c] = 0

        current_pos += safe_prefix_len

        # 5. We found a key cell at current_pos - essential for a unique solution.
        keys_at.append(current_pos)
        # Skip it and continue the search from the next cell.
        current_pos += 1

    current_blanks = count_blanks(puzzle)
    clues_on_board = 81 - current_blanks
    _ = debug and print(
        f"DEBUG: Failed reaching target {num_clues} clues,"
        f" Generated a puzzle with {clues_on_board} clues / {current_blanks} blanks: "
        f"{count_sol_count} calls to count_solutions(), "
        f"{len(keys_at)} keys at {keys_at}."
    )
    return puzzle


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


def test_get_unique_sudoku(debug: bool = False):
    """Test _get_unique_sudoku()."""
    print("\n\nTesting _get_unique_sudoku(), debug={debug}")
    testcases = [
        {
            "num_clues": 81 - 54,
            "sol": "783521496946837512152469873429678351531294687867315249375142968698753124214986735",
            # fmt: off
            "coords": [
                (2, 0), (4, 7), (4, 1), (8, 1), (3, 8), (5, 1), (7, 6), (1, 1), (7, 1),
                (7, 8), (8, 5), (1, 8), (6, 3), (3, 7), (1, 4), (2, 8), (5, 8), (6, 2),
                (3, 2), (6, 0), (0, 4), (4, 4), (1, 5), (8, 4), (2, 3), (4, 6), (3, 5),
                (3, 3), (4, 0), (0, 2), (6, 8), (7, 3), (0, 1), (6, 1), (2, 7), (0, 8),
                (8, 7), (0, 5), (7, 4), (5, 2), (5, 0), (1, 3), (2, 2), (0, 6), (8, 3),
                (4, 2), (7, 2), (8, 8), (0, 3), (6, 6), (3, 6), (1, 7), (1, 6), (0, 7),
                (7, 7), (0, 0), (7, 5), (5, 4), (8, 6), (2, 5), (1, 2), (4, 3), (3, 1),
                (7, 0), (6, 5), (5, 6), (6, 7), (4, 5), (3, 0), (3, 4), (2, 1), (5, 3),
                (4, 8), (2, 6), (2, 4), (5, 7), (8, 2), (8, 0), (6, 4), (1, 0), (5, 5),
            ],
            # fmt: on
            "puzzle": "700000000906000500052069800400070300001000000000305240000042960600003000004900705",
            # A rare case of very difficult generation - a lot of key cells.
            # Makes 64 calls to count_solutions(), 24 keys, generated in 0:10:12
        },
    ]
    errors = 0
    for i, testcase in enumerate(testcases):
        num_clues = testcase["num_clues"]
        solution = str_to_arr(testcase["sol"])
        expected_puzzle = str_to_arr(testcase["puzzle"])
        coords = testcase["coords"]
        print(
            f"Test case {i+1:3d} of {len(testcases):3d}: "
            f"blanks={81-num_clues}, "
            f"num_clues={num_clues}, "
            f"sol={arr_to_str(solution)}"
        )
        start_time = timer()
        puzzle = _get_unique_sudoku(solution, num_clues, coords, debug)
        end_time = timer()
        elapsed_time = end_time - start_time
        # elapsed_time = elapsed_time / num_iter
        time_str = str(timedelta(seconds=elapsed_time))
        # print_grid(sol)
        # print_grid(puzzle)
        print_grids(
            [solution, puzzle, puzzle - solution], ["Solution", "Puzzle", "Difference"]
        )
        if (puzzle == expected_puzzle).all():
            print(
                f"PASS: result match expected {arr_to_str(expected_puzzle)}, "
                f"generated in {time_str}"
            )
        else:
            errors += 1
            print(
                f"FAIL: result mismatch, "
                f"\n  - expected: {arr_to_str(expected_puzzle)}, "
                f"\n  - got     : {arr_to_str(puzzle)}, "
                f"\n  - generated in {time_str}"
            )
    print(
        "\n"
        f"{'FAIL' if errors else 'PASS'} testing _get_unique_sudoku(), "
        f"{errors} errors in {len(testcases)} test cases."
        "\n"
    )

def main():
    """Main function to run the Sudoku solver."""

    test_get_unique_sudoku(debug=True)

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
