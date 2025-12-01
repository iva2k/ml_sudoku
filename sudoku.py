#!/usr/bin/env python3
# sudoku.py

"""Sudoku puzzle solver and utilities."""

import copy
from datetime import timedelta
from timeit import default_timer as timer
from typing import Callable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

Board = NDArray[np.int_]  # Shape: (9, 9) = np.ndarray((9, 9), dtype=int)
BoardOrStr = Union[Board, str]


class Sudoku:
    """Sudoku puzzle solver and utilities."""

    def __init__(self, quiz: Optional[BoardOrStr] = None, solution: Optional[BoardOrStr] = None):
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

    def print(self, board_str: BoardOrStr) -> None:
        """Print Sudoku board."""
        board_str = self.arr_to_str(board_str) if not isinstance(
            board_str, str) else board_str
        if not board_str or len(board_str) != 81:
            raise ValueError(
                f'Given board "{board_str}" is not a sudoku, should be 81 characters')
        line = "     +-------+-------+-------+"
        print(line)
        for r in range(9):
            # print "---- row"
            print(f"[{r}]: |", end='')
            for c in range(9):
                val = board_str[r * 9 + c]
                if val in ['0', ' ', '_', '-']:
                    val = " "
                print(' ' + val, end='')
                if (c + 1) % 3 == 0:
                    print(" |", end='')
            print()
            if (r + 1) % 3 == 0:
                print(line)

    def arr_to_str(self, board: Board) -> str:
        """Convert a 2D NumPy array to a string."""
        s = ''
        for row in board:
            for c in row:
                s += str(c)
        return s

    def str_to_arr(self, board: str) -> Board:
        """Convert a string to a 2D NumPy array."""
        a = [int(c) for c in board]
        return np.reshape(a, (9, 9,))

    def count_blanks(self, board: BoardOrStr):
        """Count the number of blank/empty cells (0s) on the board."""
        board = self.str_to_arr(board) if isinstance(board, str) else board
        count = 0
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    count += 1
        return count

    def next_box(self, board: Board, row: int = None, col: int = None):
        """
        Finds the next blank/empty cell (0) on the board.
        - If row/col are None, finds the very first empty cell.
        - If row/col are provided, finds the first empty cell after (row, col).
        Returns a (row, col) tuple or False if no empty cells are found.
        """
        board = self.str_to_arr(board) if isinstance(board, str) else board

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

    def possible(self, board: BoardOrStr, row: int, col: int, n: int) -> bool:
        """
        Optimized check to see if placing number 'n' at (row, col) is valid.
        Assumes the cell at (row, col) is currently 0.
        """
        board = self.str_to_arr(board) if isinstance(board, str) else board
        # Check if 'n' is already in the same row or column
        if np.any(board[row, :] == n) or np.any(board[:, col] == n):
            return False

        # Check 3x3 subgrid
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if np.any(board[start_row:start_row + 3, start_col:start_col + 3] == n):
            return False
        return True

    def all_possible(self, board: BoardOrStr, row: int, col: int) -> List[int]:
        """
        Returns all possible numbers that can be placed at (row, col) without violating the Sudoku rules.
        """
        board = self.str_to_arr(board) if isinstance(board, str) else board
        possibilities = [n for n in range(
            1, 10) if self.possible(board, row, col, n)]
        return possibilities

    def solve_brute(self, board: BoardOrStr) -> Optional[Board]:
        """Solves the Sudoku puzzle using backtracking."""
        board = self.str_to_arr(board) if isinstance(board, str) else board
        return self._solve_brute(board)

    def _solve_brute(self, board: Board) -> Optional[Board]:
        """Solves the Sudoku puzzle using backtracking."""
        blank_cell = self.next_box(board)
        if blank_cell is False:
            return board
        row, col = blank_cell
        for n in range(1, 10):
            if self.possible(board, row, col, n):
                board[row][col] = n
                result = self._solve_brute(board)
                if result is not False:
                    return result
            board[row][col] = 0  # Backtrack
        return False

    def count_solutions(self, board: BoardOrStr, count_limit: int = 2) -> int:
        """Counts the number of solutions for a board up to a limit using backtracking."""
        board = self.str_to_arr(board) if isinstance(board, str) else board
        return self._count_solutions(board, count_limit)

    def _count_solutions(self, board: Board, count_limit: int = 2) -> int:
        """Counts the number of solutions for a board up to a limit using backtracking."""
        count = 0

        def _solve():
            nonlocal count
            blank_cell = self.next_box(board)
            if blank_cell is False:
                count += 1
                return count >= count_limit

            row, col = blank_cell
            vals = self.all_possible(board, row, col)
            for n in vals:
                board[row][col] = n
                if _solve():
                    return True
            board[row][col] = 0  # Backtrack
            return False

        _solve()
        return count

    def solve_eliminator(self, board: BoardOrStr) -> Optional[Board]:
        """Solves the Sudoku puzzle using backtracking and eliminator heuristic."""
        board = self.str_to_arr(board) if isinstance(board, str) else board
        return self._solve_eliminator(board)

    def _solve_eliminator(self, board: Board) -> Optional[Board]:
        """Solves the Sudoku puzzle using backtracking and eliminator heuristic."""
        row, col = None, None
        while True:
            not_solved = self.next_box(board)
            if not_solved is False:
                # Solved
                return board
            next_blank_cell = self.next_box(board, row, col)
            if next_blank_cell is False:
                # Reached the end of board, no elimination possible, do brute-force
                row, col = not_solved
                vals = self.all_possible(board, row, col)
                for n in vals:
                    # brute_board = copy.copy(board)
                    brute_board = board
                    brute_board[row][col] = n
                    result = self._solve_eliminator(brute_board)
                    if result is not False:
                        return result
                    brute_board[row][col] = 0
                return False
            row, col = next_blank_cell
            vals = self.all_possible(board, row, col)
            if len(vals) == 1:
                board[row][col] = vals[0]
                row, col = None, None

        return False

    def solver(self,
        board: BoardOrStr,
        fnc: Optional[Callable[[BoardOrStr], Optional[Board]]] = None,
        num_iter: int = 1000
    ) -> None:
        """
        Runs Sudoku solver using the specified function and prints the results.
        """
        if num_iter <= 0:
            raise ValueError('num_iter must be a positive integer')
        if not fnc:
            fnc = self.solve_brute
        name = ''
        if fnc == self.solve_brute:
            name = 'solve_brute'
        elif fnc == self.solve_eliminator:
            name = 'solve_eliminator'

        print()
        print(f'Solver using method {name}')
        self.print(board)
        print(f'Blanks: {self.count_blanks(board)}')
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
            print(f'Method {name} solved in {time_str}:')
            self.print(result)
        else:
            print(f'Method {name} failed to solve')


def main():
    """Main function to run the Sudoku solver."""
    num_iter = 100
    s = Sudoku('000308600302400058005020071586000400000007002090140000403096105001280006070000030',
               '719358624362471958845629371586932417134867592297145863423796185951283746678514239')
    # s = Sudoku('000260701680070090190004500820100040004602900050003028009300074040050036703018000', '435269781682571493197834562826195347374682915951743628519326874248957136763418259')
    s.solver(s.quiz, s.solve_brute, num_iter)
    s.solver(s.quiz, s.solve_eliminator, num_iter)
    print()
    print('Known solution:')
    s.print(s.solution)

    start_time = timer()
    for _i in range(num_iter):
        count = s.count_solutions(s.quiz)
    end_time = timer()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time / num_iter
    time_str = str(timedelta(seconds=elapsed_time))
    print(f"Number of solutions for quiz: {count}, found in {time_str}")


if __name__ == "__main__":
    main()
