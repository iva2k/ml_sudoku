#!/usr/bin/env python3

import copy
import time
from timeit import default_timer as timer
import numpy as np

class Sudoku:
    '''
    '''
    def __init__(self, quiz = None, solution = None):
        self._quiz = quiz
        self._solution = solution

    @property
    def quiz(self):
        return self._quiz

    @property
    def solution(self):
        return self._solution

    def print(self, board):
        board = self.arr_to_str(board) if not isinstance(board, str) else board
        if not board or len(board) != 81:
            raise ValueError(f'Given board "{board}" is not a sudoku, should be 81 characters')
        line = "     +-------+-------+-------+"
        print(line)
        for r in range(9):
            # print "---- row"
            print(f"[{r}]: |", end='')
            for c in range(9):
                val = board[r * 9 + c]
                if val in ['0', ' ', '_', '-']:
                    val = " "
                print(' ' + val, end='')
                if (c + 1) % 3 == 0:
                    print(" |", end='')
            print()
            if (r + 1) % 3 == 0:
                print(line)

    def arr_to_str(self, board):
        s = ''
        for row in board:
            for c in row:
                s += str(c)
        return s


    def str_to_arr(self, board):
        a = [int(c) for c in board]
        return np.reshape(a, (9,9,))

    def count_blanks(self, board):
        board = self.str_to_arr(board) if isinstance(board, str) else board
        count = 0
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    count += 1
        return count

    def next_box(self, board: np.ndarray, row: int = None, col: int = None):
        """
        Finds the next empty cell (0) on the board.
        - If row/col are None, finds the very first empty cell.
        - If row/col are provided, finds the first empty cell after (row, col).
        Returns a (row, col) tuple or False if no empty cells are found.
        """
        board = self.str_to_arr(board) if isinstance(board, str) else board

        empty_cells = np.argwhere(board == 0)
        if empty_cells.size == 0:
            return False

        if row is None: # Find the first empty cell on the board
            return tuple(empty_cells[0])

        # Find the first empty cell *after* the given (row, col)
        current_pos_flat = row * 9 + col
        for r, c in empty_cells:
            if r * 9 + c > current_pos_flat:
                return (r, c)
        return False

    def possible(self, board: np.ndarray, row: int, col: int, n: int) -> bool:
        """
        Optimized check to see if placing number 'n' at (row, col) is valid.
        Assumes the board is a NumPy array and the cell at (row, col) is currently 0.
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

    def solve_brute(self, board):
        """Solves the Sudoku puzzle using backtracking."""
        board = self.str_to_arr(board) if isinstance(board, str) else board
        blank_cell = self.next_box(board)
        if blank_cell is False:
            return board
        row, col = blank_cell
        for n in range(1,10):
            if self.possible(board, row, col, n):
                board[row][col] = n
                result = self.solve_brute(board)
                if result is not False:
                    return result
            board[row][col] = 0  # Backtrack
        return False

    def count_solutions(self, board, count_limit=2):
        """Counts the number of solutions for a board up to a limit using backtracking."""
        board = self.str_to_arr(board) if isinstance(board, str) else board
        count = 0

        def _solve():
            nonlocal count
            blank_cell = self.next_box(board)
            if blank_cell is False:
                count += 1
                return count >= count_limit

            row, col = blank_cell
            for n in range(1, 10):
                if self.possible(board, row, col, n):
                    board[row][col] = n
                    if _solve():
                        return True
                    board[row][col] = 0  # Backtrack
            return False

        _solve()
        return count

    def solve_eliminator(self, board):
        board = self.str_to_arr(board) if isinstance(board, str) else board
        row, col = None, None
        while True:
            not_solved = self.next_box(board)
            blank_cell = self.next_box(board, row, col)
            if blank_cell is False:
                if not_solved is False:
                    return board
                # Reached the end of board, no elimination possible, do brute-force
                row, col = not_solved
                for n in range(1,10):
                    if self.possible(board, row, col, n):
                        brute_board = copy.copy(board)
                        brute_board[row][col] = n
                        result = self.solve_eliminator(brute_board)
                        if result is not False:
                            return result
                    # brute_board[row][col] = 0
                return False
            row, col = blank_cell
            vals = []
            for n in range(1,10):
                if self.possible(board, row, col, n):
                    vals += [n]
            if len(vals) == 1:
                board[row][col] = vals[0]
                row, col = None, None

        return False

    def solver(self, board, fnc = None):
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
        q = copy.copy(board)
        start_time = timer()
        result = fnc(q)
        end_time = timer()
        elapsed_time = end_time - start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
        if result is not False:
            print(f'Method {name} solved in {time_str}:')
            self.print(result)
        else:
            print(f'Method {name} failed to solve')


if __name__ == "__main__":
    s = Sudoku('000308600302400058005020071586000400000007002090140000403096105001280006070000030', '719358624362471958845629371586932417134867592297145863423796185951283746678514239')
    # s = Sudoku('000260701680070090190004500820100040004602900050003028009300074040050036703018000', '435269781682571493197834562826195347374682915951743628519326874248957136763418259')
    s.solver(s.quiz)
    s.solver(s.quiz, s.solve_eliminator)
    print()
    print('Known solution:')
    s.print(s.solution)
