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

    def next_box(self, board, row = None, col = None):
        board = self.str_to_arr(board) if isinstance(board, str) else board
        if row is not None and col is not None:
            if col == 8:
                if row == 8:
                    return False
                row += 1
                col = 0
            else:
                col += 1
        for r in (range(9) if row is None else range(row, 9)):
            for c in (range(9) if col is None or r > row else range(col, 9)):
                if board[r][c] == 0:
                    return (r, c)
        return False

    def possible(self, board, row, col, n):
        board = self.str_to_arr(board) if isinstance(board, str) else board
        for i in range(9):
            if board[row][i] == n and col != i:
                return False
            if board[i][col] == n and row != i:
                return False
        row0 = row // 3
        col0 = col // 3
        for i in range(3):
            for j in range(3):
                if board[row0*3 + i][col0*3 + j] == n and (row0*3 + i,col0*3 + j) != (row,col):
                    return False
        return True

    def solve_brute(self, board):
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
            board[row][col] = 0
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
