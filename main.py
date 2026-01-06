from __future__ import annotations
import numpy as np
import numpy.typing as npt

SUDOKO: npt.NDArray[np.int8]  = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ], dtype=np.int8)
ROWS = 9
COLS = 9


def fill_initial_sudoku(sudoku: npt.NDArray[np.int8]):
    for row in sudoku:
        initial_numbers = set(row)
        missing_numbers = [n for n in range(1,10) if n not in initial_numbers]
        # shuffle array
        shuffled_numbers = np.random.permutation(missing_numbers)
        # Create a boolean mask to find all positions that are 0 (True for 0, False otherwise), 
        # then directly assign the shuffled_numbers into those specific slots.
        row[row == 0] = shuffled_numbers

    print(sudoku)
def main():
    fixed_mask = SUDOKO != 0
    fill_initial_sudoku(SUDOKO)


if __name__ == "__main__":
    main()
