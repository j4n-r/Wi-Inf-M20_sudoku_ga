from __future__ import annotations
import random
import time
from ga import (
    BLOCK_SIZE,
    GRID_SIZE,
    SudokuCandidate,
    SudokuException,
    make_initial_population,
    run_evolution,
    set_mask,
)

test_sudoku: list[list[int]] = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# test_sudoku2: list[list[int]] = [
#     [0, 7, 0, 6, 5, 8, 0, 0, 2],
#     [0, 0, 1, 0, 4, 3, 0, 0, 8],
#     [8, 0, 0, 2, 0, 0, 7, 0, 0],
#     [5, 4, 0, 0, 0, 0, 0, 0, 1],
#     [0, 2, 0, 0, 0, 5, 3, 0, 0],
#     [0, 9, 0, 0, 0, 0, 2, 0, 6],
#     [0, 0, 0, 0, 0, 6, 4, 2, 0],
#     [0, 8, 6, 0, 0, 0, 0, 0, 5],
#     [0, 5, 2, 0, 7, 0, 0, 0, 0],
# ]


###########
# Config  #
###########

# Program config
SEED = 10
USE_SEED = "True"  # "True" or "False"
USE_PARALLELIZATION = "True"  # "True" or "False"
GUI = "Generations"  # "Sudoku", "Generations", or "None"
RUNS = 1  # number of repeated runs for timing

# Parameters
INTITIAL_BOARD = test_sudoku
POPULATION_SIZE = 8000
MUTATION_RATE = 0.05
ELITISM_RATE = 10
TOURNAMENT_MEMBERS = 3
STAGNATION_LIMIT = 70
CHUNK_SIZE = (
    400  # how big the array of sudokus is for the fitness calculation for each worker
)

def validate_sudoku(sudoku: SudokuCandidate) -> None:
    """
    Raises exception
    """
    # check number of rows
    if len(sudoku) != GRID_SIZE:
        raise SudokuException(f"Sudoku does not have exactly {GRID_SIZE} rows")

    for row_idx, row in enumerate(sudoku):
        # check row length
        if len(row) != GRID_SIZE:
            raise SudokuException(f"Sudoku row {row_idx} width is not {GRID_SIZE}")

        for col_idx, value in enumerate(row):
            # check for non int values
            if not isinstance(value, int):
                raise SudokuException(
                    f"Sudoku value at ({row_idx}, {col_idx}) is not an int"
                )
            # check for > then GRID_SIZE values
            if value < 0 or value > GRID_SIZE:
                raise SudokuException(
                    f"Sudoku value at (row: {row_idx}, col: {col_idx}) is out of range"
                )

        # get all non_zero values in a row
        non_zero = [value for value in row if value != 0]
        # compare the list with a set (only unique numbers allowed) to check for duplicates
        if len(non_zero) != len(set(non_zero)):
            raise SudokuException(f"Sudoku row {row_idx} has duplicates")

    # check for column duplicates
    for col_idx in range(GRID_SIZE):
        column = [sudoku[row_idx][col_idx] for row_idx in range(GRID_SIZE)]
        non_zero = [value for value in column if value != 0]
        if len(non_zero) != len(set(non_zero)):
            raise SudokuException(f"Sudoku column {col_idx} has duplicates")

    # check for block duplicates
    for block_row in range(0, GRID_SIZE, BLOCK_SIZE):
        for block_col in range(0, GRID_SIZE, BLOCK_SIZE):
            block_vals: list[int] = []
            for row in range(block_row, block_row + BLOCK_SIZE):
                for col in range(block_col, block_col + BLOCK_SIZE):
                    block_vals.append(sudoku[row][col])
            non_zero = [value for value in block_vals if value != 0]
            if len(non_zero) != len(set(non_zero)):
                raise SudokuException(
                    f"Sudoku block at (row: {block_row}, col: {block_col}) has duplicates"
                )


def run_once(seed: int) -> float:
    then = time.perf_counter()
    if USE_SEED == "True":
        random.seed(seed)
    try:
        validate_sudoku(INTITIAL_BOARD)
    except SudokuException as e:
        print(e)
        exit(1)

    fixed_mask, row_mutable_indices = set_mask(INTITIAL_BOARD)
    population = make_initial_population(
        INTITIAL_BOARD, POPULATION_SIZE, row_mutable_indices
    )
    (winning_sudoku, generation) = run_evolution(
        initial_board=INTITIAL_BOARD,
        population=population,
        mutation_rate=MUTATION_RATE,
        elitism_rate=ELITISM_RATE,
        tournament_members=TOURNAMENT_MEMBERS,
        fixed_mask=fixed_mask,
        row_mutable_indices=row_mutable_indices,
        stagnation_limit=STAGNATION_LIMIT,
        use_parallelization=USE_PARALLELIZATION == "True",
        gui_mode=GUI,
        chunk_size=CHUNK_SIZE,
    )
    now = time.perf_counter()
    elapsed = now - then
    print(f"It took {elapsed}")
    return elapsed


if __name__ == "__main__":
    timings: list[float] = []
    for offset in range(RUNS):
        timings.append(run_once(SEED + offset))
    if RUNS > 1:
        average = sum(timings) / RUNS
        print(f"Average over {RUNS} runs: {average}")



