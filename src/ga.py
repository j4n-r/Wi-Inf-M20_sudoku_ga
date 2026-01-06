from __future__ import annotations
from typing import cast
import numpy as np
import numpy.typing as npt

type SudokuCandidate = npt.NDArray[np.int8]
type SudokuPopulation = npt.NDArray[np.int8]

SUDOKU: npt.NDArray[np.int8] = np.array(
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    dtype=np.int8,
)
GRID_SIZE = 9
FIXED_MASK = cast(npt.NDArray[np.bool_], SUDOKU != 0)


def fill_initial_sudoku(sudoku: SudokuCandidate):
    """Make sure that each row has 1-9 in them respecting the intial values"""
    for row in sudoku:
        initial_numbers = set(row)
        missing_numbers = [n for n in range(1, 10) if n not in initial_numbers]
        # shuffle array
        shuffled_numbers = np.random.permutation(missing_numbers)
        # Create a boolean mask to find all positions that are 0 (True for 0, False otherwise),
        # then directly assign the shuffled_numbers into those specific slots.
        row[row == 0] = shuffled_numbers


def calculate_fitness(sudoku: SudokuCandidate) -> int:
    """
    Calculates a score based on duplicates in columns and 3x3 blocks.
    Lower is better. 0 = Solved.
    """
    penalty = 0

    # 1. Check Columns
    # We iterate over column indices (0 to 8)
    for c in range(GRID_SIZE):
        col = sudoku[:, c]
        # A perfect column has 9 unique numbers.
        # If it has 7 unique numbers, it means 2 are missing/duplicated.
        penalty += GRID_SIZE - len(np.unique(col))

    # 2. Check 3x3 Blocks
    # We step through the grid in jumps of 3 (0, 3, 6)
    for r in range(0, GRID_SIZE, 3):
        for c in range(0, GRID_SIZE, 3):
            # Slice out the 3x3 subgrid
            block = sudoku[r : r + 3, c : c + 3]
            penalty += GRID_SIZE - len(np.unique(block))

    return penalty


def make_initial_population(
    sudoku: SudokuCandidate, population_size: int
) -> SudokuPopulation:
    population = np.tile(sudoku, (population_size, 1, 1))

    for soduku in population:
        fill_initial_sudoku(soduku)

    return population


def tournament_selection(
    population: SudokuPopulation, num_of_tournament_members: int = 3
) -> SudokuCandidate:
    """
    Selects one parent by running a tournament.
    """
    rng = np.random.default_rng()
    # Randomly select contenders from the population
    candidates = rng.choice(population, size=num_of_tournament_members, axis=0)

    # DEBUG
    # for c in candidates:
    #     print(calculate_fitness(c))

    # 2. Find the candidate with the highest fitness (closest to 0)
    winner: SudokuCandidate = min(candidates, key=lambda sudoku: calculate_fitness(sudoku))

    return winner
