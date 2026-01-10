from __future__ import annotations
from copy import deepcopy
import random

from ui import render_sudoku


type SudokuCandidate = list[list[int]]
type SudokuPopulation = list[SudokuCandidate]

fixed_mask: list[list[bool]] = []
row_mutable_indices: list[list[int]] = []
# my_rng = random.

def debug_print(sudoku_like: list[list[Any]]):
    for row in sudoku_like:
        print(row)
    print()


def set_seed(seed: int):
    global my_rng
    pass


def set_mask(sudoku: SudokuCandidate):
    """
    Cache mutable cell positions to avoid recomputing them inside hot loops.
    """
    global fixed_mask, row_mutable_indices
    for row in sudoku:
        mask_row: list[bool] = []
        mut_row_idx: list[int] = []
        for cell_idx, num in enumerate(row):
            if num == 0:
                mask_row.append(False)
                mut_row_idx.append(cell_idx)
            else:
                mask_row.append(True)
        fixed_mask.append(mask_row)
        row_mutable_indices.append(mut_row_idx)


def make_initial_population(sudoku: SudokuCandidate, population_size: int)  -> SudokuPopulation:
    missing_numbers : list[list[int]] = []
    all_numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    for row in sudoku:
        current = set(row)
        missing = list(all_numbers - current)
        missing_numbers.append(missing)

    population: SudokuPopulation = []
    for _ in range(population_size):
        # Create a fresh copy for this individual
        child = deepcopy(sudoku)
        
        for row_idx in range(9):
            # Get the specific missing numbers for THIS row
            # We copy it because random.shuffle works in-place
            missing_vals = missing_numbers[row_idx][:] 
            random.shuffle(missing_vals)

            # Get the locations that need filling for THIS row
            target_indices = row_mutable_indices[row_idx]
            
            # Plug the shuffled numbers into the holes
            for i, col_idx in enumerate(target_indices):
                child[row_idx][col_idx] = missing_vals[i]
                
        population.append(child)
        debug_print(child)
    return population
            


def calculate_population_fitness(population: SudokuPopulation) -> list[int]:
    """
    Fitness = number of conflicts in columns and 3x3 boxes.
    Lower is better. 0 means a solved Sudoku.
    """
    fitness_scores: list[int] = []

    for sudoku in population:
        conflicts = 0

        # --- column conflicts ---
        for col in range(9):
            column_values = [sudoku[row][col] for row in range(9)]
            conflicts += 9 - len(set(column_values))

        # --- 3x3 box conflicts ---
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box_values: list[int] = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        box_values.append(sudoku[r][c])
                conflicts += 9 - len(set(box_values))

        fitness_scores.append(conflicts)
    return fitness_scores


def crossover(parent1: SudokuCandidate, parent2: SudokuCandidate) -> SudokuCandidate:
    """
    Row-based crossover.

    Pick a cut point, take the top part of rows from parent1 and the rest from parent2.
    This preserves row validity because whole rows are copied.
    """
    cut = random.randint(1, 9 - 1)  # 1..8 so both parents contribute

    child: SudokuCandidate = []
    for row_idx in range(9):
        if row_idx < cut:
            child.append(parent1[row_idx][:])  # copy row
        else:
            child.append(parent2[row_idx][:])  # copy row

    return child


def mutate(sudoku: SudokuCandidate, mutation_rate: float) -> SudokuCandidate:
    """
    For each row: with probability mutation_rate, swap two *mutable* cells in that row.
    Fixed cells (givens) are never changed.
    """
    for r in range(9):
        if random.random() >= mutation_rate:
            continue

        mutable_cols = row_mutable_indices[r]
        if len(mutable_cols) < 2:
            continue  # nothing to swap

        c1, c2 = random.sample(mutable_cols, 2)
        sudoku[r][c1], sudoku[r][c2] = sudoku[r][c2], sudoku[r][c1]

    return sudoku

def batch_tournament_winners(
    fitness_scores: list[int], selection_count: int, tournament_members: int = 3
) -> list[SudokuCandidate]:
    pass


def evolve_population(
    current_pop: SudokuPopulation,
    fitness_scores: list[int],
    mutation_rate: float,
    elitism_rate: int,
    tournament_members: int) -> SudokuPopulation:
    pass

def run_evolution(
    initial_board: SudokuCandidate,
    population: SudokuPopulation,
    mutation_rate: float,
    elitism_rate: int,
    tournament_members: int,
    stagnation_limit: int = 100,
) -> int:
    pass
