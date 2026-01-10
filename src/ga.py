from __future__ import annotations
from copy import deepcopy
import random

from ui import render_sudoku


type SudokuCandidate = list[list[int]]
type SudokuPopulation = list[SudokuCandidate]

GRID_SIZE = 9
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
        
        for row_idx in range(GRID_SIZE):
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
    pass


def crossover(parent1: SudokuCandidate, parent2: SudokuCandidate) -> SudokuCandidate:
    pass


def mutate(sudoku: SudokuCandidate, mutation_rate: float):
    """
    Iterates through EVERY row. If a row hits the mutation_rate,
    we swap two non-fixed numbers in that row.
    """
    pass

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
