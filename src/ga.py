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
my_rng = np.random.default_rng()
FIXED_SEED = 10

def set_seed(seed: int):
    global my_rng
    my_rng = np.random.default_rng(seed)

def fill_initial_sudoku(sudoku: SudokuCandidate):
    """Make sure that each row has 1-9 in them respecting the intial values"""
    for row in sudoku:
        initial_numbers = set(row)
        missing_numbers = [n for n in range(1, 10) if n not in initial_numbers]
        # shuffle array
        shuffled_numbers = my_rng.permutation(missing_numbers)
        # Create a boolean mask to find all positions that are 0 (True for 0, False otherwise),
        # then directly assign the shuffled_numbers into those specific slots.
        row[row == 0] = shuffled_numbers


def calculate_fitness(sudoku: SudokuCandidate) -> int:
    """Vectorized fitness calculation."""
    penalty = 0
    
    # 1. Check Columns (Transpose makes columns into rows)
    #    Iterating over .T is faster than indexing [:, c]
    for col in sudoku.T:
        penalty += (GRID_SIZE - len(np.unique(col)))

    # 2. Check Blocks 
    #    Magic reshape: (9,9) -> (3,3,3,3) -> swap axes -> (3,3,3,3) -> (9,9)
    #    This transforms 3x3 blocks into linear rows
    blocks = sudoku.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 9)
    for block in blocks:
        penalty += (GRID_SIZE - len(np.unique(block)))
        
    return penalty


def make_initial_population(
    sudoku: SudokuCandidate, population_size: int
) -> SudokuPopulation:
    population = np.tile(sudoku, (population_size, 1, 1))

    for soduku in population:
        fill_initial_sudoku(soduku)

    return population


def tournament_selection(
    population: SudokuPopulation, 
    fitness_scores: np.ndarray, 
    k: int = 3
) -> SudokuCandidate:
    
    # 1. Pick random INDICES, not the actual boards
    pop_size = len(population)
    indices = my_rng.integers(0, pop_size, size=k)
    
    # 2. Look up their pre-calculated scores
    #    We want the index with the LOWEST score
    best_idx = indices[np.argmin(fitness_scores[indices])]
    
    return population[best_idx]

def crossover(parent1: SudokuCandidate, parent2: SudokuCandidate) -> SudokuCandidate:
    # Use RNG.integers with endpoint=True to allow cutting after the 8th row
    point = my_rng.integers(1, GRID_SIZE) 
    
    # FIX: Ensure strict type consistency
    child_sudoku = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    
    child_sudoku[:point] = parent1[:point]
    child_sudoku[point:] = parent2[point:]
    return child_sudoku

def mutate(sudoku: SudokuCandidate, mutation_rate: float) -> SudokuCandidate:
    """
    Mutates the input sudoku in-place.
    Iterates through EVERY row. If a row hits the mutation_rate, 
    we swap two non-fixed numbers in that row.
    """
    for  row_idx in range(GRID_SIZE):
        if my_rng.random() < mutation_rate:

            # 1. Get indices that are strictly mutable (False in fixed_mask)
            mutable_indices = np.where(~FIXED_MASK[row_idx])[0]

            # 2. We need at least 2 numbers to swap
            if len(mutable_indices) >= 2:
                # IMPORTANT: replace=False ensures we pick two DIFFERENT indices
                i1, i2 = my_rng.choice(mutable_indices, size=2, replace=False)
                
                # 3. Swap values
                sudoku[row_idx, i1], sudoku[row_idx, i2] = sudoku[row_idx, i2], sudoku[row_idx, i1]
    return sudoku

def evolve_population(
    current_pop: SudokuPopulation, 
    fitness_scores: np.ndarray, 
    mutation_rate: float
) -> SudokuPopulation:
    
    pop_size = len(current_pop)
    next_gen = np.empty_like(current_pop)

    # 1. Elitism: Just look up the best index
    best_idx = np.argmin(fitness_scores)
    next_gen[0] = current_pop[best_idx].copy()

    # 2. Reproduction
    for i in range(1, pop_size):
        # Pass the SCORES to the tournament
        p1 = tournament_selection(current_pop, fitness_scores)
        p2 = tournament_selection(current_pop, fitness_scores)
        
        child = crossover(p1, p2)
        next_gen[i] = mutate(child, mutation_rate)

    return next_gen

def run_evolution(
    population: SudokuPopulation, generations: int, mutation_rate: float
):
    """Main loop: Evaluate -> Check Win -> Evolve"""
    for gen in range(generations):
        # 1. Calculate Fitness ONCE
        fitness_scores = np.array([calculate_fitness(sudoku) for sudoku in population])
        
        best_score = fitness_scores.min()
        print(f"Gen {gen}: Best Fitness = {best_score}")

        if best_score == 0:
            print("\nSOLVED!")
            return population[fitness_scores.argmin()]

        # 2. Pass the scores into the evolution step
        population = evolve_population(population, fitness_scores, mutation_rate)
