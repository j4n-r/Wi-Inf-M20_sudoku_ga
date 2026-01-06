from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import cast

type SudokuCandidate = npt.NDArray[np.int8]
type SudokuPopulation = npt.NDArray[np.int8]

GRID_SIZE = 9
fixed_mask = []
my_rng = np.random.default_rng()

def set_seed(seed: int):
    global my_rng
    my_rng = np.random.default_rng(seed)

def set_mask(sudoku):
    global fixed_mask
    fixed_mask = cast(npt.NDArray[np.bool_], sudoku == 0)

# TODO no how idea how this works
def calculate_population_fitness(population: SudokuPopulation) -> np.ndarray:
    N = population.shape[0]

    # 1. Create Views
    # Columns: (N, 9, 9) - Transpose row/col axes
    cols = population.transpose(0, 2, 1)
    
    # Blocks: (N, 9, 9) - The standard 5D reshape/swap trick
    # Reshapes to (N, 3x3 grid of 3x3 blocks), then aligns them
    blocks = population.reshape(N, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(N, 9, 9)

    # 2. Concatenate into one array of shape (N, 18, 9)
    # We now have 18 "rows" (9 cols + 9 blocks) to check per board
    all_sets = np.concatenate([cols, blocks], axis=1)

    # 3. Calculate Fitness in one go
    # Sort every set of 9 numbers independently
    sorted_sets = np.sort(all_sets, axis=2)
    
    # Check for duplicates: where difference between neighbors is 0
    # Sum across the sets (axis 1) and numbers (axis 2) to get one score per board
    penalties = (np.diff(sorted_sets, axis=2) == 0).sum(axis=(1, 2))

    return penalties.astype(int)


def make_initial_population(sudoku: SudokuCandidate, population_size: int):
    # make N copys of sudoku where N is population_size
    population = np.repeat(sudoku[None, ...], population_size, axis=0) 

    # iterate for each row in the sudoku
    for i in range(GRID_SIZE):
        # Look at the original board to find what's missing in this row
        initial_numbers = set(sudoku[i])
        # Only take numbers that are not in intial numbers
        missing_numbers = [n for n in range(1, 10) if n not in initial_numbers]
        
        # Create a matrix of missing numbers for the whole population
        values_block = np.tile(missing_numbers, (population_size, 1))
        
        # Shuffle every row of this block independently
        # 'axis=1' shuffles row-wise
        shuffled_block = my_rng.permuted(values_block, axis=1)
        
        # 5. Fill into the population
        # row_template == 0 gives us the boolean mask for the empty slots
        # We write to: All Boards (:), Current Row (i), Empty Columns (mask)
        population[:, i, sudoku[i] == 0] = shuffled_block
        
    return population

def tournament_selection(
    population: SudokuPopulation, fitness_scores: np.ndarray, tournament_members: int = 3
) -> SudokuCandidate:
    # pick random indice out of the population size (one for each tournament_member)
    indices = my_rng.integers(0, len(population), size=tournament_members)

    # Get the one with the lowest score
    best_idx = indices[np.argmin(fitness_scores[indices])]

    return population[best_idx]


def crossover(parent1: SudokuCandidate, parent2: SudokuCandidate) -> SudokuCandidate:
    # Get a random row to split the sudoku
    point = my_rng.integers(1, GRID_SIZE)

    # make a new sudoku with all zeros
    child_sudoku = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

    # put the rows from parent1 and parent2 togeather
    child_sudoku[:point] = parent1[:point]
    child_sudoku[point:] = parent2[point:]
    return child_sudoku


def mutate(sudoku: SudokuCandidate, mutation_rate: float) -> SudokuCandidate:
    """
    Iterates through EVERY row. If a row hits the mutation_rate,
    we swap two non-fixed numbers in that row.
    """
    for row_idx in range(GRID_SIZE):
        if my_rng.random() < mutation_rate:
            # Get indices that were not in the intial sudoku 
            mutable_indices = np.where(fixed_mask[row_idx])[0]

            # NOTE we assume that each row has at least 2 non intial values

            # replace=False ensures we pick two different indices
            i1, i2 = my_rng.choice(mutable_indices, size=2, replace=False)

            # Swap values
            sudoku[row_idx, i1], sudoku[row_idx, i2] = (
                sudoku[row_idx, i2],
                sudoku[row_idx, i1],
            )
    return sudoku


def evolve_population(
        current_pop: SudokuPopulation, fitness_scores: np.ndarray, mutation_rate: float, elitism_rate: int
) -> SudokuPopulation:
    pop_size = len(current_pop)
    next_gen = np.empty_like(current_pop)

    # --- Elitism ---
    # Get indices of fitness scores sorted from lowest (best) to highest (worst)
    sorted_indices = np.argsort(fitness_scores)
    
    # Select the top N indices
    elite_indices = sorted_indices[:elitism_rate]
    
    # Copy the elite boards into the start of next_gen
    next_gen[:elitism_rate] = current_pop[elite_indices].copy()

    # --- Reproduction ---
    # Start the loop from 'elitism_rate' so we don't overwrite the elites
    for i in range(elitism_rate, pop_size):
        
        # Pick parents 
        p1 = tournament_selection(current_pop, fitness_scores)
        p2 = tournament_selection(current_pop, fitness_scores)

        # Make child
        child = crossover(p1, p2)

        # mutate child
        mutate(child, mutation_rate)

        next_gen[i] = child

    return next_gen


def run_evolution(
    initial_board: SudokuCandidate, 
    population: SudokuPopulation, 
    generations: int, 
    mutation_rate: float,
    elitism_rate:int,
    stagnation_limit: int = 100
):
    # Track the best score seen so far to detect stagnation
    last_best_score = float('inf')
    stagnation_counter = 0
    
    population_size = population.shape[0]

    for gen in range(generations):
        # 1. Calculate Fitness
        fitness_scores = calculate_population_fitness(population)
        
        # 2. Find Best Score
        best_idx = fitness_scores.argmin()
        best_score = fitness_scores[best_idx]

        # 3. Check for Solution
        if best_score == 0:
            print(f"Gen {gen}: SOLVED!")
            return population[best_idx]

        # 4. Logging
        if gen % 10 == 0:
            print(f"Gen {gen}: Best Fitness = {best_score} (Stagnation: {stagnation_counter}/{stagnation_limit})")

        # 5. Stagnation Logic
        if best_score < last_best_score:
            # We found a better score! Reset the counter.
            last_best_score = best_score
            stagnation_counter = 0
        else:
            # No improvement (or worse). Increment counter.
            stagnation_counter += 1

        # 6. Trigger Restart if Stuck
        if stagnation_counter >= stagnation_limit:
            print(f"--> STUCK at score {best_score} for {stagnation_limit} gens. RESTARTING population...")
            
            # WIPE EVERYTHING: Create a fresh random population from the initial board
            population = make_initial_population(initial_board, population_size)
            
            # Reset tracking variables
            stagnation_counter = 0
            last_best_score = float('inf')
            
            # Skip the 'evolve' step this turn since we just made new ones
            continue

        # 7. Evolve (Selection -> Crossover -> Mutation)
        population = evolve_population(population, fitness_scores, mutation_rate, elitism_rate)

    return population[fitness_scores.argmin()]
