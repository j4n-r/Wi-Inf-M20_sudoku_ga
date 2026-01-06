from __future__ import annotations
import numpy as np
import numpy.typing as npt

type SudokuCandidate = npt.NDArray[np.int8]
type SudokuPopulation = npt.NDArray[np.int8]

GRID_SIZE = 9
fixed_mask: npt.NDArray[np.bool_] = np.empty((0, 0), dtype=np.bool_)
row_mutable_indices: list[npt.NDArray[np.intp]] = []
my_rng = np.random.default_rng()


def set_seed(seed: int):
    global my_rng
    my_rng = np.random.default_rng(seed)


def set_mask(sudoku):
    """
    Cache mutable cell positions to avoid recomputing them inside hot loops.
    """
    global fixed_mask, row_mutable_indices
    # marks mutable values
    fixed_mask = sudoku == 0
    # get all the indices of all mutable values
    row_mutable_indices = [np.nonzero(fixed_mask[i])[0] for i in range(GRID_SIZE)]

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
    # get all rows where we should mutate
    mutate_rows = my_rng.random(GRID_SIZE) < mutation_rate
    # iterate over the nonzero rows (where the mutation rate was higher than the random number)
    for row_idx in np.nonzero(mutate_rows)[0]:
        # check if at least 2 values are mutable in this row
        mutable_indices = row_mutable_indices[row_idx]
        if mutable_indices.size < 2:
            continue

        # get two random indices out of the mutable ones 
        i1, i2 = my_rng.choice(mutable_indices, size=2, replace=False)
        # swap the values in the row
        sudoku[row_idx, i1], sudoku[row_idx, i2] = sudoku[row_idx, i2], sudoku[row_idx, i1]
    return sudoku


def batch_tournament_winners(fitness_scores: np.ndarray, selection_count: int, tournament_members: int = 3) -> np.ndarray:
    """
    Vectorized tournament selection: pick winners for all children in one go.
    """
    # get the population size from the fitness_scores since they have the same shape
    population_size = fitness_scores.shape[0]
    # select K times N random tournament candidates where N is tournament_members and K is selection count
    # e.g. selection_count=2 tournament_members=3 ==> array([2,5,7],[3,8,10])
    candidates = my_rng.integers(0, population_size, size=(selection_count, tournament_members))
    # get the fitness scores of the candidates, has the same shape as candidates but holds the fitness_scores
    candidate_scores = fitness_scores[candidates]
    # the the index for each candidate_scores subarray for the members with the lowest score
    # e.g. candidate_scores = array([2,5,7],[3,8,10]) ===> winner_offset =  array([1,2,1,2])
    winner_offsets = candidate_scores.argmin(axis=1)
    # gets the index of all the winners into population
    winners = np.take_along_axis(candidates, winner_offsets[:, None], axis=1).reshape(-1)
    return winners


def evolve_population(
    current_pop: SudokuPopulation,
    fitness_scores: np.ndarray,
    mutation_rate: float,
    elitism_rate: int,
) -> SudokuPopulation:
    population_size = len(current_pop)
    next_gen = np.empty_like(current_pop)
    offspring_count = population_size - elitism_rate

    # --- Elitism ---
    # Get indices of fitness scores sorted from lowest (best) to highest (worst)
    sorted_indices = np.argsort(fitness_scores)
    
    # Select the top N indices
    elite_indices = sorted_indices[:elitism_rate]
    
    # Copy the elite boards into the start of next_gen
    next_gen[:elitism_rate] = current_pop[elite_indices]

    # --- Reproduction (batched selections) ---
    parent_indices_1 = batch_tournament_winners(fitness_scores, offspring_count)
    parent_indices_2 = batch_tournament_winners(fitness_scores, offspring_count)
    crossover_points = my_rng.integers(1, GRID_SIZE, size=offspring_count)

    # iterate over all three arrays at the same time
    for offset, (p1_idx, p2_idx, point) in enumerate(zip(parent_indices_1, parent_indices_2, crossover_points)):
        # get both the parents out f parent_indices_1 and parent_indices_2 and make a child
        child = crossover(current_pop[p1_idx], current_pop[p2_idx])
        # mutate child
        mutate(child, mutation_rate)
        # add the child to the new generation
        next_gen[elitism_rate + offset] = child

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
        # Calculate Fitness for the whole generation
        fitness_scores = calculate_population_fitness(population)
        
        # Find Best Score
        best_idx = fitness_scores.argmin()
        best_score = fitness_scores[best_idx]

        #  Check for Solution
        if best_score == 0:
            print(f"Gen {gen}: SOLVED!")
            return population[best_idx]

        #  Logging
        if gen % 10 == 0:
            pass
            # print(f"Gen {gen}: Best Fitness = {best_score} (Stagnation: {stagnation_counter}/{stagnation_limit})")

        # Stagnation Logic
        if best_score < last_best_score:
            # We found a better score! Reset the counter.
            last_best_score = best_score
            stagnation_counter = 0
        else:
            # No improvement (or worse). Increment counter.
            stagnation_counter += 1

        # Trigger Restart if Stuck
        if stagnation_counter >= stagnation_limit:
            # print(f"--> STUCK at score {best_score} for {stagnation_limit} gens. RESTARTING population...")
            
            # WIPE EVERYTHING: Create a fresh random population from the initial board
            population = make_initial_population(initial_board, population_size)
            
            # Reset tracking variables
            stagnation_counter = 0
            last_best_score = float('inf')
            
            # Skip the 'evolve' step this turn since we just made new ones
            continue

        # Evolve (Selection -> Crossover -> Mutation)
        population = evolve_population(population, fitness_scores, mutation_rate, elitism_rate)

    return population[fitness_scores.argmin()]
