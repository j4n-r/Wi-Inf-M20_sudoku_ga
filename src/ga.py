from __future__ import annotations
from copy import deepcopy
import random
from typing import Any, Tuple

from ui import render_sudoku, update_board
from concurrent.futures import ProcessPoolExecutor
import os


class SudokuException(Exception):
    pass


type SudokuCandidate = list[list[int]]
type SudokuPopulation = list[SudokuCandidate]

GRID_SIZE = 9
BLOCK_SIZE = 3


def debug_print(sudoku_like: list[list[Any]]):
    for row in sudoku_like:
        print(row)
    print()


def make_initial_population(
    sudoku: SudokuCandidate,
    population_size: int,
    row_mutable_indices: list[list[int]],
) -> SudokuPopulation:
    missing_numbers: list[list[int]] = []
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
    return population


def calculate_fitness(grid: SudokuCandidate) -> int:
    # same logic you already used inside calculate_population_fitness
    conflicts = 0

    # column conflicts
    for col in range(GRID_SIZE):
        col_vals = [grid[r][col] for r in range(GRID_SIZE)]
        conflicts += GRID_SIZE - len(set(col_vals))

    # block conflicts
    for block_row in range(0, GRID_SIZE, BLOCK_SIZE):
        for block_col in range(0, GRID_SIZE, BLOCK_SIZE):
            box = []
            for r in range(block_row, block_row + BLOCK_SIZE):
                for c in range(block_col, block_col + BLOCK_SIZE):
                    box.append(grid[r][c])
            conflicts += GRID_SIZE - len(set(box))

    return conflicts


def calculate_fitness_parallel(
    population: SudokuPopulation, worker_pool: ProcessPoolExecutor, chunk_size: int
) -> list[int]:
    # split the population into chunks of chunk size
    # make each worker call calculate_fitness(sudoku) for each sudoku in his chunk
    # convert the returned iterator to a list
    return list(worker_pool.map(calculate_fitness, population, chunksize=chunk_size))


def calculate_fitness_population(population: SudokuPopulation) -> list[int]:
    """
    Fitness = number of conflicts in columns and 3x3 boxes.
    Lower is better. 0 means a solved Sudoku.
    """
    fitness_scores: list[int] = []

    for sudoku in population:
        score = calculate_fitness(sudoku)
        fitness_scores.append(score)
    return fitness_scores


def crossover(parent1: SudokuCandidate, parent2: SudokuCandidate) -> SudokuCandidate:
    """
    Row-based crossover.

    Pick a cut point, take the top part of rows from parent1 and the rest from parent2.
    This preserves row validity because whole rows are copied.
    """
    cut = random.randint(1, GRID_SIZE - 1)  # 1..8 so both parents contribute

    child: SudokuCandidate = []
    for row_idx in range(GRID_SIZE):
        if row_idx < cut:
            child.append(parent1[row_idx][:])  # copy row
        else:
            child.append(parent2[row_idx][:])  # copy row

    return child


def mutate(
    sudoku: SudokuCandidate,
    mutation_rate: float,
    row_mutable_indices: list[list[int]],
) -> None:
    """
    For each row: with probability mutation_rate, swap two *mutable* cells in that row.
    Fixed cells (givens) are never changed.
    """
    for r in range(GRID_SIZE):
        if random.random() >= mutation_rate:
            continue

        mutable_cols = row_mutable_indices[r]
        if len(mutable_cols) < 2:
            continue  # nothing to swap

        c1, c2 = random.sample(mutable_cols, 2)
        sudoku[r][c1], sudoku[r][c2] = sudoku[r][c2], sudoku[r][c1]


def get_parents_from_tournament(
    population: SudokuPopulation,
    fitness_scores: list[int],
    selection_count: int,
    tournament_members: int = 3,
) -> list[list[SudokuCandidate]]:
    """
    Select parent pairs using tournament selection.
    Lower fitness score = better individual.
    """
    parent_pairs: list[list[SudokuCandidate]] = []
    population_size = len(population)

    for _ in range(selection_count):
        parents: list[SudokuCandidate] = []

        for _ in range(2):  # select two parents
            # pick random competitors
            competitors = random.sample(range(population_size), tournament_members)

            # choose the best among them
            best_index = competitors[0]
            for idx in competitors[1:]:
                if fitness_scores[idx] < fitness_scores[best_index]:
                    best_index = idx

            parents.append(population[best_index])

        parent_pairs.append(parents)

    return parent_pairs


def get_elites(
    population: SudokuPopulation, fitness_scores: list[int], elitism_rate: float
) -> tuple[list[SudokuCandidate], int]:
    pop_size = len(population)
    elite_count = int(elitism_rate * pop_size)

    paired: list[tuple[int, SudokuCandidate]] = []
    for i in range(pop_size):
        paired.append((fitness_scores[i], population[i]))

    # sort by fitness
    paired.sort(key=lambda pair: pair[0])

    elites: list[SudokuCandidate] = []
    for pair in paired[:elite_count]:
        individual = pair[1]
        # copy sudoku
        elite = [row[:] for row in individual]
        # add to new population
        elites.append(elite)
    return elites, elite_count


def make_children(
    population: SudokuPopulation,
    fitness_scores: list[int],
    tournament_size: int,
    children_needed: int,
    mutation_rate: float,
    row_mutable_indices: list[list[int]],
) -> list[SudokuCandidate]:
    children: list[SudokuCandidate] = []
    if children_needed <= 0:
        return children

    parent_pairs = get_parents_from_tournament(
        population=population,
        fitness_scores=fitness_scores,
        selection_count=children_needed,
        tournament_members=tournament_size,
    )

    for parent1, parent2 in parent_pairs:
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate, row_mutable_indices)
        children.append(child)
    return children


def evolve_population(
    current_population: SudokuPopulation,
    fitness_scores: list[int],
    mutation_rate: float,
    elitism_rate: float,
    tournament_size: int,
    row_mutable_indices: list[list[int]],
    seed: int,
) -> SudokuPopulation:
    """
    Create the next generation:
      1) keep the best elitism_rate% individuals
      2) fill the rest via tournament selection -> crossover -> mutation
    """
    next_population: SudokuPopulation = []
    random.seed(seed)
    pop_size = len(current_population)
    elites, elitism_count = get_elites(current_population, fitness_scores, elitism_rate)

    next_population.extend(elites)

    children_needed = pop_size - elitism_count
    children = make_children(
        current_population,
        fitness_scores,
        tournament_size,
        children_needed,
        mutation_rate,
        row_mutable_indices,
    )
    next_population.extend(children)

    return next_population


def make_children_wrapper(args: tuple[Any, ...]) -> SudokuPopulation:
    """
    Wrapper, since we have to pass the args as tuple to the workers       |
                                                                          V
        Here: next_chunks = list(worker_pool.map(make_children_wrapper, args))
    """
    (
        population,
        fitness_scores,
        tournament_members,
        children_needed,
        mutation_rate,
        row_mutable,
        seed,
    ) = args
    random.seed(seed)
    return make_children(
        population=population,
        fitness_scores=fitness_scores,
        tournament_size=tournament_members,
        children_needed=children_needed,
        mutation_rate=mutation_rate,
        row_mutable_indices=row_mutable,
    )


def split_into_chunks(items: list[Any], chunk_count: int) -> list[list[Any]]:
    """
    Split a list into chunk_count chunks
    """
    if chunk_count <= 1 or len(items) <= 1:
        return [items]
    chunk_count = min(chunk_count, len(items))
    base = len(items) // chunk_count
    extra = len(items) % chunk_count
    chunks: list[list[Any]] = []
    start = 0
    for i in range(chunk_count):
        size = base + (1 if i < extra else 0)
        end = start + size
        chunks.append(items[start:end])
        start = end
    return chunks


def balanced_split(total: int, chunk_count: int) -> list[int]:
    """
    Distributes a total value into a fixed number of chunks as evenly as possible.
    Example: 10 into 3 chunks becomes [4, 3, 3].
    """
    if total <= 0:
        return []
    chunk_count = min(chunk_count, total)
    base = total // chunk_count
    extra = total % chunk_count
    return [base + (1 if i < extra else 0) for i in range(chunk_count)]


def evolve_population_parallel(
    current_population: SudokuPopulation,
    fitness_scores: list[int],
    mutation_rate: float,
    elitism_rate: float,
    tournament_members: int,
    row_mutable_indices: list[list[int]],
    worker_pool: ProcessPoolExecutor,
    worker_count: int,
    seed: int,
) -> SudokuPopulation:
    pop_size = len(current_population)
    # get the elites from the whole population
    elites, elitism_count = get_elites(current_population, fitness_scores, elitism_rate)
    next_population: SudokuPopulation = []
    # put them directly into the next one
    next_population.extend(elites)

    children_needed = pop_size - elitism_count
    if children_needed <= 0:
        return next_population

    population_chunks = split_into_chunks(current_population, worker_count)
    fitness_chunks = split_into_chunks(fitness_scores, worker_count)
    # how many children each worker should make
    children_per_worker = balanced_split(children_needed, len(population_chunks))
    # new seeds for each subprocess
    worker_seeds = [seed + i for i in range(len(population_chunks))]
    args = []
    # make a seperate argument list for each chunk
    for i in range(len(population_chunks)):
        if children_per_worker[i] <= 0:
            continue
        args.append(
            (
                population_chunks[i],
                fitness_chunks[i],
                tournament_members,
                children_per_worker[i],
                mutation_rate,
                row_mutable_indices,
                worker_seeds[i],
            )
        )
    # make childrens for next population with workers
    # the parents can only be from the subpopulation
    next_chunks = list(worker_pool.map(make_children_wrapper, args))
    # add the children to the next_population
    for chunk in next_chunks:
        next_population.extend(chunk)

    return next_population


def run_evolution(
    initial_board: SudokuCandidate,
    population: SudokuPopulation,
    mutation_rate: float,
    elitism_rate: float,
    tournament_members: int,
    row_mutable_indices: list[list[int]],
    seed: int,
    stagnation_limit: int = 100,
    use_parallelization: bool = True,
    gui_mode: str = "Sudoku",
    chunk_size: int = 256,
    population_shuffle_interval: int = 20,
):
    population_size: int = len(population)

    best_fitness_ever: int = 100
    generations_without_improvement: int = 0
    generation: int = 0
    global_generations: int = 0

    show_board = gui_mode == "Sudoku"
    show_generations = gui_mode == "Generations"

    if show_board:
        render_sudoku(initial_board)

    def report_progress(
        best_fitness_now: int, best_individual: SudokuCandidate
    ) -> None:
        if generation % 10 != 0:
            return
        if show_generations:
            print(
                "Best fitness at generation "
                + f"{generation}: {best_fitness_now} --- Global Generation {global_generations}"
            )
        if show_board:
            update_board(best_individual)

    worker_pool = None
    worker_count = 1
    # make worker pool if use_parallelization is True
    if use_parallelization:
        worker_count = os.cpu_count() or 1
        worker_pool = ProcessPoolExecutor(max_workers=worker_count)

    #############
    # Main Loop #
    #############
    while True:
        if worker_pool is None:
            fitness_scores = calculate_fitness_population(population)
        else:
            fitness_scores = calculate_fitness_parallel(
                population, worker_pool, chunk_size
            )
        best_fitness_now = min(fitness_scores)
        best_index = fitness_scores.index(best_fitness_now)
        best_individual = population[best_index]

        report_progress(best_fitness_now, best_individual)

        # SOLVED
        if best_fitness_now == 0:
            print(f"SOLVED in Generation: {global_generations}")
            # we could remove this, the OS will kill it anyways
            if worker_pool is not None:
                worker_pool.shutdown()
            return best_individual, generation

        # Track if we improved
        if best_fitness_now < best_fitness_ever:
            best_fitness_ever = best_fitness_now
            generations_without_improvement = 0
        else:
            # if not track how long we are already on the same fitness level
            generations_without_improvement += 1

        # if the stagnation limit is reached, reset everything and try agin
        if generations_without_improvement >= stagnation_limit:
            population = make_initial_population(
                initial_board, population_size, row_mutable_indices
            )
            best_fitness_ever = 100
            generations_without_improvement = 0
            generation = 0
        else:
            if worker_pool is None:
                population = evolve_population(
                    population,
                    fitness_scores,
                    mutation_rate,
                    elitism_rate,
                    tournament_members,
                    row_mutable_indices,
                    seed,
                )
            else:
                population = evolve_population_parallel(
                    population,
                    fitness_scores,
                    mutation_rate,
                    elitism_rate,
                    tournament_members,
                    row_mutable_indices,
                    worker_pool,
                    worker_count,
                    seed,
                )

        if global_generations % population_shuffle_interval == 0:
            # shuffle the population, so that next time the childrens in one subpopulation
            # do not have the same parents
            random.shuffle(population)

        generation += 1
        global_generations += 1
