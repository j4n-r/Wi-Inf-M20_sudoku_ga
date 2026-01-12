from __future__ import annotations
from copy import deepcopy
import random
from typing import Any

from ui import render_sudoku, update_board
from concurrent.futures import ProcessPoolExecutor
import os


class SudokuException(Exception):
    pass


type SudokuCandidate = list[list[int]]
type SudokuPopulation = list[SudokuCandidate]

fixed_mask: list[list[bool]] = []
row_mutable_indices: list[list[int]] = []

GRID_SIZE = 9
BLOCK_SIZE = 3


def debug_print(sudoku_like: list[list[Any]]):
    for row in sudoku_like:
        print(row)
    print()


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


def make_initial_population(
    sudoku: SudokuCandidate, population_size: int
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


def mutate(sudoku: SudokuCandidate, mutation_rate: float) -> None:
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


def evolve_population(
    current_pop: SudokuPopulation,
    fitness_scores: list[int],
    mutation_rate: float,
    elitism_rate: int,
    tournament_members: int,
) -> SudokuPopulation:
    """
    Create the next generation:
      1) keep the best 'elitism_rate' individuals
      2) fill the rest via tournament selection -> crossover -> mutation
    """
    pop_size = len(current_pop)
    elitism_count = min(elitism_rate, pop_size)

    paired: list[tuple[int, SudokuCandidate]] = []
    for i in range(pop_size):
        paired.append((fitness_scores[i], current_pop[i]))

    # sort by fitness
    paired.sort(key=lambda pair: pair[0])

    next_population: SudokuPopulation = []
    for pair in paired[:elitism_count]:
        individual = pair[1]
        # copy sudoku
        elite = [row[:] for row in individual]
        # add to new population
        next_population.append(elite)

    # --- 2) Create the rest of the population ---
    children_needed = pop_size - elitism_count
    if children_needed <= 0:
        return next_population

    parent_pairs = get_parents_from_tournament(
        population=current_pop,
        fitness_scores=fitness_scores,
        selection_count=children_needed,
        tournament_members=tournament_members,
    )

    for parent1, parent2 in parent_pairs:
        child = crossover(parent1, parent2)
        mutate(child, mutation_rate)
        next_population.append(child)

    return next_population


def run_evolution(
    initial_board: SudokuCandidate,
    population: SudokuPopulation,
    mutation_rate: float,
    elitism_rate: int,
    tournament_members: int,
    stagnation_limit: int = 100,
    use_parallelization: bool = True,
    gui_mode: str = "Sudoku",
    chunk_size: int = 256,
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
                f"{generation}: {best_fitness_now} --- Global Generation {global_generations}"
            )
        if show_board:
            update_board(best_individual)

    if use_parallelization:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as worker_pool:
            while True:
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
                    population = make_initial_population(initial_board, population_size)
                    best_fitness_ever = 100
                    generations_without_improvement = 0
                    generation = 0
                    continue

                population = evolve_population(
                    population,
                    fitness_scores,
                    mutation_rate,
                    elitism_rate,
                    tournament_members,
                )

                generation += 1
                global_generations += 1
    else:
        while True:
            fitness_scores = calculate_fitness_population(population)
            best_fitness_now = min(fitness_scores)
            best_index = fitness_scores.index(best_fitness_now)
            best_individual = population[best_index]

            report_progress(best_fitness_now, best_individual)

            # SOLVED
            if best_fitness_now == 0:
                print(f"SOLVED in Generation: {global_generations}")
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
                population = make_initial_population(initial_board, population_size)
                best_fitness_ever = 100
                generations_without_improvement = 0
                generation = 0
                continue

            population = evolve_population(
                population,
                fitness_scores,
                mutation_rate,
                elitism_rate,
                tournament_members,
            )

            generation += 1
            global_generations += 1
