from __future__ import annotations
import random
import time
from ga import (
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
USE_PARALLELIZATION = "False" # "True" or "False"
GUI = "Generations" # "Sudoku" or "Generations" 

# Parameters
INTITIAL_BOARD = test_sudoku
POPULATION_SIZE = 8000
MUTATION_RATE = 0.05
ELITISM_RATE = 10 
TOURNAMENT_MEMBERS = 3
STAGNATION_LIMIT = 70
CHUNK_SIZE = 400 # how big the array of sudokus is for the fitness calculation for each worker


def run_once(seed: int):
    then = time.perf_counter()
    random.seed(seed)
    set_mask(INTITIAL_BOARD)
    population = make_initial_population(INTITIAL_BOARD, POPULATION_SIZE)
    (winning_sudoku, generation) = run_evolution(
        initial_board=INTITIAL_BOARD,
        population=population,
        mutation_rate=MUTATION_RATE,
        elitism_rate=ELITISM_RATE,
        tournament_members=TOURNAMENT_MEMBERS,
        stagnation_limit=STAGNATION_LIMIT,
        use_parallelization=USE_PARALLELIZATION == "True",
        gui_mode=GUI,
        chunk_size=CHUNK_SIZE
    )
    now = time.perf_counter()
    print(f"It took {now - then}")


if __name__ == "__main__":
    run_once(SEED)
    # set_mask(test_sudoku)
    # population =  make_initial_population(test_sudoku, 2)
    # print(calculate_population_fitness(population))

    # child = crossover(population[0], population[1])
    # print("Child")
    # debug_print(child)
    # child = mutate(child, 1)
    # print("Child mutated")
    # debug_print(child)
