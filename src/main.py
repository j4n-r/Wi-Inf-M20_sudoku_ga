import random

from ga import (
    make_initial_population,
    run_evolution,
    set_mask,
)
from ui import render_sudoku, update_board

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

test_sudoku2: list[list[int]] = [
    [0, 7, 0, 6, 5, 8, 0, 0, 2],
    [0, 0, 1, 0, 4, 3, 0, 0, 8],
    [8, 0, 0, 2, 0, 0, 7, 0, 0],
    [5, 4, 0, 0, 0, 0, 0, 0, 1],
    [0, 2, 0, 0, 0, 5, 3, 0, 0],
    [0, 9, 0, 0, 0, 0, 2, 0, 6],
    [0, 0, 0, 0, 0, 6, 4, 2, 0],
    [0, 8, 6, 0, 0, 0, 0, 0, 5],
    [0, 5, 2, 0, 7, 0, 0, 0, 0],
]


def run_once(seed: int):
    """Run one GA solve with an isolated RNG/mask per process."""
    random.seed(seed)
    set_mask(test_sudoku)
    population = make_initial_population(test_sudoku, 1000)
    (winning_sudoku, generation) = run_evolution(
        initial_board=test_sudoku,
        population=population,
        mutation_rate=0.05,
        elitism_rate=10,
        tournament_members=3,
        stagnation_limit=70,
    )


if __name__ == "__main__":
    run_once(10)
    # set_mask(test_sudoku)
    # population =  make_initial_population(test_sudoku, 2)
    # print(calculate_population_fitness(population))

    # child = crossover(population[0], population[1])
    # print("Child")
    # debug_print(child)
    # child = mutate(child, 1)
    # print("Child mutated")
    # debug_print(child)
