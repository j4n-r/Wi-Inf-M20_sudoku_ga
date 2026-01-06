import time
import timeit
from ga import SUDOKU, calculate_fitness, make_initial_population, mutate, run_evolution, tournament_selection


def main():
    # sudoku = fill_initial_sudoku(SUDOKO)
    # penalty = calculate_fitness(sudoku)
    # winner = tournament_selection(population)
    # print(winner)
    # mutate(winner, 1)
    # print(winner)

    
    population = make_initial_population(SUDOKU, 1000)
    best = run_evolution(population, 100, 0.08)


if __name__ == "__main__":
    execution_time = timeit.timeit(main, number=10)
    # main()

    print(f"Average time per run: {execution_time / 1000:.6f} seconds")

# import multiprocessing
# import os

# def run_one_instance(run_id):
#     population = make_initial_population(SUDOKU, 1000)
#     best_individual = run_evolution(population, 100, 0.08)
#     fitness = calculate_fitness(best_individual)
#     return (fitness, best_individual)

# if __name__ == '__main__':
#     # Dynamically get core count
#     cores = os.cpu_count()
#     print(f"Detected {cores} cores. Launching {cores} parallel evolutions...")

#     with multiprocessing.Pool(processes=cores) as pool:
#         all_results = pool.map(run_one_instance, range(cores))
    
#     best_fitness, best_board = max(all_results, key=lambda x: x[0])
    
#     print(f"Best Fitness: {best_fitness}")
#     print(best_board)
