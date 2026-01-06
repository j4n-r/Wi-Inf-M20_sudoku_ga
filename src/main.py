import time
import timeit
from ga import FIXED_SEED, SUDOKU, calculate_fitness, make_initial_population, mutate, run_evolution, set_seed, tournament_selection


def main():
    # 1. Bring the global RNG variable into scope

    # Now the rest of your logic runs with a fresh sequence
    set_seed(FIXED_SEED)
    population = make_initial_population(SUDOKU, 1000)
    best = run_evolution(population, 100, 0.08)

if __name__ == "__main__":
    ITERATIONS = 10
    execution_time = timeit.timeit(main, number=ITERATIONS)
    # main()

    print(f"Average time per run: {execution_time / ITERATIONS:.6f} seconds")

# 1.80 s

# 1.70



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
