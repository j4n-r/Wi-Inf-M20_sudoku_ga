import time
import timeit
from typing import cast
from ga import (
    make_initial_population,
    run_evolution,
    set_mask,
    set_seed,
)
import numpy as np
import numpy.typing as npt

# SUDOKU: npt.NDArray[np.int8] = np.array(
#     [
#         [5, 3, 0, 0, 7, 0, 0, 0, 0],
#         [6, 0, 0, 1, 9, 5, 0, 0, 0],
#         [0, 9, 8, 0, 0, 0, 0, 6, 0],
#         [8, 0, 0, 0, 6, 0, 0, 0, 3],
#         [4, 0, 0, 8, 0, 3, 0, 0, 1],
#         [7, 0, 0, 0, 2, 0, 0, 0, 6],
#         [0, 6, 0, 0, 0, 0, 2, 8, 0],
#         [0, 0, 0, 4, 1, 9, 0, 0, 5],
#         [0, 0, 0, 0, 8, 0, 0, 7, 9],
#     ],
#     dtype=np.int8,
# )
SUDOKU: npt.NDArray[np.int8] = np.array(
    [
        [0, 7, 0, 6, 5, 8, 0, 0, 2],
        [0, 0, 1, 0, 4, 3, 0, 0, 8],
        [8, 0, 0, 2, 0, 0, 7, 0, 0],
        [5, 4, 0, 0, 0, 0, 0, 0, 1],
        [0, 2, 0, 0, 0, 5, 3, 0, 0],
        [0, 9, 0, 0, 0, 0, 2, 0, 6],
        [0, 0, 0, 0, 0, 6, 4, 2, 0],
        [0, 8, 6, 0, 0, 0, 0, 0, 5],
        [0, 5, 2, 0, 7, 0, 0, 0, 0],
    ],
    dtype=np.int8,
)


def main():
    # 1. Bring the global RNG variable into scope

    # Now the rest of your logic runs with a fresh sequence

    SEED = 11
    # set_seed(SEED)
    set_mask(SUDOKU)
    population = make_initial_population(SUDOKU, 1000)
    best = run_evolution(SUDOKU, population, 200, 0.10, 1, 60)


if __name__ == "__main__":
    ITERATIONS = 10

    execution_time = timeit.timeit(main, number=ITERATIONS)
    main()

    print(f"Average time per run: {execution_time / ITERATIONS:.6f} seconds")

# 1.80 s

# 1.70

# 0.72

