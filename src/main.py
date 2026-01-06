import argparse
import multiprocessing as mp
import time
import timeit
import numpy as np
import numpy.typing as npt
from ga import (
    make_initial_population,
    run_evolution,
    set_mask,
    set_seed,
)

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
# SUDOKU: npt.NDArray[np.int8] = np.array(
#     [
#         [0, 7, 0, 6, 5, 8, 0, 0, 2],
#         [0, 0, 1, 0, 4, 3, 0, 0, 8],
#         [8, 0, 0, 2, 0, 0, 7, 0, 0],
#         [5, 4, 0, 0, 0, 0, 0, 0, 1],
#         [0, 2, 0, 0, 0, 5, 3, 0, 0],
#         [0, 9, 0, 0, 0, 0, 2, 0, 6],
#         [0, 0, 0, 0, 0, 6, 4, 2, 0],
#         [0, 8, 6, 0, 0, 0, 0, 0, 5],
#         [0, 5, 2, 0, 7, 0, 0, 0, 0],
#     ],
#     dtype=np.int8,
# )


def run_once(seed: int):
    """Run one GA solve with an isolated RNG/mask per process."""
    set_seed(seed)
    set_mask(SUDOKU)
    population = make_initial_population(SUDOKU, 15)
    return run_evolution(SUDOKU, population, 10, 0.10, 10, 30)


def main():
    worker_count = mp.cpu_count()
    base_seed = 11
    seeds = [base_seed + i for i in range(worker_count)]

    # run_once(base_seed)
    start = time.perf_counter()

    pool = mp.Pool(processes=worker_count)
    try:
        # Launch one run per core; stop the pool as soon as the first completes.
        results_iter = pool.imap_unordered(run_once, seeds)
        best = next(results_iter)
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.perf_counter() - start
    print(f"First worker finished in {elapsed:.3f} seconds using {worker_count} workers.")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Sudoku GA solver.")
    parser.add_argument(
        "--timeit",
        action="store_true",
        help="Benchmark main() with timeit (spawns processes each iteration).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of runs for timeit when --timeit is set.",
    )
    args = parser.parse_args()

    if args.timeit:
        duration = timeit.timeit(main, number=args.iterations)
        print(f"Average time per run: {duration / args.iterations:.6f} seconds over {args.iterations} runs.")
    else:
        main()
# 14s
# 14.67
