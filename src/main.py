import argparse
import csv
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

test_sudoku: npt.NDArray[np.int8] = np.array(
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
    set_mask(test_sudoku)
    population = make_initial_population(test_sudoku, 800)
    return run_evolution(
        initial_board=test_sudoku,
        population=population,
        mutation_rate=0.12,
        elitism_rate=10,
        tournament_members=3,
        stagnation_limit=30,
    )


def iter_puzzles_from_csv(path: str):
    """Yield (quiz, solution) strings from a Sudoku CSV with quizzes/solutions columns."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["quizzes"], row["solutions"]


def str_to_board(s: str) -> npt.NDArray[np.int8]:
    """Convert 81-char digit string (0 for blanks) into 9x9 int8 board."""
    return np.fromiter(
        (ord(ch) - 48 for ch in s.strip()), dtype=np.int8, count=81
    ).reshape(9, 9)


def solve_current_board():
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
    print(
        f"First worker finished in {elapsed:.3f} seconds using {worker_count} workers."
    )
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Sudoku GA solver.")
    _ = parser.add_argument(
        "--csv",
        type=str,
        help="Path to sudoku.csv (quizzes,solutions). Uses puzzles from the file instead of the hardcoded one.",
    )
    _ = parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of puzzles to take from the CSV (only when --csv is set).",
    )
    _ = parser.add_argument(
        "--timeit",
        action="store_true",
        help="Benchmark solve_current_board() with timeit (spawns processes each iteration).",
    )
    _ = parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of runs for timeit when --timeit is set.",
    )
    args = parser.parse_args()

    if args.csv:
        if args.timeit:
            print("--timeit is ignored when --csv is set; running each puzzle once.")
        puzzles = []
        for i, (quiz, solution) in enumerate(iter_puzzles_from_csv(args.csv)):
            if i >= args.count:
                break
            puzzles.append((quiz, solution))
        if not puzzles:
            raise SystemExit(f"No rows found in {args.csv}")

        for idx, (quiz, solution) in enumerate(puzzles, start=1):
            test_sudoku = str_to_board(quiz)
            print(f"Puzzle {idx}/{len(puzzles)} from {args.csv}")
            _ = solve_current_board()
    else:
        if args.timeit:
            duration = timeit.timeit(solve_current_board, number=args.iterations)
            print(
                f"Average time per run: {duration / args.iterations:.6f} seconds over {args.iterations} runs."
            )
        else:
            _ = solve_current_board()
# 14s
# 14.67
