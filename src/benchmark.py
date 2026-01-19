from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ga import (
    calculate_fitness_parallel,
    calculate_fitness_population,
    evolve_population,
    evolve_population_parallel,
    make_initial_population,
)
from main import calculate_mutable_indices, test_sudoku as SUDOKU, validate_sudoku

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "suoko-altair-benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "benchmark.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)

POPULATION_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
STAGNATION_LIMIT = 70
MUTATION_RATE = 0.1
ELITISM_RATE = 0.05
TOURNAMENT_SIZE = 3
CHUNK = 0.01
RUNS = 100
BASE_SEED = 51


@dataclass
class RunResult:
    population: int
    mode: str
    run: int
    seed: int
    runtime_s: float | None
    generations: int | None
    ms_per_gen: float | None
    worker_count: int
    status: str
    error: str | None


class BenchmarkError(RuntimeError):
    pass


def run_ga(
    pop_size: int,
    seed: int,
    parallel: bool,
    run_idx: int,
    total_runs: int,
    fitness_log: list[dict],
) -> tuple[float, int, int]:
    validate_sudoku(SUDOKU)
    mode = "Parallel" if parallel else "Single"
    logging.info(
        "Starting %s run %s/%s for population %s",
        mode,
        run_idx + 1,
        total_runs,
        pop_size,
    )
    row_mutable = calculate_mutable_indices(SUDOKU)
    random.seed(seed)
    population = make_initial_population(SUDOKU, pop_size, row_mutable)

    worker_pool = None
    worker_count = 1
    if parallel:
        max_workers = os.cpu_count() or 1
        chunk_size = max(1, int(pop_size * CHUNK))
        worker_count = max(1, min(max_workers, pop_size // chunk_size))
        if worker_count < 2:
            raise BenchmarkError("Parallel mode requires >=2 workers")
        worker_pool = ProcessPoolExecutor(max_workers=worker_count)

    best_fitness_ever = 100
    generations_without_improvement = 0
    generation = 0
    global_generations = 0
    restart_count = 0
    start = time.perf_counter()

    try:
        while True:
            if worker_pool is None:
                fitness_scores = calculate_fitness_population(population)
            else:
                chunk_size = max(1, pop_size // worker_count)
                fitness_scores = calculate_fitness_parallel(
                    population, worker_pool, chunk_size
                )

            best_fitness = min(fitness_scores)
            if best_fitness == 0:
                runtime_s = time.perf_counter() - start
                logging.info(
                    "%s run %s/%s pop %s solved at gen %s in %.2fs",
                    mode,
                    run_idx + 1,
                    total_runs,
                    pop_size,
                    global_generations,
                    runtime_s,
                )
                return runtime_s, global_generations, worker_count

            if global_generations and global_generations % 20 == 0:
                fitness_log.append(
                    {
                        "Population": pop_size,
                        "Mode": mode,
                        "Run": run_idx,
                        "Generation": global_generations,
                        "Best Fitness": best_fitness,
                    }
                )
                logging.info(
                    "%s run %s/%s pop %s: gen %s best %s",
                    mode,
                    run_idx + 1,
                    total_runs,
                    pop_size,
                    global_generations,
                    best_fitness,
                )

            if best_fitness < best_fitness_ever:
                best_fitness_ever = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= STAGNATION_LIMIT:
                restart_count += 1
                generations_without_improvement = 0
                best_fitness_ever = 100
                generation = 0
                restart_seed = seed + restart_count * 1_000_003
                random.seed(restart_seed)
                population = make_initial_population(SUDOKU, pop_size, row_mutable)
            else:
                generation_seed = seed + restart_count * 1_000_003 + generation
                if worker_pool is None:
                    population = evolve_population(
                        population,
                        fitness_scores,
                        MUTATION_RATE,
                        ELITISM_RATE,
                        TOURNAMENT_SIZE,
                        row_mutable,
                        generation_seed,
                    )
                else:
                    population = evolve_population_parallel(
                        population,
                        fitness_scores,
                        MUTATION_RATE,
                        ELITISM_RATE,
                        TOURNAMENT_SIZE,
                        row_mutable,
                        worker_pool,
                        worker_count,
                        generation_seed,
                    )

            generation += 1
            global_generations += 1
    finally:
        if worker_pool is not None:
            worker_pool.shutdown(cancel_futures=True)


def save_plot(summary: pd.DataFrame, y: str, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 4))
    order = sorted(summary["Population"].unique())
    sns.barplot(data=summary, x="Population", y=y, hue="Mode", order=order)
    plt.xlabel("Population")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


if __name__ == "__main__":
    results: list[RunResult] = []
    fitness_log: list[dict] = []

    for pop_size in POPULATION_SIZES:
        logging.info("Benchmarking population size %s", pop_size)
        for run in range(RUNS):
            seed = BASE_SEED + run
            for parallel, mode in [(False, "Single"), (True, "Parallel")]:
                try:
                    runtime_s, generations, workers = run_ga(
                        pop_size,
                        seed,
                        parallel=parallel,
                        run_idx=run,
                        total_runs=RUNS,
                        fitness_log=fitness_log,
                    )
                    results.append(
                        RunResult(
                            population=pop_size,
                            mode=mode,
                            run=run,
                            seed=seed,
                            runtime_s=runtime_s,
                            generations=generations,
                            ms_per_gen=(runtime_s / generations * 1000)
                            if generations
                            else 0,
                            worker_count=workers,
                            status="ok",
                            error=None,
                        )
                    )
                except Exception as exc:
                    logging.exception(
                        "%s run failed for population %s run %s", mode, pop_size, run
                    )
                    results.append(
                        RunResult(
                            population=pop_size,
                            mode=mode,
                            run=run,
                            seed=seed,
                            runtime_s=None,
                            generations=None,
                            ms_per_gen=None,
                            worker_count=(os.cpu_count() or 1) if parallel else 1,
                            status="failed",
                            error=str(exc),
                        )
                    )

    raw_df = pd.DataFrame([result.__dict__ for result in results])
    raw_csv = OUTPUT_DIR / "benchmark_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    fitness_df = pd.DataFrame(fitness_log)
    if not fitness_df.empty:
        fitness_path = OUTPUT_DIR / "fitness_trajectory.csv"
        fitness_df.to_csv(fitness_path, index=False)
        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=fitness_df,
            x="Generation",
            y="Best Fitness",
            hue="Mode",
            errorbar="sd",
        )
        plt.title("Best Fitness Trajectory")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fitness_trajectory.png")
        plt.close()

    valid_df = raw_df[raw_df["status"] == "ok"]
    summary = (
        valid_df.groupby(["population", "mode"], as_index=False)
        .agg(
            {
                "runtime_s": "median",
                "generations": "median",
                "ms_per_gen": "median",
                "worker_count": "median",
            }
        )
        .rename(
            columns={
                "population": "Population",
                "mode": "Mode",
                "runtime_s": "Runtime (s)",
                "generations": "Generations",
                "ms_per_gen": "ms/Gen",
                "worker_count": "Workers",
            }
        )
    )
    summary["Population"] = pd.to_numeric(summary["Population"], errors="coerce")
    summary = summary.dropna(subset=["Population"])
    summary["Population"] = summary["Population"].astype(int)
    summary = summary.sort_values(["Population", "Mode"]).reset_index(drop=True)

    speedup_rows = []
    for pop in summary["Population"].unique():
        single_row = summary[
            (summary["Population"] == pop) & (summary["Mode"] == "Single")
        ]
        parallel_row = summary[
            (summary["Population"] == pop) & (summary["Mode"] == "Parallel")
        ]
        if single_row.empty or parallel_row.empty:
            continue
        single_time = single_row.iloc[0]["Runtime (s)"]
        parallel_time = parallel_row.iloc[0]["Runtime (s)"]
        workers = parallel_row.iloc[0]["Workers"]
        speedup = single_time / parallel_time if parallel_time else 0
        efficiency = speedup / workers if workers else 0
        speedup_rows.append(
            {"Population": pop, "Speedup": speedup, "Efficiency": efficiency}
        )

    speedup_df = pd.DataFrame(speedup_rows)
    summary = summary.merge(speedup_df, on="Population", how="left")
    summary_csv = OUTPUT_DIR / "benchmark_summary.csv"
    summary.to_csv(summary_csv, index=False)

    if summary.empty:
        raise BenchmarkError("No successful benchmark runs")

    sns.set_theme(style="whitegrid")
    save_plot(summary, "Runtime (s)", "Runtime", "runtime.png")
    save_plot(summary, "Generations", "Generations to Solution", "generations.png")
    save_plot(summary, "ms/Gen", "Time per Generation (ms)", "time_per_generation.png")

    logging.info("Saved raw results to %s", raw_csv)
    logging.info("Saved summary to %s", summary_csv)
    logging.info("Saved charts to %s", OUTPUT_DIR)
