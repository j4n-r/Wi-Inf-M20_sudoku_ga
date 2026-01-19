# what: import future annotations for type hints
# how: use __future__ import
# when: at the start of the file
# why: to enable postponed evaluation of type annotations
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


# what: set up output directory and log path
# how: create output directory and define log file path
# when: before logging setup
# why: to store benchmark results and logs
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "suoko-altair-benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)


# what: define benchmark parameters
# how: set constants for population, mutation, elitism, etc.
# when: before running benchmarks
# why: to control experiment settings
POPULATION_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
STAGNATION_LIMIT = 70
MUTATION_RATE = 0.1
ELITISM_RATE = 0.05
TOURNAMENT_SIZE = 3
CHUNK_SIZES = [40, 80, 160, 320, 640, 1280]
RUNS = 10
BASE_SEED = 51
MODE_ORDER = ["Single", "Parallel"]
PALETTE = {"Single": "#6b6b6b", "Parallel": "#e67e22"}


# what: define result data structure
# how: use dataclass for run results
# when: before collecting results
# why: to organize benchmark output
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
    chunk_size: int
    status: str
    error: str | None


# what: define custom error for benchmark
# how: subclass RuntimeError
# when: before error handling
# why: to signal benchmark-specific issues
class BenchmarkError(RuntimeError):
    pass


# what: run genetic algorithm for sudoku
# how: initialize population, run evolution loop, handle parallelism
# when: for each benchmark run
# why: to measure performance and collect results
def run_ga(
    pop_size: int,
    chunk_size: int,
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

    # what: set up worker pool for parallel mode
    # how: use ProcessPoolExecutor if parallel
    # when: before evolution loop
    # why: to enable parallel fitness calculation
    worker_pool = None
    worker_count = 1
    if parallel:
        max_workers = os.cpu_count() or 1
        worker_count = max(1, min(max_workers, pop_size // max(1, chunk_size)))
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
            # what: calculate fitness for population
            # how: use parallel or single fitness function
            # when: every generation
            # why: to evaluate current solutions
            if worker_pool is None:
                fitness_scores = calculate_fitness_population(population)
            else:
                fitness_scores = calculate_fitness_parallel(
                    population, worker_pool, chunk_size
                )

            best_fitness = min(fitness_scores)
            # what: check for solution found
            # how: if best fitness is zero
            # when: every generation
            # why: to stop when sudoku is solved
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

            # what: log fitness progress
            # how: append to fitness_log every 20 generations
            # when: every 20 generations
            # why: to track progress over time
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

            # what: check for improvement or stagnation
            # how: compare best fitness to previous best
            # when: every generation
            # why: to reset if stuck
            if best_fitness < best_fitness_ever:
                best_fitness_ever = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # what: handle stagnation reset
            # how: reinitialize population if no improvement
            # when: after stagnation limit reached
            # why: to escape local optima
            if generations_without_improvement >= STAGNATION_LIMIT:
                restart_count += 1
                logging.info(
                    "%s run %s/%s pop %s: stagnation reset %s at gen %s",
                    mode,
                    run_idx + 1,
                    total_runs,
                    pop_size,
                    restart_count,
                    global_generations,
                )
                generations_without_improvement = 0
                best_fitness_ever = 100
                generation = 0
                restart_seed = seed + restart_count * 1_000_003
                random.seed(restart_seed)
                population = make_initial_population(SUDOKU, pop_size, row_mutable)
            else:
                # what: evolve population to next generation
                # how: use single or parallel evolution function
                # when: every generation
                # why: to improve solutions
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
        # what: clean up worker pool
        # how: shutdown ProcessPoolExecutor
        # when: after evolution loop ends
        # why: to release system resources
        if worker_pool is not None:
            worker_pool.shutdown(cancel_futures=True)


# what: save summary plot to file
# how: use seaborn and matplotlib to plot and save
# when: after summarizing results
# why: to visualize benchmark outcomes
def save_plot(summary: pd.DataFrame, y: str, title: str, filename: str) -> None:
    plt.figure(figsize=(8, 4))
    order = sorted(summary["Population"].unique())
    sns.barplot(
        data=summary,
        x="Population",
        y=y,
        hue="Mode",
        order=order,
        hue_order=MODE_ORDER,
        palette=PALETTE,
    )
    plt.xlabel("Population")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


# what: main benchmark execution
# how: run benchmarks, collect results, save outputs
# when: when script is run directly
# why: to perform and record sudoku ga benchmarks
if __name__ == "__main__":
    results: list[RunResult] = []
    fitness_log: list[dict] = []

    # what: run benchmarks for each population and chunk size
    # how: loop over parameter combinations and run ga
    # when: at start of main
    # why: to gather performance data
    for pop_size, chunk_size in zip(POPULATION_SIZES, CHUNK_SIZES):
        logging.info(
            "Benchmarking population size %s with chunk %s", pop_size, chunk_size
        )
        for run in range(RUNS):
            seed = BASE_SEED + run
            for parallel, mode in [(False, "Single"), (True, "Parallel")]:
                try:
                    runtime_s, generations, workers = run_ga(
                        pop_size,
                        chunk_size,
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
                            chunk_size=chunk_size,
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
                            chunk_size=chunk_size,
                            status="failed",
                            error=str(exc),
                        )
                    )

    # what: save raw results to csv
    # how: convert results to dataframe and save
    # when: after all runs complete
    # why: to persist experiment data
    raw_df = pd.DataFrame([result.__dict__ for result in results])
    run_plot_df = raw_df[raw_df["status"] == "ok"]

    if not run_plot_df.empty:
        plt.figure(figsize=(8, 4))
        sns.boxplot(
            data=run_plot_df,
            x="population",
            y="runtime_s",
            hue="mode",
            order=POPULATION_SIZES,
            hue_order=MODE_ORDER,
            palette=PALETTE,
            fliersize=0,
        )
        sns.stripplot(
            data=run_plot_df,
            x="population",
            y="runtime_s",
            hue="mode",
            order=POPULATION_SIZES,
            hue_order=MODE_ORDER,
            dodge=True,
            palette=PALETTE,
            alpha=0.6,
            size=2.5,
            linewidth=0,
            jitter=0.18,
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:2], labels[:2], title="Mode", loc="upper left")
        plt.title("Runtime Distribution")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "runtime_distribution.png")
        plt.close()

    # what: save and plot fitness trajectory
    # how: save fitness log and plot line chart
    # when: after all runs
    # why: to analyze fitness progress
    fitness_df = pd.DataFrame(fitness_log)
    if not fitness_df.empty:
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

    # what: summarize results by group
    # how: group by population and mode, aggregate metrics
    # when: after plotting
    # why: to create summary table
    valid_df = raw_df[raw_df["status"] == "ok"]
    summary = (
        valid_df.groupby(["population", "mode"], as_index=False)
        .agg(
            {
                "runtime_s": "median",
                "generations": "median",
                "ms_per_gen": "median",
                "worker_count": "median",
                "chunk_size": "median",
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
                "chunk_size": "Chunk",
            }
        )
    )
    summary["Population"] = pd.to_numeric(summary["Population"], errors="coerce")
    summary = summary.dropna(subset=["Population"])
    summary["Population"] = summary["Population"].astype(int)
    summary = summary.sort_values(["Population", "Mode"]).reset_index(drop=True)

    # what: calculate speedup and efficiency
    # how: compare single and parallel runtimes
    # when: after summary table
    # why: to measure parallel performance
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

    # what: check for successful runs
    # how: raise error if summary is empty
    # when: after all processing
    # why: to ensure results exist
    if summary.empty:
        raise BenchmarkError("No successful benchmark runs")

    # what: save summary plots
    # how: call save_plot for each metric
    # when: after summary table
    # why: to visualize key results
    sns.set_theme(style="whitegrid")
    save_plot(summary, "Runtime (s)", "Runtime", "runtime.png")
    save_plot(summary, "Generations", "Generations to Solution", "generations.png")
    save_plot(summary, "ms/Gen", "Time per Generation (ms)", "time_per_generation.png")

    # what: log output file locations
    # how: use logging.info to print paths
    # when: after saving files
    # why: to inform user of results
    logging.info("Saved charts to %s", OUTPUT_DIR)
