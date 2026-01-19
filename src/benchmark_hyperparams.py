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
LOG_PATH = OUTPUT_DIR / "benchmark_hyperparams.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

POPULATION_SIZES = [500, 4000, 16000]
RUNS = 3
BASE_SEED = 51
MODE_ORDER = ["Single", "Parallel"]
PALETTE = {"Single": "#6b6b6b", "Parallel": "#e67e22"}

DEFAULT_MUTATION_RATE = 0.1
DEFAULT_ELITISM_SIZE = 20
DEFAULT_TOURNAMENT_SIZE = 3
STAGNATION_LIMIT = 70

MUTATION_RATES = [0.05, 0.1, 0.15, 0.2, 0.3]
ELITISM_SIZES = [5, 10, 20, 40, 80]
TOURNAMENT_SIZES = [2, 3, 4, 5, 7]


@dataclass
class RunResult:
    parameter: str
    value: float
    population: int
    mode: str
    run: int
    seed: int
    runtime_s: float
    generations: int
    ms_per_gen: float
    worker_count: int


class BenchmarkError(RuntimeError):
    pass


def run_ga(
    pop_size: int,
    seed: int,
    parallel: bool,
    run_idx: int,
    total_runs: int,
    mutation_rate: float,
    elitism_size: int,
    tournament_size: int,
) -> RunResult:
    validate_sudoku(SUDOKU)
    mode = "Parallel" if parallel else "Single"
    row_mutable = calculate_mutable_indices(SUDOKU)
    random.seed(seed)
    population = make_initial_population(SUDOKU, pop_size, row_mutable)

    worker_pool = None
    worker_count = 1
    if parallel:
        max_workers = os.cpu_count() or 1
        worker_count = max(2, min(max_workers, pop_size // max(1, pop_size // 12)))
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
                    "%s run %s/%s pop %s solved in %.2fs",
                    mode,
                    run_idx + 1,
                    total_runs,
                    pop_size,
                    runtime_s,
                )
                return RunResult(
                    parameter="",
                    value=0.0,
                    population=pop_size,
                    mode=mode,
                    run=run_idx,
                    seed=seed,
                    runtime_s=runtime_s,
                    generations=global_generations,
                    ms_per_gen=(runtime_s / global_generations * 1000)
                    if global_generations
                    else 0,
                    worker_count=worker_count,
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
                        mutation_rate,
                        elitism_size,
                        tournament_size,
                        row_mutable,
                        generation_seed,
                    )
                else:
                    population = evolve_population_parallel(
                        population,
                        fitness_scores,
                        mutation_rate,
                        elitism_size,
                        tournament_size,
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


def save_param_plot(
    df: pd.DataFrame,
    param: str,
    y: str,
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(10, 4))
    plot_df = df[df["parameter"] == param]
    sns.boxplot(
        data=plot_df,
        x="value",
        y=y,
        hue="mode",
        hue_order=MODE_ORDER,
        palette=PALETTE,
        fliersize=0,
    )
    sns.stripplot(
        data=plot_df,
        x="value",
        y=y,
        hue="mode",
        hue_order=MODE_ORDER,
        dodge=True,
        palette=PALETTE,
        alpha=0.45,
        size=2,
        linewidth=0,
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Mode", loc="upper left")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def run_param_sweep(param: str, values: list, results: list[RunResult]) -> None:
    for value in values:
        for pop_size in POPULATION_SIZES:
            logging.info("Benchmarking %s=%s pop %s", param, value, pop_size)
            for run in range(RUNS):
                seed = BASE_SEED + run
                if param == "MUTATION_RATE":
                    mutation_rate = float(value)
                    elitism_size = DEFAULT_ELITISM_SIZE
                    tournament_size = DEFAULT_TOURNAMENT_SIZE
                elif param == "ELITISM_SIZE":
                    mutation_rate = DEFAULT_MUTATION_RATE
                    elitism_size = int(value)
                    tournament_size = DEFAULT_TOURNAMENT_SIZE
                else:
                    mutation_rate = DEFAULT_MUTATION_RATE
                    elitism_size = DEFAULT_ELITISM_SIZE
                    tournament_size = int(value)

                for parallel in [False, True]:
                    result = run_ga(
                        pop_size,
                        seed,
                        parallel,
                        run,
                        RUNS,
                        mutation_rate,
                        elitism_size,
                        tournament_size,
                    )
                    results.append(
                        RunResult(
                            parameter=param,
                            value=float(value),
                            population=pop_size,
                            mode=result.mode,
                            run=result.run,
                            seed=result.seed,
                            runtime_s=result.runtime_s,
                            generations=result.generations,
                            ms_per_gen=result.ms_per_gen,
                            worker_count=result.worker_count,
                        )
                    )


def main() -> None:
    results: list[RunResult] = []
    run_param_sweep("MUTATION_RATE", MUTATION_RATES, results)
    run_param_sweep("ELITISM_SIZE", ELITISM_SIZES, results)
    run_param_sweep("TOURNAMENT_SIZE", TOURNAMENT_SIZES, results)

    df = pd.DataFrame([result.__dict__ for result in results])
    df["Population"] = df["population"].astype(int)
    df["Mode"] = df["mode"].astype(str)

    sns.set_theme(style="whitegrid")
    for param in ["MUTATION_RATE", "ELITISM_SIZE", "TOURNAMENT_SIZE"]:
        save_param_plot(
            df,
            param,
            "runtime_s",
            f"Runtime by {param}",
            f"hp_{param.lower()}_runtime.png",
        )
        save_param_plot(
            df,
            param,
            "generations",
            f"Generations by {param}",
            f"hp_{param.lower()}_generations.png",
        )


if __name__ == "__main__":
    main()
