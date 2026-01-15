import time
import random
import altair as alt
import pandas as pd
import webbrowser
import tempfile
from ga import (
    make_initial_population,
    calculate_fitness_population,
    calculate_fitness_parallel,
    evolve_population,
    evolve_population_parallel)
from main import test_sudoku as SUDOKU
from concurrent.futures import ProcessPoolExecutor
import os

ROW_MUTABLE = [[j for j in range(9) if SUDOKU[i][j] == 0] for i in range(9)]  # mutable cells
POPULATION_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
STAGNATION_LIMIT = 100
MUTATION_RATE = 0.1
ELITISM_RATE = 2
TOURNAMENT_SIZE = 3
SEED = 11
USE_SEED = True
CHUNK_SIZE = 400
RUNS = 5


def benchmark(pop_size: int, parallel: bool) -> tuple[float, int]:
    # run one ga experiment
    if USE_SEED:
        random.seed(SEED)  # set random seed
    population = make_initial_population(SUDOKU, pop_size, ROW_MUTABLE)  # create population
    worker_pool = None
    worker_count = os.cpu_count() or 1

    if parallel:
        worker_pool = ProcessPoolExecutor(max_workers=worker_count)  # parallel pool

    best_fitness_ever = 100
    generations_without_improvement = 0
    generation = 0
    global_generations = 0

    start = time.perf_counter()  # start timer
    while True:
        if parallel:
            fitness = calculate_fitness_parallel(population, worker_pool, CHUNK_SIZE)  # parallel fitness
        else:
            fitness = calculate_fitness_population(population)  # single fitness

        best_fitness = min(fitness)

        if best_fitness == 0:  # solution found
            elapsed = time.perf_counter() - start
            if worker_pool:
                worker_pool.shutdown()
            return elapsed, global_generations

        if best_fitness < best_fitness_ever:  # improved
            best_fitness_ever = best_fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= STAGNATION_LIMIT:  # restart
            population = make_initial_population(SUDOKU, pop_size, ROW_MUTABLE)
            best_fitness_ever = 100
            generations_without_improvement = 0
            generation = 0
        else:
            if parallel:
                population = evolve_population_parallel(
                    population, fitness, MUTATION_RATE, ELITISM_RATE, TOURNAMENT_SIZE,
                    ROW_MUTABLE, worker_pool, worker_count, generation
                )
            else:
                population = evolve_population(
                    population, fitness, MUTATION_RATE, ELITISM_RATE, TOURNAMENT_SIZE,
                    ROW_MUTABLE, generation
                )

        generation += 1
        global_generations += 1


if __name__ == "__main__":
    # run benchmarks for all sizes
    data = []

    for size in POPULATION_SIZES:  # test each pop size
        print(f"testing {size} population ({RUNS} runs)")
        
        for mode in ['Single', 'Parallel']:  # test both modes
            times, gens = [], []
            for _ in range(RUNS):  # repeat runs
                t, g = benchmark(size, parallel=(mode == 'Parallel'))
                times.append(t)
                gens.append(g)
            avg_time = sum(times) / RUNS
            avg_gen = sum(gens) / RUNS
            ms_per_gen = (avg_time / avg_gen * 1000) if avg_gen > 0 else 0
            data.append({'Population': size, 'Mode': mode, 'Runtime (s)': avg_time, 'Generations': avg_gen, 'ms/Gen': ms_per_gen})
            print(f"  {mode}: {avg_time:.2f}s (gen {avg_gen:.0f})")

    df = pd.DataFrame(data)  # results table
    
    config = f"stagnation: {STAGNATION_LIMIT}, mutation: {MUTATION_RATE}, elitism: {ELITISM_RATE}, tournament: {TOURNAMENT_SIZE}, seed: {SEED}, chunk: {CHUNK_SIZE}, runs: {RUNS}"

    # build altair charts
    base = alt.Chart(df).encode(
        x=alt.X('Population:O', title='Population Size'),
        color=alt.Color('Mode:N', scale=alt.Scale(domain=['Single', 'Parallel'], range=['#6fa8dc', '#e06666']))
    )

    c1 = base.mark_bar().encode(y=alt.Y('Runtime (s):Q'), xOffset='Mode:N').properties(title='Runtime', width=250)
    c2 = base.mark_bar().encode(y=alt.Y('Generations:Q'), xOffset='Mode:N').properties(title='Generations to Solution', width=250)
    c3 = base.mark_bar().encode(y=alt.Y('ms/Gen:Q', title='Time per Generation (ms)'), xOffset='Mode:N').properties(title='Time per Generation', width=250)

    chart = (c1 | c2 | c3).properties(title=config)
    # save chart as html
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        chart.save(f.name)
        webbrowser.open(f'file://{f.name}')
