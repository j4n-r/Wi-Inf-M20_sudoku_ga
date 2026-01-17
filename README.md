## Sudoku GA Solver
Genetic algorithm Sudoku solver.

### Run
```bash
uv run python src/main.py
```

Change the config in `main.py`

### Numpy Version
Numpy Version is on the `numpy-optimized` branch
https://github.com/j4n-r/Wi-Inf-M20_sudoku_ga/tree/numpy-optimized

### Profiling
Flamegraph
``` bash
 uv run py-spy record --subprocesses -o profiles/profile.svg -- python src/main.py
```

Top
``` bash
uv run py-spy  top --subprocesses -- python src/main.py
```
The time is aggregated for all workers running a function

Speedscope 

https://www.speedscope.app/
``` bash
uv run py-spy record \
        --subprocesses \
        -f speedscope \
        -o profile.json \
        -- python src/main.py
```

Benchmark

To start the benchmark that compares single vs. parallel runs. output are graphs and logs

``` bash
uv run src/benchmark.py
```


