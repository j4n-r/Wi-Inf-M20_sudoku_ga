## Sudoku GA Solver

Genetic algorithm Sudoku solver using NumPy. It can launch one independent search per CPU core; the first solved board wins.

### Run
```bash
uv run python src/main.py
```

### Numpy Version
Numpy Version is on the `numpy-optimized` branch
https://github.com/j4n-r/Wi-Inf-M20_sudoku_ga/tree/numpy-optimized

### Run options (see `src/main.py`)
- `--timeit --iterations N` — benchmark the default puzzle (spawns processes each run).
- `--csv path/to/sudoku.csv --count N` — solve the first N puzzles from a `quizzes,solutions` CSV (ignores `--timeit`).

### Dataset
If you want to run with the dataset: (https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download)

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




