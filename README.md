## Sudoku GA Solver

Genetic algorithm Sudoku solver using NumPy. It can launch one independent search per CPU core; the first solved board wins.

### Run
```bash
uv run python src/main.py
```

### Run options (see `src/main.py`)
- `--timeit --iterations N` — benchmark the default puzzle (spawns processes each run).
- `--csv path/to/sudoku.csv --count N` — solve the first N puzzles from a `quizzes,solutions` CSV (ignores `--timeit`).

### Dataset
If you want to run with the dataset: (https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download)



