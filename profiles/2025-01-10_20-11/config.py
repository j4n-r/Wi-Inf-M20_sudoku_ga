test_sudoku: list[list[int]] = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# Program config
SEED = 10
USE_PARALLELIZATION = "True"  # "True" or "False"
GUI = "Generations"  # "Sudoku" or "Generations"

# Parameters
INTITIAL_BOARD = test_sudoku
POPULATION_SIZE = 8000
MUTATION_RATE = 0.05
ELITISM_RATE = 10
TOURNAMENT_MEMBERS = 3
STAGNATION_LIMIT = 70
