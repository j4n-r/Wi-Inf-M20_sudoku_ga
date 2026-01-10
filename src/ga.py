from __future__ import annotations

type SudokuCandidate = list[list[int]]
type SudokuPopulation = list[SudokuCandidate]

GRID_SIZE = 9
fixed_mask: list[list[bool]]
row_mutable_indices: list[list[int]] = []
# my_rng = random.


def set_seed(seed: int):
    global my_rng
    pass


def set_mask(sudoku):
    """
    Cache mutable cell positions to avoid recomputing them inside hot loops.
    """
    global fixed_mask, row_mutable_indices
    pass


def calculate_population_fitness(population: SudokuPopulation) -> list[int]:
    pass

def make_initial_population(sudoku: SudokuCandidate, size)  -> SudokuPopulation:
    pass


def crossover(parent1: SudokuCandidate, parent2: SudokuCandidate) -> SudokuCandidate:
    pass


def mutate(sudoku: SudokuCandidate, mutation_rate: float):
    """
    Iterates through EVERY row. If a row hits the mutation_rate,
    we swap two non-fixed numbers in that row.
    """
    pass

def batch_tournament_winners(
    fitness_scores: list[int], selection_count: int, tournament_members: int = 3
) -> list[SudokuCandidate]:
    pass


def evolve_population(
    current_pop: SudokuPopulation,
    fitness_scores: list[int],
    mutation_rate: float,
    elitism_rate: int,
    tournament_members: int) -> SudokuPopulation:
    pass

def run_evolution(
    initial_board: SudokuCandidate,
    population: SudokuPopulation,
    mutation_rate: float,
    elitism_rate: int,
    tournament_members: int,
    stagnation_limit: int = 100,
) -> int:
    pass
