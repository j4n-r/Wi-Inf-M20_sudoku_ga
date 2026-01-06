from ga import SUDOKU, make_initial_population, tournament_selection


def main():
    # sudoku = fill_initial_sudoku(SUDOKO)
    # penalty = calculate_fitness(sudoku)
    population = make_initial_population(SUDOKU, 3)
    winner = tournament_selection(population)
    print(population)
    print(winner)


if __name__ == "__main__":
    main()
