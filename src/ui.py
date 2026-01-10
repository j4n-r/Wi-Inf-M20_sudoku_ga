import sys

type SudokuCandidate = list[list[int]]


# This is LLM generated
def render_sudoku(sudoku: SudokuCandidate):
    """
    Renders a 9x9 Sudoku board with a clean, minimal look.
    Vertical internal separators have been removed.
    """
    # --- Configuration ---
    C_GRID = "\033[38;5;240m"  # Dark Grey/Blue for the grid lines
    C_NUM = "\033[1;37m"  # Bright White for numbers
    C_ZERO = "\033[38;5;236m"  # Dim Grey for empty cells
    C_RESET = "\033[0m"

    EMPTY_CELL = "·"

    # Box Drawing Characters
    H_WALL = "═"
    V_WALL = "║"

    # Intersections (Double lines only)
    TOP_L, TOP_M, TOP_R = "╔", "╦", "╗"
    MID_L, MID_M, MID_R = "╠", "╬", "╣"
    BOT_L, BOT_M, BOT_R = "╚", "╩", "╝"

    # --- Construction ---

    # Width Calculation:
    # Each number is formatted as " N " (3 chars).
    # A block has 3 numbers: 3 * 3 = 9 chars wide.
    spacer = H_WALL * 9

    top_border = (
        f"{C_GRID}{TOP_L}{spacer}{TOP_M}{spacer}{TOP_M}{spacer}{TOP_R}{C_RESET}"
    )
    mid_divider = (
        f"{C_GRID}{MID_L}{spacer}{MID_M}{spacer}{MID_M}{spacer}{MID_R}{C_RESET}"
    )
    bot_border = (
        f"{C_GRID}{BOT_L}{spacer}{BOT_M}{spacer}{BOT_M}{spacer}{BOT_R}{C_RESET}"
    )

    # --- Helper to format a single row ---
    def format_row(row_data):
        formatted_cells = []
        for num in row_data:
            if num == 0:
                # Dimmed placeholder: " · "
                formatted_cells.append(f"{C_ZERO} {EMPTY_CELL} {C_RESET}")
            else:
                # Bright number: " N "
                formatted_cells.append(f"{C_NUM} {num} {C_RESET}")

        # Group cells into chunks of 3
        # We join them with nothing ("") because the cells already contain padding spaces
        left = "".join(formatted_cells[0:3])
        center = "".join(formatted_cells[3:6])
        right = "".join(formatted_cells[6:9])

        # Construct the row with Double Walls only
        return f"{C_GRID}{V_WALL}{C_RESET}{left}{C_GRID}{V_WALL}{C_RESET}{center}{C_GRID}{V_WALL}{C_RESET}{right}{C_GRID}{V_WALL}{C_RESET}"

    # --- Print the Board ---
    print(top_border)

    for i, row in enumerate(sudoku):
        print(format_row(row))

        # Add a divider after row 3 and 6
        if (i + 1) % 3 == 0 and (i + 1) < 9:
            print(mid_divider)

    print(bot_border)


def update_board(sudoku: SudokuCandidate):
    """
    Moves the cursor up 13 lines (height of the board)
    and re-prints the sudoku over the old one.
    """
    # \033[F is the ANSI code for "Previous Line"
    # We print it 13 times to get back to the top of the board
    sys.stdout.write("\033[F" * 13)
    render_sudoku(sudoku)
