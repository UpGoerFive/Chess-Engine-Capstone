import numpy as np
import pandas as pd
import chess


def fen_to_array(fen):
    """
    Takes a FEN board position, and converts it to a 64 x 13 array. The array is structured similarly to the FEN with array[0] representing a8 on the chess board,
    with the rest of the rank following and then each successive rank in descending order, array[-1] is h1.
    """
    board_array = np.zeros((64, 13))

    board_string = ''
    for row in fen.split()[0].split('/'):
        for char in row:
            addon = char if char not in '12345678' else '0' * int(char)
            board_string += addon

    if len(board_string) != 64:
        raise ValueError(f"Board string has incorrect string length of {len(board_string)}")

    # Maps a FEN character to an index
    piece_dict = {'0': 0,
                  'P': 1,
                  'N': 2,
                  'B': 3,
                  'R': 4,
                  'Q': 5,
                  'K': 6,
                  'p': 7,
                  'n': 8,
                  'b': 9,
                  'r': 10,
                  'q': 11,
                  'k': 12}

    for _ in range(64):
        board_array[_, piece_dict[board_string[_]]] = 1

    return board_array


def puzzle_cleaning(puzzle_df):
    """
    Function to advance puzzle FEN by one ply and create single move target column. Returns a new data frame of FEN and target move.
    """
    new_df = pd.DataFrame(index=puzzle_df.index)

    new_df['FEN'] = puzzle_df.apply(single_move, axis=1)
    new_df['target_move'] = puzzle_df['Moves'].apply(lambda move_string: move_string.split()[1])

    return new_df

def single_move(game_series):
    """
    Advances game fen by one move, will probably be slow, but ideally should only be used once.
    """
    board = chess.Board(game_series.FEN)
    board.push_uci(game_series.Moves.split()[0])
    return board.fen()