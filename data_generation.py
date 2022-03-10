import numpy as np
import pandas as pd
import chess
import random
import math
from pathlib import Path
from tensorflow.keras.utils import Sequence

######################################## Preprocessing ########################################

def fen_to_array(fen):
    """
    Takes a FEN board position, and converts it to a 64 x 13 array. The array is structured similarly to the FEN with array[0] representing a8 on the chess board,
    with the rest of the rank following and then each successive rank in descending order, array[-1] is h1.
    """
    board_array = np.zeros(832)

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
        board_array[13 * _ + piece_dict[board_string[_]]] = 1

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

def possible_moves(fen_series):
    """
    Gets a list of all possible moves from a starting fen. Takes each move and creates a new fen, then applies the array function to get
    the array representation of the board. The `target` column is a 1 or a 0 depending on if the position is the correct move or not in the puzzle.
    Intended to be used after puzzle cleaning, as it takes a series with one fen and one target move.
    """

    start_fen = fen_series.FEN
    target = fen_series.target_move

    board = chess.Board(start_fen)
    array_list = []

    for move in board.legal_moves:
        board.push_uci(move.uci())
        candidate_fen = board.fen()
        target_val = 1. if move == chess.Move.from_uci(target) else 0.
        array_list.append((fen_to_array(candidate_fen), target_val))
        board.pop()
    return pd.DataFrame(array_list, columns=['Position', 'Target'])

def make_converted_file(infile, outfile, out_cols = None):
    """
    Takes a csv cleaned with puzzle_cleaning and writes a version with fens transformed into array representations for each move, and a target value
    corresponding to whether or not the position is the puzzle solution.
    """

    if out_cols is None:
        out_cols = ['Position', 'Target']

    for i, chunk in enumerate(pd.read_csv(Path(infile), chunksize=100)):
        header = i == 0
        mode = 'w' if i == 0 else 'a'
        chunk_df = pd.DataFrame(chunk).apply(possible_moves, axis=1)
        restacked_df = pd.concat(chunk_df.values, ignore_index=True)
        restacked_df.to_csv(outfile, columns=out_cols, header=header, mode=mode, index=False)


######################################## Data Generators ########################################

def position_generator(position_df, batch_size=32):
    """
    Takes a data frame and yields a tuple of numpy arrays of batch size.
    """
    idx = 0
    while True:
        targets = position_df.iloc[idx:idx + batch_size, 1].to_numpy()
        positions = np.stack(position_df.iloc[idx:idx + batch_size, 0].apply(fix_positions))
        idx += batch_size
        yield (positions, targets)

def fix_positions(position_string):
    """
    Deals with weird newline thing until I can figure out why things were saved that way.
    """
    return np.fromstring(position_string.replace('\n', '')[1: -1], dtype=np.float, sep=' ').reshape((8,8,13))

class ChessPositionGen(Sequence):
    """
    Data generator class based on the Keras documentation for the Sequence class.
    """
    def __init__(self, position_df, shuffle=True, batch_size=32, xpos=0, ypos=1):
        self.batch_size = batch_size
        self.df = position_df
        self.df_index = position_df.index
        self.shuffle = shuffle
        self.xpos = xpos
        self.ypos = ypos
        self.on_epoch_end()

    def __len__(self):
        """
        Starting len method used in the Keras documentation.
        """
        return math.ceil(len(self.df_index) / self.batch_size)

    def __getitem__(self, idx):
        """
        Adapted from Keras documentation, and helpful blog here: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        """

        # Pulls out shuffled indices from df_index using indices from on_epoch_end
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df_index[batch_indices]

        return self.__data_generation(batch)

    def __data_generation(self, indices, sample_size=4):
        """
        Returns properly converted data for a batch. Adapts code from possible moves function in fenpreprocessing.py.
        Each starting FEN is converted to 1 correct move array, and sample_size randomly selected incorrect moves.
        """
        array_list = []
        target_list = []

        for fen_entry in self.df.loc[indices, :].values:

            # Initialize board and target move
            start_fen = fen_entry[self.xpos]
            target = chess.Move.from_uci(fen_entry[self.ypos])
            board = chess.Board(start_fen)

            # Create sampled move list. Random Sampling so that data is less imbalanced.
            move_list = list(board.legal_moves)
            move_list = random.sample(move_list, min(sample_size, len(move_list)))
            if target not in move_list:
                move_list.append(target)

            # Create board array and label for each position in move list
            for move in move_list:
                board.push(move)
                candidate_fen = board.fen()
                target_val = 1. if move == target else 0.
                array_list.append(fen_to_array(candidate_fen).reshape((8,8,13)))
                target_list.append(target_val)
                board.pop()
        return np.array(array_list), np.array(target_list)


    def on_epoch_end(self):
        """
        Shuffles indices after each epoch.
        """
        self.indices = np.arange(len(self.df_index))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


class DenseGenerator(Sequence):
    """
    Data generator class based on the Keras documentation for the Sequence class, slightly modified for dense networks.
    Will be merged with ChessPositionGen, but included for right now for notebook compatibility.
    """
    def __init__(self, position_df, shuffle=True, batch_size=32, xpos=0, ypos=1):
        self.batch_size = batch_size
        self.df = position_df
        self.df_index = position_df.index
        self.shuffle = shuffle
        self.xpos = xpos
        self.ypos = ypos
        self.on_epoch_end()

    def __len__(self):
        """
        Starting len method used in the Keras documentation.
        """
        return math.ceil(len(self.df_index) / self.batch_size)

    def __getitem__(self, idx):
        """
        Adapted from Keras documentation, and helpful blog here: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        """

        # Pulls out shuffled indices from df_index using indices from on_epoch_end
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df_index[batch_indices]

        return self.__data_generation(batch)

    def __data_generation(self, indices, sample_size=4):
        """
        Returns properly converted data for a batch. Adapts code from possible moves function in fenpreprocessing.py.
        Each starting FEN is converted to 1 correct move array, and sample_size randomly selected incorrect moves.
        """
        array_list = []
        target_list = []

        for fen_entry in self.df.loc[indices, :].values:

            # Initialize board and target move
            start_fen = fen_entry[self.xpos]
            target = chess.Move.from_uci(fen_entry[self.ypos])
            board = chess.Board(start_fen)

            # Create sampled move list. Random Sampling so that data is less imbalanced.
            move_list = list(board.legal_moves)
            move_list = random.sample(move_list, min(sample_size, len(move_list)))
            if target not in move_list:
                move_list.append(target)

            # Create board array and label for each position in move list
            for move in move_list:
                board.push(move)
                candidate_fen = board.fen()
                target_val = 1. if move == target else 0.
                array_list.append(fen_to_array(candidate_fen))
                target_list.append(target_val)
                board.pop()
        return np.array(array_list), np.array(target_list)


    def on_epoch_end(self):
        """
        Shuffles indices after each epoch.
        """
        self.indices = np.arange(len(self.df_index))
        if self.shuffle == True:
            np.random.shuffle(self.indices)