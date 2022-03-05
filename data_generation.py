import numpy as np
import pandas as pd
import chess
import random
import math
from tensorflow.keras.utils import Sequence
from fenpreprocessing import fen_to_array


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