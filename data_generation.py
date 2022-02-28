import numpy as np
import pandas as pd
import chess
import math
from tensorflow.keras.utils import Sequence


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

class PosGen(Sequence):
    """
    Class based generator based on the Keras documentation for the Sequence class.
    """
    def __init__(self, position_df, xlabel, ylabel, batch_size=32):
        self.batch_size = batch_size
        self.df = position_df
        self.df_index = position_df.index
        self.x = xlabel
        self.y = ylabel

    def __len__(self):
        """
        Starting len method used in the Keras documentation.
        """
        return math.ceil(len(self.df_index) / self.batch_size)

    def __getitem__(self, idx):
        """
        Adapted from Keras documentation.
        """
        batch = self.df_index[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([fix_positions(position_array) for position_array in self.df.loc[batch, self.x]]), np.array(self.df.loc[batch, self.y])
