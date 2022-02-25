import numpy as np
import pandas as pd
import chess


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

# class PosGen:
