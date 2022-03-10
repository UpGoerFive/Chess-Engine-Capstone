from random import shuffle
import numpy as np
import pandas as pd
import chess
import pytest

import fenpreprocessing

@pytest.fixture
def board_positions(db_path='lichess_db_puzzle.csv'):
    """
    Create test set.
    """
    return pd.read_csv(
        db_path,
        names=[
            'PuzzleId',
            'FEN',
            'Moves',
            'Rating',
            'RatingDeviation',
            'Popularity',
            'NbPlays',
            'Themes',
            'GameUrl',
        ],
        nrows=200,
    )

def test_puzzle_clean(board_positions):
    """
    Tests the puzzle cleaning function by comparing random entries with their starting values.
    """
    subset = board_positions.sample(n=20)
    cleaned = fenpreprocessing.puzzle_cleaning(subset)
    assert cleaned.index.equals(subset.index)
    for idx in subset.index:
        moves = subset.Moves[idx].split()
        test_board = chess.Board(subset.FEN[idx])
        test_board.push_uci(moves[0])
        assert cleaned.FEN[idx] == test_board.fen()
        assert cleaned.target_move[idx] == moves[1]
