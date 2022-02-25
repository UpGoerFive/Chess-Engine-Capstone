import numpy as np
import pandas as pd
import chess
import tensorflow as tf
from tensorflow.keras import models
from fenpreprocessing import fen_to_array

class Player:
    def __init__(self, model, start_fen=chess.STARTING_FEN):
        """
        Basic player that uses python chess to make moves.
        """
        self.model = models.load_model(model)
        self.board = chess.Board(fen=start_fen)
        self.show_board()

    def show_board(self):
        """
        Displays board for play.
        """
        return self.board

    def play_move(self):
        """
        Plays best predicted move and displays board.
        """
        options = list(self.board.legal_moves)
        fens = []

        for move in options:
            self.board.push(move)
            fens.append(self.board.fen())
            self.board.pop()

        fens = np.array([fen_to_array(fen).reshape(8,8,13) for fen in fens]) # This is slow, but I just need something working first
        move_vals = self.model.predict(fens)
        best_move = options[np.argmax(move_vals)]

        self.board.push(best_move)
        self.show_board()
        return best_move

    def op_move(self, move):
        """
        Receives opponent move.
        """
        self.board.push_san(move)
        self.show_board()
