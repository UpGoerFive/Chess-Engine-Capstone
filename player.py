import numpy as np
import pandas as pd
import chess
import tensorflow as tf
from tensorflow.keras import models

class Player:
    def __init__(self, model, start_fen=None):
        self.model = models.load_model(model)
        self.board = chess.Board(fen=start_fen)

    def play_move(self):
        options = list(self.board.legal_moves)
        move_vals = []
        for move in options:
            self.board.push_san(move)
            quality = self.model.predict(self.board.fen())
            self.board.pop()
            move_vals.append(move, quality)
        best_move = max(move_vals, key=lambda pair: pair[1])
        self.board.push_san(best_move[0])
        return best_move[0], self.board

    def op_move(self, move):
        self.board.push_uci(move)
        return self.board
