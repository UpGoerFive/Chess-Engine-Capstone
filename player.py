import numpy as np
import chess
from tensorflow.keras import models
from fenpreprocessing import fen_to_array

class Player:
    """
    Basic player class. Must have a keras model, but can accept an h5 or the heavier save model type.
    Optionally takes a searching algorithm object, which is recommended for any appreciable strength.
    """
    def __init__(self, model, searcher=None, start_fen=chess.STARTING_FEN):
        """
        Basic player that uses python chess to make moves.
        """
        self.model = models.load_model(model)
        self.searcher = searcher
        self.set_position(fen=start_fen)

    def set_position(self, fen):
        """
        Sets the board position for the Player, useful for having the player play a new game.
        """
        self.board = chess.Board(fen=fen)

    def show_board(self):
        """
        Displays board for play.
        """
        return self.board

    def play_move(self):
        """
        Plays best predicted move and displays board.
        Optionally accepts a searching algorithm which then uses the player model
        as an evaluation metric.
        """
        if self.searcher:
            best_move = self.searcher.search(self.board.fen(), self.model)
        else:
            options = list(self.board.legal_moves)
            fens = []

            for move in options:
                self.board.push(move)
                fens.append(self.board.fen())
                self.board.pop()

            fens = np.array([fen_to_array(fen).reshape(8,8,13) for fen in fens]) # This is slow, but I just need something working first
            best_move = options[np.argmax(self.model.predict(fens))]

        self.board.push(best_move)
        self.show_board()
        return best_move

    def op_move(self, move):
        """
        Receives opponent move.
        """
        self.board.push(move) if type(move) == chess.Move else self.board.push_san(move)
        self.show_board()

class Searcher:
    """
    Searching class for evaluation. Takes a starting fen and a model for an evaluation metric.
    """
    def __init__(self):
        pass

    def search(self, starting_fen, model):
        """
        Searching method using basic fail soft alpha beta search based on the wikipedia [pseudocode](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
        """
        board = chess.Board(fen=starting_fen)


class Single_Game:
    """
    Plays one game between two Player objects. player_names should be a dictionary with 'white' and 'black'
    as the keys, and the desired names as values.
    """
    def __init__(self, white: Player, black: Player, player_names=None):
        self.players = {'white': white, 'black': black}
        self.active_player = 'white'
        self.passive_player = 'black'

        self.game_board = chess.Board()

        self.players['white'].set_position(self.game_board.fen())
        self.players['black'].set_position(self.game_board.fen())

        if player_names:
            self.names = player_names

    def play(self):
        """
        Plays out the game. Returns basic information.
        """
        ply_num = 0
        switcher = ('white', 'black')
        while not self.game_board.outcome():
            next_passive = ply_num % 2
            move = self.players[self.active_player].play_move()
            self.game_board.push(move)
            self.players[self.passive_player].op_move(move)

            # Advance Ply by one and switch players
            ply_num += 1
            self.active_player = switcher[ply_num % 2]
            self.passive_player = switcher[next_passive]

        return self.game_board.outcome(), self.game_board.move_stack

