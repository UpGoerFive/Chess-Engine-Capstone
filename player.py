import numpy as np
import chess
import time
import random
import math
from tensorflow.keras import models
from fenpreprocessing import fen_to_array

class Player:
    """
    Basic player class. Must have a keras model, but can accept an h5 or the heavier save model type.
    Optionally takes a searching algorithm class, which is recommended for any appreciable strength.
    """
    def __init__(self, model, searcher=None, start_fen=chess.STARTING_FEN):
        """
        Basic player that uses python chess to make moves.
        model should be a keras saved model, either a .h5 file or a save model type.
        """
        self.model = models.load_model(model)
        self.searcher = searcher(self.model) if searcher else None
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
            best_move = self.searcher.search(self.board.copy())
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
    def __init__(self, model=None):
        self.evmetric = model

    def fen_dict(self, board):
        """
        Returns a move: FEN dictionary of all legal moves.
        """
        fens = {}
        for move in board.legal_moves:
                board.push(move)
                fens[move] = board.fen()
                board.pop()

        return fens

    def search(self, board, time_limit=10000):
        """
        Searching method using basic fail-soft alpha-beta search based on the wikipedia [pseudocode](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning)
        Takes a board object and time limit for search in miliseconds (this is how lichess provides time values).
        """

        # Setup timing
        current = time.perf_counter() * 1000
        stop_time = current + time_limit * 0.9 # Leave a buffer to avoid time outs if time limit represents total time left.
        print("Start times: ", current, stop_time)

        # Initial pruning values
        moves = None
        alpha = -math.inf
        beta = math.inf
        depth = 1

        while current < stop_time:
            # This is an IDDFS loop limited by available time
            value, best_move, moves = self.alpha_beta(board.fen(), moves=moves, depth=depth, alpha=alpha, beta=beta, max_player=True)
            depth += 1
            current = time.perf_counter() * 1000
            print(depth, value, best_move, moves)

        return best_move

    def alpha_beta(self, fen, moves, depth, alpha, beta, max_player: bool):
        """
        Alpha beta pruning. Returns the best move, and the moves in order of value, for the next iteration.
        """
        board = chess.Board(fen=fen)

        # Setup and shuffle all potential future positions if this is the first pass
        if not moves:
            moves = self.fen_dict(board)
            random.shuffle(list(moves.keys()))

        move_list = list(moves.keys()) if moves else None

        if depth == 0 or not moves: # End of depth and leaf nodes
            # Function is always called with depth of at least 1 when multiple returns are expected.
            return (self.evmetric.predict(fen_to_array(fen).reshape(1,8,8,13))[0, 0],) # return tuple for *some* consistency


        move_order = []
        value = -math.inf if max_player else math.inf
        for move in move_list:
            board.push(move)
            if max_player:
                value = max(value, self.alpha_beta(board.fen(), moves=None, depth=depth-1, alpha=alpha, beta=beta, max_player=False)[0])
                alpha = max(value, alpha)

                # Next two statements have to be included both times due to breaks
                move_order.append((move, value))
                board.pop()
                if value >= beta:
                    break # Beta cutoff
            else:
                value = min(value, self.alpha_beta(board.fen(), moves=None, depth=depth-1, alpha=alpha, beta=beta, max_player=True)[0])
                beta = min(value, beta)

                move_order.append((move, value))
                board.pop()
                if value <= alpha:
                    break # Alpha cutoff

        # Sort moves by value, alpha-beta pruning is best when best moves are searched first,
        # so the ordering is returned to be used in the next depth as the search order.
        return_order = sorted(move_order, key=lambda move_tup: move_tup[1], reverse=True)
        return return_order[0][1], return_order[0][0], {tup[0]: moves[tup[0]] for tup in return_order}


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

