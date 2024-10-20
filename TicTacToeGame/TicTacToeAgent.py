# TicTacToeAgent.py

import sys
import os
import pickle
from .TicTacToeModel import EMPTY, X, O

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class TicTacToeAgent:
    def __init__(self, player='O'):
        self.player = O if player == 'O' else X
        # Use resource_path to handle paths correctly in a packaged app
        self.model = self.load_model(resource_path('Models/TicTacToe_model.pkl'))

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_move(self, game):
        state = self.game_to_state(game)
        if state in self.model:
            return self.model[state]
        else:
            # Fallback to any available move (should not happen with a complete model)
            for i in range(3):
                for j in range(3):
                    if game.board[i][j] == ' ':
                        return (i, j)

    def game_to_state(self, game):
        return tuple(tuple(self.char_to_int(cell) for cell in row) for row in game.board)

    def char_to_int(self, char):
        if char == ' ':
            return EMPTY
        elif char == 'X':
            return X
        elif char == 'O':
            return O