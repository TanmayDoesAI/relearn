# TicTacToeModel.py
EMPTY = 0
X = 1
O = 2

class TicTacToeModel:
    def __init__(self):
        self.board = [[' ']*3 for _ in range(3)]
        self.current_winner = None

    def mark_square(self, row, col, player):
        self.board[row][col] = player

    def available_square(self, row, col):
        return self.board[row][col] == ' '

    def empty_squares(self):
        return ' ' in [cell for row in self.board for cell in row]

    def get_state(self):
        return tuple(cell for row in self.board for cell in row)

    def check_winner(self, player):
        # Vertical check
        for col in range(3):
            if all([self.board[row][col] == player for row in range(3)]):
                self.current_winner = player
                return True

        # Horizontal check
        for row in range(3):
            if all([self.board[row][col] == player for col in range(3)]):
                self.current_winner = player
                return True

        # Diagonal checks
        if all([self.board[i][i] == player for i in range(3)]) or \
           all([self.board[i][2-i] == player for i in range(3)]):
            self.current_winner = player
            return True

        return False

    def reset(self):
        self.__init__()