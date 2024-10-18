# TicTacToeGame.py
import pygame
import sys
import random
from .TicTacToeAgent import TicTacToeAgent
from .TicTacToeModel import TicTacToeModel, EMPTY, X, O

# Constants
WIDTH, HEIGHT = 600, 700
ROWS, COLS = 3, 3
SQUARE_SIZE = WIDTH // COLS
LINE_WIDTH = 15
CROSS_WIDTH = 25
CIRCLE_WIDTH = 15
CIRCLE_RADIUS = SQUARE_SIZE // 3

# Colors
BG_COLOR = (240, 230, 140)  # Khaki
LINE_COLOR = (70, 130, 180)  # Steel Blue
CROSS_COLOR = (220, 20, 60)  # Crimson
CIRCLE_COLOR = (34, 139, 34)  # Forest Green
TEXT_COLOR = (25, 25, 112)  # Midnight Blue

class TicTacToeGame:
    def __init__(self):
        pygame.init()
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Tic Tac Toe')
        self.font = pygame.font.Font(None, 40)
        self.model = TicTacToeModel()
        self.ai_player = TicTacToeAgent('O')
        self.human_player = 'X'
        self.current_player = random.choice(['X', 'O'])
        self.game_over = False
        self.message = f"{self.current_player}'s turn"

    def draw_lines(self):
        for i in range(1, ROWS):
            pygame.draw.line(self.win, LINE_COLOR, (0, i * SQUARE_SIZE), 
                             (WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
        for i in range(1, COLS):
            pygame.draw.line(self.win, LINE_COLOR, (i * SQUARE_SIZE, 0), 
                             (i * SQUARE_SIZE, HEIGHT - 100), LINE_WIDTH)

    def draw_figures(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.model.board[row][col] == 'X':
                    start_desc = (col * SQUARE_SIZE + SQUARE_SIZE // 5, 
                                  row * SQUARE_SIZE + SQUARE_SIZE // 5)
                    end_desc = (col * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5, 
                                row * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5)
                    pygame.draw.line(self.win, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH)
                    
                    start_asc = (col * SQUARE_SIZE + SQUARE_SIZE // 5, 
                                 row * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5)
                    end_asc = (col * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5, 
                               row * SQUARE_SIZE + SQUARE_SIZE // 5)
                    pygame.draw.line(self.win, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)
                elif self.model.board[row][col] == 'O':
                    center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, 
                              row * SQUARE_SIZE + SQUARE_SIZE // 2)
                    pygame.draw.circle(self.win, CIRCLE_COLOR, center, CIRCLE_RADIUS, CIRCLE_WIDTH)

    def draw_status(self):
        pygame.draw.rect(self.win, BG_COLOR, (0, HEIGHT - 100, WIDTH, 100))
        text = self.font.render(self.message, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(WIDTH/2, HEIGHT - 50))
        self.win.blit(text, text_rect)

    def handle_click(self, row, col):
        if self.model.available_square(row, col):
            self.model.mark_square(row, col, self.human_player)
            if self.model.check_winner(self.human_player):
                self.game_over = True
                self.message = f"{self.human_player} wins!"
            else:
                self.current_player = 'O'
                self.message = f"{self.current_player}'s turn"

    def ai_move(self):
        move = self.ai_player.get_move(self.model)
        if move:
            self.model.mark_square(move[0], move[1], 'O')
            if self.model.check_winner('O'):
                self.game_over = True
                self.message = "O wins!"
            else:
                self.current_player = 'X'
                self.message = f"{self.current_player}'s turn"

    def reset_game(self):
        self.model.reset()
        self.game_over = False
        self.current_player = random.choice(['X', 'O'])
        self.message = f"{self.current_player}'s turn"

    def run(self):
        running = True
        while running:
            self.win.fill(BG_COLOR)
            self.draw_lines()
            self.draw_figures()
            self.draw_status()
            pygame.display.update()

            if self.model.current_winner:
                pygame.time.delay(2000)
                self.reset_game()
            elif not self.model.empty_squares():
                self.message = "It's a tie!"
                self.draw_status()
                pygame.display.update()
                pygame.time.delay(2000)
                self.reset_game()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if not self.game_over:
                    if self.current_player == self.human_player:
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            mouseX, mouseY = event.pos
                            clicked_row = int(mouseY // SQUARE_SIZE)
                            clicked_col = int(mouseX // SQUARE_SIZE)
                            if clicked_row < ROWS and clicked_col < COLS:
                                self.handle_click(clicked_row, clicked_col)
                    if not self.game_over and self.current_player == 'O':
                        pygame.time.delay(500)
                        self.ai_move()

            pygame.display.update()

        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    game = TicTacToeGame()
    game.run()