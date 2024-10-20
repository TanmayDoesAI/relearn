#TicTacToeGame.py
import pygame
import sys
import random
from .TicTacToeAgent import TicTacToeAgent
from .TicTacToeModel import TicTacToeModel, EMPTY, X, O

# Constants
DEFAULT_WIDTH, DEFAULT_HEIGHT = 600, 700
ROWS, COLS = 3, 3
SQUARE_SIZE = DEFAULT_WIDTH // COLS
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
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 40)
        self.model = TicTacToeModel()
        self.ai_player = TicTacToeAgent('O')
        self.human_player = 'X'
        self.current_player = random.choice(['X', 'O'])
        self.game_over = False
        self.message = f"{self.current_player}'s turn"
        self.clock = pygame.time.Clock()  # Added clock for consistent timing

    def draw_lines(self):
        for i in range(1, ROWS):
            pygame.draw.line(self.screen, LINE_COLOR, (0, i * SQUARE_SIZE), 
                             (DEFAULT_WIDTH, i * SQUARE_SIZE), LINE_WIDTH)
        for i in range(1, COLS):
            pygame.draw.line(self.screen, LINE_COLOR, (i * SQUARE_SIZE, 0), 
                             (i * SQUARE_SIZE, DEFAULT_HEIGHT - 100), LINE_WIDTH)

    def draw_figures(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.model.board[row][col] == 'X':
                    start_desc = (col * SQUARE_SIZE + SQUARE_SIZE // 5, 
                                  row * SQUARE_SIZE + SQUARE_SIZE // 5)
                    end_desc = (col * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5, 
                                row * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5)
                    pygame.draw.line(self.screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH)
                    
                    start_asc = (col * SQUARE_SIZE + SQUARE_SIZE // 5, 
                                 row * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5)
                    end_asc = (col * SQUARE_SIZE + SQUARE_SIZE - SQUARE_SIZE // 5, 
                               row * SQUARE_SIZE + SQUARE_SIZE // 5)
                    pygame.draw.line(self.screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)
                elif self.model.board[row][col] == 'O':
                    center = (col * SQUARE_SIZE + SQUARE_SIZE // 2, 
                              row * SQUARE_SIZE + SQUARE_SIZE // 2)
                    pygame.draw.circle(self.screen, CIRCLE_COLOR, center, CIRCLE_RADIUS, CIRCLE_WIDTH)

    def draw_status(self):
        pygame.draw.rect(self.screen, BG_COLOR, (0, DEFAULT_HEIGHT - 100, DEFAULT_WIDTH, 100))
        text = self.font.render(self.message, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(DEFAULT_WIDTH/2, DEFAULT_HEIGHT - 50))
        self.screen.blit(text, text_rect)

    def handle_click(self, row, col, player):
        if self.model.available_square(row, col):
            self.model.mark_square(row, col, player)
            if self.model.check_winner(player):
                self.game_over = True
                self.message = f"{player} wins!"
            elif not self.model.empty_squares():
                self.game_over = True
                self.message = "It's a tie!"
            else:
                # Switch to the other player
                self.current_player = 'O' if player == 'X' else 'X'
                self.message = f"{self.current_player}'s turn"

    def ai_move(self):
        move = self.ai_player.get_move(self.model)
        if move:
            self.model.mark_square(move[0], move[1], 'O')
            if self.model.check_winner('O'):
                self.game_over = True
                self.message = "O wins!"
            elif not self.model.empty_squares():
                self.game_over = True
                self.message = "It's a tie!"
            else:
                self.current_player = 'X'
                self.message = f"{self.current_player}'s turn"

    def reset_game(self):
        self.model.reset()
        self.game_over = False
        self.current_player = random.choice(['X', 'O'])
        self.message = f"{self.current_player}'s turn"

    def display_home_screen(self):
        self.screen.fill(BG_COLOR)
        title_font = pygame.font.SysFont('Arial', 48)
        text_font = pygame.font.SysFont('Arial', 36)

        title_text = title_font.render('Tic Tac Toe', True, TEXT_COLOR)
        self.screen.blit(title_text, (DEFAULT_WIDTH / 2 - title_text.get_width() / 2, DEFAULT_HEIGHT / 4))

        option1_text = text_font.render('1. Two Humans Play', True, TEXT_COLOR)
        self.screen.blit(option1_text, (DEFAULT_WIDTH / 2 - option1_text.get_width() / 2, DEFAULT_HEIGHT / 2))

        option2_text = text_font.render('2. Play Against AI', True, TEXT_COLOR)
        self.screen.blit(option2_text, (DEFAULT_WIDTH / 2 - option2_text.get_width() / 2, DEFAULT_HEIGHT / 2 + 50))

        instruction_text = text_font.render('Press 1 or 2 to select', True, TEXT_COLOR)
        self.screen.blit(instruction_text, (DEFAULT_WIDTH / 2 - instruction_text.get_width() / 2, DEFAULT_HEIGHT * 3 / 4))

        pygame.display.flip()

    def run_human_vs_human(self):
        running_game = True
        while running_game:
            self.screen.fill(BG_COLOR)
            self.draw_lines()
            self.draw_figures()
            self.draw_status()
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running_game = False
                        return
                if not self.game_over:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouseX, mouseY = event.pos
                        clicked_row = int(mouseY // SQUARE_SIZE)
                        clicked_col = int(mouseX // SQUARE_SIZE)
                        if clicked_row < ROWS and clicked_col < COLS:
                            self.handle_click(clicked_row, clicked_col, self.current_player)

            if self.game_over:
                pygame.time.delay(2000)
                self.reset_game()

            self.clock.tick(30)  # Control the frame rate

    def run_human_vs_ai(self):
        running_game = True
        while running_game:
            self.screen.fill(BG_COLOR)
            self.draw_lines()
            self.draw_figures()
            self.draw_status()
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running_game = False
                        return
                if not self.game_over:
                    if self.current_player == self.human_player and event.type == pygame.MOUSEBUTTONDOWN:
                        mouseX, mouseY = event.pos
                        clicked_row = int(mouseY // SQUARE_SIZE)
                        clicked_col = int(mouseX // SQUARE_SIZE)
                        if clicked_row < ROWS and clicked_col < COLS:
                            self.handle_click(clicked_row, clicked_col, self.human_player)

            if not self.game_over and self.current_player == 'O':
                pygame.time.delay(500)
                self.ai_move()

            if self.game_over:
                pygame.time.delay(2000)
                self.reset_game()

            self.clock.tick(30)  # Control the frame rate

    def run(self):
        # Display home screen with options
        self.display_home_screen()

        running = True
        mode = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        return
                    elif event.key == pygame.K_1:
                        mode = 'HumanVsHuman'
                        running = False
                    elif event.key == pygame.K_2:
                        mode = 'HumanVsAI'
                        running = False
            self.clock.tick(15)

        self.reset_game()

        if mode == 'HumanVsAI':
            self.run_human_vs_ai()
        else:
            self.run_human_vs_human()
