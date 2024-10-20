#Snakegame.py
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple
import sys
import os
import torch
from .SnakeAgent import Agent
from .SnakeModel import Linear_QNet

# Initialize pygame font
pygame.init()
font = pygame.font.SysFont('arial', 25)

# Direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Point tuple
Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Helper function for resource paths
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Snake Game class
class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # Initialize game state
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - 20, self.head.y),
            Point(self.head.x - (2 * 20), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - 20) // 20) * 20
        y = random.randint(0, (self.h - 20) // 20) * 20
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def capture_frame(self):
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def play_step(self, action=None):
        self.frame_iteration += 1

        # 1. Collect user input (Handled in game loop)

        # 2. Move
        self._move(action)  # Update the head
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(20)

        # 6. Return reward, game over, and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - 20 or pt.x < 0 or pt.y > self.h - 20 or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, 20, 20))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, 20, 20))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action=None):
        if action is not None:
            # [straight, right, left]
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            idx = clock_wise.index(self.direction)

            if np.array_equal(action, [1, 0, 0]):
                # No change
                new_dir = clock_wise[idx]
            elif np.array_equal(action, [0, 1, 0]):
                # Right turn
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]
            else:  # [0, 0, 1]
                # Left turn
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]

            self.direction = new_dir
        # In Player mode, direction is already set by key presses

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 20
        elif self.direction == Direction.LEFT:
            x -= 20
        elif self.direction == Direction.DOWN:
            y += 20
        elif self.direction == Direction.UP:
            y -= 20

        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        # Points around the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # Food left
            self.food.x > self.head.x,  # Food right
            self.food.y < self.head.y,  # Food up
            self.food.y > self.head.y   # Food down
        ]

        return np.array(state, dtype=int)

    def run(self):
        # Display home screen with options
        self.display_home_screen()

        # Wait for user to select mode
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return  # Exit back to main menu
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        return
                    elif event.key == pygame.K_1:
                        self.mode = 'Player'
                        running = False
                    elif event.key == pygame.K_2:
                        self.mode = 'AI'
                        running = False
            self.clock.tick(15)

        # After selection, run the game
        self.game_loop()

    def display_home_screen(self):
        self.display.fill(BLACK)
        title_font = pygame.font.SysFont('Arial', 48)
        text_font = pygame.font.SysFont('Arial', 36)

        title_text = title_font.render('Snake Game', True, WHITE)
        self.display.blit(title_text, (self.w / 2 - title_text.get_width() / 2, self.h / 4))

        option1_text = text_font.render('1. Play Yourself', True, WHITE)
        self.display.blit(option1_text, (self.w / 2 - option1_text.get_width() / 2, self.h / 2))

        option2_text = text_font.render('2. Watch AI Play', True, WHITE)
        self.display.blit(option2_text, (self.w / 2 - option2_text.get_width() / 2, self.h / 2 + 50))

        instruction_text = text_font.render('Press 1 or 2 to select', True, WHITE)
        self.display.blit(instruction_text, (self.w / 2 - instruction_text.get_width() / 2, self.h * 3 / 4))

        pygame.display.flip()

    def game_loop(self):
        # If mode is AI, set up agent and model
        if self.mode == 'AI':
            self.agent = Agent()
            self.model = Linear_QNet(11, 256, 3)
            model_path = resource_path('Models/snake_model_200.pth')
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()

        # Game loop
        self.reset()
        self.clock = pygame.time.Clock()
        running_game = True
        while running_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_game = False
                    return  # Exit back to main menu
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running_game = False
                        return
                    if self.mode == 'Player':
                        if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                            self.direction = Direction.LEFT
                        elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                            self.direction = Direction.RIGHT
                        elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                            self.direction = Direction.UP
                        elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                            self.direction = Direction.DOWN

            if self.mode == 'AI':
                # Get state
                state_old = self.agent.get_state(self)
                # Get action from model
                state_old_tensor = torch.tensor(state_old, dtype=torch.float)
                with torch.no_grad():
                    prediction = self.model(state_old_tensor)
                # Convert prediction to action
                final_move = [0, 0, 0]
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                reward, done, score = self.play_step(final_move)
            else:
                # Player mode
                reward, done, score = self.play_step()

            if done:
                self.reset()

            self.clock.tick(20)
