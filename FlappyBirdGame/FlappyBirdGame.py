# FlappyBirdGame.py
import pygame
import random
import numpy as np
import torch
from .FlappyBirdAgent import FlappyBirdAgent  # Correct Import
import sys
# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_WIDTH = 80
PIPE_GAP = 150
BIRD_SIZE = 24
FPS = 60
GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_SPEED = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

class FlappyBirdGame:
    def __init__(self, screen, render_mode=False):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT / 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.done = False
        self.spawn_pipe()
        self.bird_rect = pygame.Rect(50, int(self.bird_y), BIRD_SIZE, BIRD_SIZE)
        return self.get_state()

    def spawn_pipe(self):
        gap_y = random.randint(100, SCREEN_HEIGHT - 100 - PIPE_GAP)
        pipe = {
            'x': SCREEN_WIDTH,
            'gap_y': gap_y
        }
        self.pipes.append(pipe)

    def step(self, action):
        reward = 0.1  # Small reward for each frame alive

        if action == 1:
            self.bird_vel = FLAP_STRENGTH  # Flap

        self.bird_vel += GRAVITY  # Gravity
        self.bird_y += self.bird_vel
        self.bird_rect.y = int(self.bird_y)

        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= PIPE_SPEED

        # Remove off-screen pipes and add new ones
        if self.pipes and self.pipes[0]['x'] < -PIPE_WIDTH:
            self.pipes.pop(0)
            self.spawn_pipe()
            self.score += 1
            reward += 1  # Reward for passing a pipe

        # Check collision
        self.done = False

        # Check if bird is out of vertical bounds
        if self.bird_y < 0 or self.bird_y > SCREEN_HEIGHT - BIRD_SIZE:
            self.done = True
            reward = -10

        # Update bird's Rect position
        self.bird_rect = pygame.Rect(50, int(self.bird_y), BIRD_SIZE, BIRD_SIZE)

        # Check collision with pipes using Rect.colliderect
        for pipe in self.pipes:
            # Define top and bottom pipe Rects
            top_pipe_rect = pygame.Rect(pipe['x'], 0, PIPE_WIDTH, pipe['gap_y'])
            bottom_pipe_rect = pygame.Rect(pipe['x'], pipe['gap_y'] + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - pipe['gap_y'] - PIPE_GAP)

            if self.bird_rect.colliderect(top_pipe_rect) or self.bird_rect.colliderect(bottom_pipe_rect):
                self.done = True
                reward = -10
                break

        # Additional reward shaping
        if not self.done and self.pipes:
            # Calculate distance to the center of the gap
            gap_center = self.pipes[0]['gap_y'] + PIPE_GAP / 2
            bird_center = self.bird_y + BIRD_SIZE / 2
            distance_to_center = abs(bird_center - gap_center)
            normalized_distance = distance_to_center / (SCREEN_HEIGHT / 2)
            reward += -normalized_distance  # Encourage being closer to the center

        return self.get_state(), reward, self.done

    def get_state(self):
        # State includes bird's y position and velocity, and the closest pipe's x and gap
        closest_pipe = None
        min_distance = float('inf')
        for pipe in self.pipes:
            distance = pipe['x'] - 50
            if distance >= 0 and distance < min_distance:
                min_distance = distance
                closest_pipe = pipe

        if closest_pipe is None and self.pipes:
            closest_pipe = self.pipes[0]

        state = np.array([
            self.bird_y / SCREEN_HEIGHT,               # Normalize bird's y position
            self.bird_vel / 10,                        # Normalize bird's velocity
            (closest_pipe['x'] - 50) / SCREEN_WIDTH if closest_pipe else 1.0,  # Normalize pipe's x position
            closest_pipe['gap_y'] / SCREEN_HEIGHT if closest_pipe else 0.5   # Normalize pipe's gap y position
        ], dtype=np.float32)
        return state

    def render(self):
        if not self.render_mode:
            return

        self.screen.fill(WHITE)

        # Draw bird
        pygame.draw.rect(self.screen, BLACK, self.bird_rect)

        # Draw pipes
        for pipe in self.pipes:
            top_pipe_rect = pygame.Rect(pipe['x'], 0, PIPE_WIDTH, pipe['gap_y'])
            bottom_pipe_rect = pygame.Rect(pipe['x'], pipe['gap_y'] + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - pipe['gap_y'] - PIPE_GAP)
            pygame.draw.rect(self.screen, GREEN, top_pipe_rect)
            pygame.draw.rect(self.screen, GREEN, bottom_pipe_rect)

        # Draw score
        font = pygame.font.SysFont(None, 36)
        img = font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(img, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def display_home_screen(self):
        self.screen.fill(WHITE)
        title_font = pygame.font.SysFont('Arial', 48)
        text_font = pygame.font.SysFont('Arial', 36)

        title_text = title_font.render('Flappy Bird', True, BLACK)
        self.screen.blit(title_text, (SCREEN_WIDTH / 2 - title_text.get_width() / 2, SCREEN_HEIGHT / 4))

        option1_text = text_font.render('1. Play Yourself', True, BLACK)
        self.screen.blit(option1_text, (SCREEN_WIDTH / 2 - option1_text.get_width() / 2, SCREEN_HEIGHT / 2))

        option2_text = text_font.render('2. Watch AI Play', True, BLACK)
        self.screen.blit(option2_text, (SCREEN_WIDTH / 2 - option2_text.get_width() / 2, SCREEN_HEIGHT / 2 + 50))

        instruction_text = text_font.render('Press 1 or 2 to select', True, BLACK)
        self.screen.blit(instruction_text, (SCREEN_WIDTH / 2 - instruction_text.get_width() / 2, SCREEN_HEIGHT * 3 / 4))

        pygame.display.flip()

    def run_player(self):
        running_game = True
        while running_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running_game = False
                        return
                    elif event.key == pygame.K_SPACE:
                        self.bird_vel = FLAP_STRENGTH

            # Player control
            action = 1 if pygame.key.get_pressed()[pygame.K_SPACE] else 0
            _, _, done = self.step(action)
            self.render()

            if done:
                self.reset()

    def run_ai(self, policy_net):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_game = True
        while running_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_game = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running_game = False
                        return

            state = self.get_state()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

            _, _, done = self.step(action)
            self.render()

            if done:
                self.reset()

    def run(self, policy_net=None):
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
                        mode = 'Player'
                        running = False
                    elif event.key == pygame.K_2:
                        mode = 'AI'
                        running = False
            self.clock.tick(15)

        self.reset()

        if mode == 'AI' and policy_net:
            self.run_ai(policy_net)
        else:
            self.run_player()

        self.reset()

        if mode == 'AI' and policy_net:
            self.run_ai(policy_net)
        else:
            self.run_player()
