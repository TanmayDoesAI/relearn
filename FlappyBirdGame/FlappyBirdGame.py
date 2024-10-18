# FlappyBirdGame.py
import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_WIDTH = 80
PIPE_GAP = 150
BIRD_SIZE = 24  # Reduced from 30 to 24 for better hitbox accuracy
FPS = 60  # Increased FPS for smoother gameplay
GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_SPEED = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

class FlappyBirdGame:
    def __init__(self, render_mode=False):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird RL")
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
        # Initialize bird's Rect with reduced size
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

        # Apply action
        if action == 1:
            self.bird_vel = FLAP_STRENGTH  # Flap

        # Update bird
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

        # Draw hitbox for bird (for debugging)
        # pygame.draw.rect(self.screen, (255, 0, 0), self.bird_rect, 2)  # Uncomment for debugging

        # Draw pipes
        for pipe in self.pipes:
            top_pipe_rect = pygame.Rect(pipe['x'], 0, PIPE_WIDTH, pipe['gap_y'])
            bottom_pipe_rect = pygame.Rect(pipe['x'], pipe['gap_y'] + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT - pipe['gap_y'] - PIPE_GAP)
            pygame.draw.rect(self.screen, GREEN, top_pipe_rect)
            pygame.draw.rect(self.screen, GREEN, bottom_pipe_rect)

            # Draw hitboxes for pipes (for debugging)
            # pygame.draw.rect(self.screen, (0, 0, 255), top_pipe_rect, 2)    # Uncomment for debugging
            # pygame.draw.rect(self.screen, (0, 0, 255), bottom_pipe_rect, 2) # Uncomment for debugging

        # Draw score
        font = pygame.font.SysFont(None, 36)
        img = font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(img, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()

# For testing the game environment
if __name__ == "__main__":
    game = FlappyBirdGame(render_mode=True)
    state = game.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = random.choice([0, 1])  # Random action
        state, reward, done = game.step(action)
        game.render()
        if done:
            print("Game Over! Score:", game.score)
            state = game.reset()

    game.close()
