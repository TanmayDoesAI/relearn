# relearn.py
import pygame
import sys
import torch
import pygame_gui
import os

# Import existing games
from SnakeGame.SnakeGame import SnakeGame
from SnakeGame.SnakeAgent import Agent
from SnakeGame.SnakeModel import Linear_QNet
from TicTacToeGame.TicTacToeGame import TicTacToeGame
from TicTacToeGame.TicTacToeAgent import TicTacToeAgent

# Import Flappy Bird game and agent
from FlappyBirdGame.FlappyBirdGame import FlappyBirdGame
from FlappyBirdGame.FlappyBirdAgent import DuelingDQN

# Initialize Pygame
pygame.init()

# Set up the display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ReLearn: AI Game Platform")

# Colors
WHITE = (255, 255, 255)
BG_COLOR = (30, 30, 30)  # Dark background for sleek look

# Setup pygame_gui
manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

# Load Icons
try:
    snake_icon = pygame.image.load("Icons/Snake.png")
    snake_icon = pygame.transform.scale(snake_icon, (150, 150))
except:
    # Placeholder if image not found
    snake_icon = pygame.Surface((150, 150))
    snake_icon.fill((0, 255, 0))  # Green for Snake

try:
    tictactoe_icon = pygame.image.load("Icons/TicTacToe.png")
    tictactoe_icon = pygame.transform.scale(tictactoe_icon, (150, 150))
except:
    # Placeholder if image not found
    tictactoe_icon = pygame.Surface((150, 150))
    tictactoe_icon.fill((255, 0, 0))  # Red for Tic Tac Toe

try:
    flappybird_icon = pygame.image.load("Icons/Flappy.png")
    flappybird_icon = pygame.transform.scale(flappybird_icon, (150, 150))
except:
    # Placeholder if image not found
    flappybird_icon = pygame.Surface((150, 150))
    flappybird_icon.fill((255, 255, 0))  # Yellow for Flappy Bird

# Create Buttons
snake_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((100, 400), (150, 50)),
    text='Play Snake',
    manager=manager,
    object_id="#snake_button"
)

tictactoe_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((325, 400), (150, 50)),
    text='Play Tic Tac Toe',
    manager=manager,
    object_id="#tictactoe_button"
)

flappybird_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((550, 400), (150, 50)),
    text='Play Flappy Bird',
    manager=manager,
    object_id="#flappybird_button"
)

# Font for Titles
title_font = pygame.font.SysFont('Arial', 48)

def draw_main_menu():
    screen.fill(BG_COLOR)
    
    # Draw Titles
    title_surf = title_font.render("ReLearn: AI Game Platform", True, WHITE)
    title_rect = title_surf.get_rect(center=(SCREEN_WIDTH//2, 100))
    screen.blit(title_surf, title_rect)
    
    # Draw Icons
    screen.blit(snake_icon, (100, 200))
    screen.blit(tictactoe_icon, (325, 200))
    screen.blit(flappybird_icon, (550, 200))

def run_snake_game():
    # Close the main menu UI
    global running_main_menu
    running_main_menu = False
    
    game = SnakeGame(w=640, h=480)
    agent = Agent()
    # Load the model
    model = Linear_QNet(11, 256, 3)
    model.load_state_dict(torch.load('Models/snake_model_200.pth'))
    model.eval()
    clock = pygame.time.Clock()

    running_game = True
    while running_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running_game = False

        state_old = agent.get_state(game)
        # Get model's prediction
        state_old_tensor = torch.tensor(state_old, dtype=torch.float)
        with torch.no_grad():
            prediction = model(state_old_tensor)
        # Convert prediction to action
        final_move = [0, 0, 0]
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)

        clock.tick(20)

        if done:
            game.reset()

def run_tictactoe_game():
    # Close the main menu UI
    global running_main_menu
    running_main_menu = False
    
    game = TicTacToeGame()
    game.run()

def run_flappybird_game():
    global running_main_menu
    running_main_menu = False
    
    model_path = 'Models/flappybird_dqn_final.pth'
    env = FlappyBirdGame(render_mode=True)
    state_size = 4
    action_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DuelingDQN(state_size, action_size).to(device)
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model file {model_path} not found.")
        return
    policy_net.eval()

    running_game = True
    while running_game:
        state = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running_game = False
                        done = True
            env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            state, reward, done = env.step(action)
        print(f"Game finished with score: {env.score}")
        # Let the user play again unless they press escape
        if not running_game:
            break

    env.close()

def main():
    global running_main_menu
    running_main_menu = True
    clock = pygame.time.Clock()

    while running_main_menu:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_object_id == '#snake_button':
                        run_snake_game()
                        running_main_menu = True  # Reopen main menu after game
                    elif event.ui_object_id == '#tictactoe_button':
                        run_tictactoe_game()
                        running_main_menu = True  # Reopen main menu after game
                    elif event.ui_object_id == '#flappybird_button':
                        run_flappybird_game()
                        running_main_menu = True  # Reopen main menu after game

            manager.process_events(event)

        manager.update(time_delta)
        draw_main_menu()
        manager.draw_ui(screen)

        pygame.display.update()

if __name__ == "__main__":
    main()

