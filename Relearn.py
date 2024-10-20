#Relearn.py
import pygame
import sys
import torch
import pygame_gui
import os

# Helper function to get the correct path for resources when bundled by PyInstaller
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Import existing games
from SnakeGame.SnakeGame import SnakeGame
from TicTacToeGame.TicTacToeGame import TicTacToeGame
from FlappyBirdGame.FlappyBirdGame import FlappyBirdGame
from FlappyBirdGame.FlappyBirdAgent import FlappyBirdAgent, DuelingDQN  # Updated import

# Initialize Pygame
pygame.init()

# Set up the display
MAIN_SCREEN_WIDTH = 800
MAIN_SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((MAIN_SCREEN_WIDTH, MAIN_SCREEN_HEIGHT))
pygame.display.set_caption("ReLearn: AI Game Platform")

# Colors
WHITE = (255, 255, 255)
BG_COLOR = (30, 30, 30)  # Dark background for sleek look

# Setup pygame_gui
manager = pygame_gui.UIManager((MAIN_SCREEN_WIDTH, MAIN_SCREEN_HEIGHT))

# Load Icons using resource_path to ensure compatibility with PyInstaller
try:
    snake_icon = pygame.image.load(resource_path("Icons/Snake.png"))
    snake_icon = pygame.transform.scale(snake_icon, (150, 150))
except:
    # Placeholder if image not found
    snake_icon = pygame.Surface((150, 150))
    snake_icon.fill((0, 255, 0))  # Green for Snake

try:
    tictactoe_icon = pygame.image.load(resource_path("Icons/TicTacToe.png"))
    tictactoe_icon = pygame.transform.scale(tictactoe_icon, (150, 150))
except:
    # Placeholder if image not found
    tictactoe_icon = pygame.Surface((150, 150))
    tictactoe_icon.fill((255, 0, 0))  # Red for Tic Tac Toe

try:
    flappybird_icon = pygame.image.load(resource_path("Icons/Flappy.png"))
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
    title_rect = title_surf.get_rect(center=(MAIN_SCREEN_WIDTH//2, 100))
    screen.blit(title_surf, title_rect)
    
    # Draw Icons
    screen.blit(snake_icon, (100, 200))
    screen.blit(tictactoe_icon, (325, 200))
    screen.blit(flappybird_icon, (550, 200))

def run_snake_game():
    global running_main_menu
    running_main_menu = False

    # Set display for Snake Game
    GAME_SCREEN_WIDTH = 640
    GAME_SCREEN_HEIGHT = 480
    pygame.display.set_mode((GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Game")

    game = SnakeGame(w=GAME_SCREEN_WIDTH, h=GAME_SCREEN_HEIGHT)
    game.run()

    # After game ends, reset display to main menu's size and title
    pygame.display.set_mode((MAIN_SCREEN_WIDTH, MAIN_SCREEN_HEIGHT))
    pygame.display.set_caption("ReLearn: AI Game Platform")
    running_main_menu = True

def run_tictactoe_game():
    global running_main_menu
    running_main_menu = False

    # Set display for Tic Tac Toe Game
    GAME_SCREEN_WIDTH = 600
    GAME_SCREEN_HEIGHT = 700
    pygame.display.set_mode((GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT))
    pygame.display.set_caption("Tic Tac Toe")

    game = TicTacToeGame(screen=pygame.display.get_surface())
    game.run()

    # After game ends, reset display to main menu's size and title
    pygame.display.set_mode((MAIN_SCREEN_WIDTH, MAIN_SCREEN_HEIGHT))
    pygame.display.set_caption("ReLearn: AI Game Platform")
    running_main_menu = True

def run_flappybird_game():
    global running_main_menu
    running_main_menu = False

    # Set display for Flappy Bird Game
    GAME_SCREEN_WIDTH = 400
    GAME_SCREEN_HEIGHT = 600
    pygame.display.set_mode((GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT))
    pygame.display.set_caption("Flappy Bird RL")

    game = FlappyBirdGame(screen=pygame.display.get_surface(), render_mode=True)
    model_path = resource_path('Models/flappybird_dqn_final.pth')
    
    state_size = 4
    action_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DuelingDQN(state_size, action_size).to(device)
    if os.path.exists(model_path):
        # Corrected model loading code
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model file {model_path} not found.")
        # Reset display to main menu's size and title
        pygame.display.set_mode((MAIN_SCREEN_WIDTH, MAIN_SCREEN_HEIGHT))
        pygame.display.set_caption("ReLearn: AI Game Platform")
        running_main_menu = True
        return

    policy_net.eval()
    
    game.run(policy_net=policy_net)

    # After game ends, reset display to main menu's size and title
    pygame.display.set_mode((MAIN_SCREEN_WIDTH, MAIN_SCREEN_HEIGHT))
    pygame.display.set_caption("ReLearn: AI Game Platform")
    running_main_menu = True

def main():
    global running_main_menu
    running_main_menu = True
    clock = pygame.time.Clock()

    while True:
        if running_main_menu:
            # Reset display to main menu's size and title
            pygame.display.set_mode((MAIN_SCREEN_WIDTH, MAIN_SCREEN_HEIGHT))
            pygame.display.set_caption("ReLearn: AI Game Platform")
            running_main_menu = False

        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_object_id == '#snake_button':
                        run_snake_game()
                    elif event.ui_object_id == '#tictactoe_button':
                        run_tictactoe_game()
                    elif event.ui_object_id == '#flappybird_button':
                        run_flappybird_game()

            manager.process_events(event)

        manager.update(time_delta)
        draw_main_menu()
        manager.draw_ui(screen)

        pygame.display.update()

if __name__ == "__main__":
    main()
