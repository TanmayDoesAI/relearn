# FlappyBirdTrain.py
import numpy as np
from FlappyBirdGame import FlappyBirdGame
from FlappyBirdAgent import FlappyBirdAgent
import torch
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

def train():
    num_episodes = 2000
    render_every = 100  # Render every 100 episodes
    target_update_freq = 1000  # Steps
    model_save_path = "flappybird_dqn.pth"
    final_model_save_path = "flappybird_dqn_final.pth"

    env = FlappyBirdGame(render_mode=False)
    state_size = 4  # [bird_y, bird_vel, pipe_x, pipe_gap_y]
    action_size = 2  # [do nothing, flap]
    agent = FlappyBirdAgent(state_size, action_size,
                            lr=5e-4, gamma=0.99,  # Adjusted learning rate
                            epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=1000,  # Adjusted epsilon_decay
                            batch_size=64, memory_size=10000, target_update_freq=target_update_freq)

    scores = []
    losses = []
    best_score = -float('inf')  # Initialize to negative infinity
    total_steps = 0

    # Early stopping parameters
    patience = 300  # Number of episodes to wait before stopping
    best_avg_score = -float('inf')
    patience_counter = 0

    try:
        for episode in tqdm(range(1, num_episodes + 1), desc="Training Episodes"):
            state = env.reset()
            total_reward = 0
            done = False
            episode_loss = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    episode_loss += loss
                state = next_state
                total_reward += reward
                total_steps += 1

                # Update target network
                if total_steps % agent.target_update_freq == 0:
                    agent.update_target_network()

            scores.append(env.score)

            # Save the model after each episode
            agent.save_model(model_save_path)

            # Update best score
            if env.score > best_score:
                best_score = env.score
                print(f"New best score: {best_score} at episode {episode}")
                # Save the best model
                agent.save_model(final_model_save_path)

            # Logging
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_loss = episode_loss / 10 if len(scores) >= 10 else episode_loss / len(scores)
                losses.append(avg_loss)
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Best Score: {best_score}, Epsilon: {agent.epsilon:.2f}, Avg Loss: {avg_loss:.4f}")

                # Early Stopping Check
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at episode {episode} due to no improvement in average score.")
                    break

            # Render the game every 'render_every' episodes
            if episode % render_every == 0:
                print(f"Rendering Episode {episode}")
                render_episode(env, agent, episode, final_model_save_path)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    env.close()

    # Save final model
    agent.save_model(final_model_save_path)
    print("Training completed. Models saved.")

    # Plot training progress
    plot_training_progress(scores, losses)

def render_episode(env, agent, episode, model_path):
    env.render_mode = True

    # Check if model file exists
    if os.path.exists(model_path):
        agent.load_model(model_path)
    else:
        print(f"Model file {model_path} not found. Skipping rendering.")
        env.render_mode = False
        return

    state = env.reset()
    done = False
    episode_score = 0

    while not done:
        env.render()
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        episode_score += 1  # Increment score based on frames survived

    print(f"Episode {episode} finished with score: {episode_score}")
    env.render_mode = False

def plot_training_progress(scores, losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Scores")
    plt.plot(scores, label='Score per Episode')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.plot(losses, label='Avg Loss per 10 Episodes')
    plt.xlabel("Episode (x10)")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
