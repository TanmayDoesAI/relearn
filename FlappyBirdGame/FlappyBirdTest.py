# FlappyBirdEvaluate.py
import torch
from FlappyBirdGame import FlappyBirdGame
from FlappyBirdAgent import DuelingDQN
import pygame

def evaluate(model_path, num_episodes=10):
    env = FlappyBirdGame(render_mode=True)
    state_size = 4
    action_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DuelingDQN(state_size, action_size).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    policy_net.eval()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            state, reward, done = env.step(action)
        print(f"Episode {episode} finished with score: {env.score}")

    env.close()

if __name__ == "__main__":
    evaluate("flappybird_dqn_final.pth", num_episodes=5)
