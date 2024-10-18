#SnakeAgent.py
import torch
import random
import numpy as np
from collections import deque
from .SnakeModel import Linear_QNet, QTrainer

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Agent class
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Exploration rate
        self.gamma = 0.9  # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)  # Popleft when exceeding max memory
        self.model = Linear_QNet(11, 256, 3)  # 11 inputs, 256 hidden units, 3 outputs
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        state = game.get_state()
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: exploration vs exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Prediction from the model
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
