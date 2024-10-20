# FlappyBirdAgent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple

# Define a named tuple for experience replay
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()

        # Value stream
        self.fc_value = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)

        # Advantage stream
        self.fc_advantage = nn.Linear(256, 256)
        self.advantage = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))

        # Value stream
        val = self.relu(self.fc_value(x))
        val = self.value(val)

        # Advantage stream
        adv = self.relu(self.fc_advantage(x))
        adv = self.advantage(adv)

        # Combine streams
        q = val + adv - adv.mean()
        return q

class FlappyBirdAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 500

        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        self.target_update_freq = 1000
        self.steps_done = 0

        # PER parameters (if using Prioritized Experience Replay)
        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta_frames = 100000
        self.frame = 1

        self.device = device
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        # Initialize priority for new experiences
        self.max_priority = 1.0

    def select_action(self, state):
        self.steps_done += 1
        # Epsilon decay
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        # Store with maximum priority to ensure it gets sampled at least once
        self.memory.append((self.max_priority, Experience(state, action, reward, next_state, done)))

    def sample_memory(self):
        if len(self.memory) == 0:
            return [], [], []

        # Calculate priorities
        priorities = np.array([priority for priority, _ in self.memory], dtype=np.float32)
        if priorities.sum() == 0:
            probabilities = np.ones(len(self.memory)) / len(self.memory)
        else:
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        experiences = [self.memory[idx][1] for idx in indices]

        # Calculate importance-sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        return indices, experiences, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            # Update priority
            self.memory[idx] = (priority, self.memory[idx][1])
            # Update max priority
            self.max_priority = max(self.max_priority, priority)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        indices, experiences, weights = self.sample_memory()
        if not experiences:
            return None

        batch = Experience(*zip(*experiences))

        states = torch.FloatTensor(batch.state).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch.next_state).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Double DQN: action selection is from policy_net, evaluation from target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Calculate TD error for updating priorities
        td_errors = target_q - current_q
        loss = (td_errors.pow(2) * weights).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        td_errors_abs = td_errors.abs().detach().cpu().numpy().flatten()
        new_priorities = td_errors_abs + 1e-6  # Small constant to avoid zero priority
        self.update_priorities(indices, new_priorities)

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
