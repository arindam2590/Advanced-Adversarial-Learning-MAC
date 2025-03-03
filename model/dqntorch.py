import json
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import deque
import random


# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class DQNModel:
    def __init__(self, state_size, action_size, env, device):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.env = env

        self.gamma = self.params['GAMMA']
        self.alpha = self.params['ALPHA']
        self.epsilon = self.params['EPSILON']
        self.epsilon_min = self.params['EPSILON_MIN']
        self.epsilon_decay = self.params['EPSILON_DECAY']
        self.update_rate = self.params['UPDATE_RATE']
        self.buffer_size = self.params['BUFFER_SIZE']
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.main_network_our_agent = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network_our_agent = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network_our_agent.load_state_dict(self.main_network_our_agent.state_dict())
        self.our_agent_loss_fn = nn.MSELoss()
        self.our_agent_optimizer = torch.optim.Adam(self.main_network_our_agent.parameters(), lr=self.alpha)

        self.main_network_opponent_agent = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network_opponent_agent = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network_opponent_agent.load_state_dict(self.main_network_opponent_agent.state_dict())
        self.opponent_agent_loss_fn = nn.MSELoss()
        self.opponent_agent_optimizer = torch.optim.Adam(self.main_network_opponent_agent.parameters(), lr=self.alpha)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, agent):
        if agent == self.env.agents[0]:
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = self.main_network_our_agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()
            return action
        else:
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = self.main_network_opponent_agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()

    def update_target_network(self):
        self.target_network_our_agent.load_state_dict(self.main_network_our_agent.state_dict())
        self.target_network_opponent_agent.load_state_dict(self.main_network_opponent_agent.state_dict())