import json
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .replay_buffer import ReplayBuffer
from Simulation.Utils.preprocessing import preprocess_state
import random


class DQN(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._to_linear = 49
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #print(np.prod(x.shape[1:]))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNModel:
    def __init__(self, input_channels, action_size, env, device):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        #self.state_size = state_size
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
        self.batch_size = self.params['BATCH_SIZE']
        self.input_channels = input_channels * self.batch_size
        self.replay_buffer_our_agent = ReplayBuffer(self.buffer_size)
        self.replay_buffer_opponent_agent = ReplayBuffer(self.buffer_size)

        self.main_network_our_agent = DQN(self.input_channels, self.action_size).to(device=device)
        self.target_network_our_agent = DQN(self.input_channels, self.action_size).to(device=device)
        self.target_network_our_agent.load_state_dict(self.main_network_our_agent.state_dict())
        self.our_agent_loss_fn = nn.MSELoss()
        self.our_agent_optimizer = torch.optim.Adam(self.main_network_our_agent.parameters(), lr=self.alpha)

        self.main_network_opponent_agent = DQN(self.input_channels, self.action_size).to(device=device)
        self.target_network_opponent_agent = DQN(self.input_channels, self.action_size).to(device=device)
        self.target_network_opponent_agent.load_state_dict(self.main_network_opponent_agent.state_dict())
        self.opponent_agent_loss_fn = nn.MSELoss()
        self.opponent_agent_optimizer = torch.optim.Adam(self.main_network_opponent_agent.parameters(), lr=self.alpha)

    def remember(self, observations, actions, rewards, next_observations, terminations):
        for agent in self.env.agents:
            state = np.array(observations[agent])
            next_state = np.array(next_observations[agent])
            if agent == self.env.agents[0]:
                self.replay_buffer_our_agent.append((state, actions[agent], rewards[agent],
                                                     next_state, terminations[agent]))
            else:
                self.replay_buffer_opponent_agent.append((state, actions[agent], rewards[agent],
                                                     next_state, terminations[agent]))

    def act(self, state, agent):
        exploration = (np.random.uniform(0, 1) < self.epsilon)
        state_stack = state * self.batch_size
        if agent == self.env.agents[0]:
            if exploration:
                action = self.env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = self.main_network_our_agent(torch.tensor(state_stack, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()
        else:
            if exploration:
                action = self.env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = self.main_network_opponent_agent(torch.tensor(state_stack, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()
        return action

    def update_target_network(self):
        self.target_network_our_agent.load_state_dict(self.main_network_our_agent.state_dict())
        self.target_network_opponent_agent.load_state_dict(self.main_network_opponent_agent.state_dict())

    def train_our_agent(self, batch_size):
        minibatch = self.replay_buffer_our_agent.sample(batch_size)
        predicted_Q_values, target_Q_values = [], []
        for state_frame, action, reward, new_state_frame, done in minibatch:
            state_frame_stack = np.array([preprocess_state(state_frame)] * self.batch_size, dtype=np.float32)
            new_state_frame_stack = np.array([preprocess_state(new_state_frame)] * self.batch_size, dtype=np.float32)
            state = torch.tensor(state_frame_stack, dtype=torch.float32, device=self.device)
            reward = torch.tensor([reward], device=self.device)
            new_state = torch.tensor(new_state_frame_stack, dtype=torch.float32, device=self.device)

            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * self.target_network_our_agent(new_state).max()
            else:
                target = reward
            target_Q = self.main_network_our_agent(state)
            target_Q[action] = target

            predicted_Q = self.main_network_our_agent(state)

            predicted_Q_values.append(predicted_Q)
            target_Q_values.append(target_Q)

        loss = self.our_agent_loss_fn(torch.stack(predicted_Q_values), torch.stack(target_Q_values))

        self.our_agent_optimizer.zero_grad()
        loss.backward()
        self.our_agent_optimizer.step()
        return loss

    def train_opponent_agent(self, batch_size):
        minibatch = self.replay_buffer_opponent_agent.sample(batch_size)
        predicted_Q_values, target_Q_values = [], []
        for state_frame, action, reward, new_state_frame, done in minibatch:
            state_frame_stack = np.array([preprocess_state(state_frame)] * self.batch_size, dtype=np.float32)
            new_state_frame_stack = np.array([preprocess_state(new_state_frame)] * self.batch_size, dtype=np.float32)
            state = torch.tensor(state_frame_stack, dtype=torch.float32, device=self.device)
            reward = torch.tensor([reward], device=self.device)
            new_state = torch.tensor(new_state_frame_stack, dtype=torch.float32, device=self.device)

            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * self.target_network_opponent_agent(new_state).max()
            else:
                target = reward
            target_Q = self.main_network_opponent_agent(state)
            target_Q[action] = target

            predicted_Q = self.main_network_opponent_agent(state)

            predicted_Q_values.append(predicted_Q)
            target_Q_values.append(target_Q)

        loss = self.opponent_agent_loss_fn(torch.stack(predicted_Q_values), torch.stack(target_Q_values))

        self.opponent_agent_optimizer.zero_grad()
        loss.backward()
        self.opponent_agent_optimizer.step()
        return loss
