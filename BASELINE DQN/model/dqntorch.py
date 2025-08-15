import json
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .replay_buffer import ReplayBuffer
from .preprocessing import preprocess_state, FrameStack
import random


class DQN(nn.Module):
    def __init__(self, in_channels, action_size, device):
        super(DQN, self).__init__()
        self.device = device
        self.conv_layer1 = self.conv_block(in_channels, 32)
        self.conv_layer2 = self.conv_block(32, 64)
        self.conv_layer3 = self.conv_block(64, 64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self._to_linear = 64 * 10 * 10  
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, action_size)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
            
    def forward(self, x):
        x = x.to(self.device)
        
        if x.device.type == 'cuda':
            torch.cuda.empty_cache()

        x = F.relu(self.conv_layer1(x))
        x = self.pool(x)  
        
        x = F.relu(self.conv_layer2(x))
        x = self.pool(x)  
        
        x = F.relu(self.conv_layer3(x))
        x = self.pool(x)  
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNModel:
    def __init__(self, in_channels, action_size, env, device, frame_stack):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.input_channels = in_channels
        self.action_size = action_size
        self.device = device
        self.env = env
        self.frame_stack = frame_stack

        self.gamma = self.params['GAMMA']
        self.alpha = self.params['ALPHA']
        self.epsilon = self.params['EPSILON']
        self.epsilon_min = self.params['EPSILON_MIN']
        self.epsilon_decay = self.params['EPSILON_DECAY']
        self.update_rate = self.params['UPDATE_RATE']
        self.buffer_size = self.params['BUFFER_SIZE']
        
        self.batch_size = min(self.params['BATCH_SIZE'], 16)  
        
        self.replay_buffer_our_agent = ReplayBuffer(self.buffer_size)
        self.replay_buffer_opponent_agent = ReplayBuffer(self.buffer_size)

        self.main_network_our_agent = DQN(self.input_channels, self.action_size, self.device).to(device=self.device)
        self.target_network_our_agent = DQN(self.input_channels, self.action_size, self.device).to(device=self.device)
        self.target_network_our_agent.load_state_dict(self.main_network_our_agent.state_dict())
        self.our_agent_loss_fn = nn.MSELoss()
        self.our_agent_optimizer = torch.optim.Adam(self.main_network_our_agent.parameters(), lr=self.alpha)

        self.main_network_opponent_agent = DQN(self.input_channels, self.action_size, self.device).to(device=self.device)
        self.target_network_opponent_agent = DQN(self.input_channels, self.action_size, self.device).to(device=self.device)
        self.target_network_opponent_agent.load_state_dict(self.main_network_opponent_agent.state_dict())
        self.opponent_agent_loss_fn = nn.MSELoss()
        self.opponent_agent_optimizer = torch.optim.Adam(self.main_network_opponent_agent.parameters(), lr=self.alpha)

        total_params = sum(p.numel() for p in self.main_network_our_agent.parameters())
        print(f"Info: Total parameters per network: {total_params:,}")

    def remember(self, observations, actions, rewards, next_observations, terminations):
        for agent in self.env.agents:            
            if agent == self.env.agents[0]:
                self.replay_buffer_our_agent.append((observations[agent], actions[agent], rewards[agent],
                                                     next_observations[agent], terminations[agent]))
            else:
                self.replay_buffer_opponent_agent.append((observations[agent], actions[agent], rewards[agent],
                                                     next_observations[agent], terminations[agent]))

    def act(self, state, agent):
        exploration = (np.random.uniform(0, 1) < self.epsilon)
        
        if agent == self.env.agents[0]:
            if exploration:
                action = self.env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = self.main_network_our_agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()
        else:
            if exploration:
                action = self.env.action_space(agent).sample()
            else:
                with torch.no_grad():
                    q_values = self.main_network_opponent_agent(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = torch.argmax(q_values).item()
        return action

    def update_target_network(self):
        self.target_network_our_agent.load_state_dict(self.main_network_our_agent.state_dict())
        self.target_network_opponent_agent.load_state_dict(self.main_network_opponent_agent.state_dict())

    def train_our_agent(self, batch_size):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        actual_batch_size = min(batch_size, self.batch_size)
        
        minibatch = self.replay_buffer_our_agent.sample(actual_batch_size)
        state_batch = np.array([s for s, _, _, _, _ in minibatch], dtype=np.float32)
        next_state_batch = np.array([ns for _, _, _, ns, _ in minibatch], dtype=np.float32)
        actions = torch.tensor([a for _, a, _, _, _ in minibatch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in minibatch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([d for _, _, _, _, d in minibatch], dtype=torch.bool, device=self.device)
        
        actions = torch.clamp(actions, 0, self.action_size - 1)
        
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)
    
        with torch.no_grad():
            max_next_q = self.target_network_our_agent(next_state_batch).max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q * (~dones)

        current_q_values = self.main_network_our_agent(state_batch)
        
        target_q_values_full = current_q_values.clone()

        target_q_values_full[range(actual_batch_size), actions] = target_q_values

        loss = self.our_agent_loss_fn(current_q_values, target_q_values_full)
        
        self.our_agent_optimizer.zero_grad()
        loss.backward()
        self.our_agent_optimizer.step()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return loss.item()  

    def train_opponent_agent(self, batch_size):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        actual_batch_size = min(batch_size, self.batch_size)
        
        minibatch = self.replay_buffer_opponent_agent.sample(actual_batch_size)
        state_batch = np.array([s for s, _, _, _, _ in minibatch], dtype=np.float32)
        next_state_batch = np.array([ns for _, _, _, ns, _ in minibatch], dtype=np.float32)
        actions = torch.tensor([a for _, a, _, _, _ in minibatch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in minibatch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([d for _, _, _, _, d in minibatch], dtype=torch.bool, device=self.device)

        actions = torch.clamp(actions, 0, self.action_size - 1)
        
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            max_next_q = self.target_network_opponent_agent(next_state_batch).max(dim=1)[0]
            target_q_values = rewards + self.gamma * max_next_q * (~dones)

        current_q_values = self.main_network_opponent_agent(state_batch)

        target_q_values_full = current_q_values.clone()

        target_q_values_full[range(actual_batch_size), actions] = target_q_values

        loss = self.opponent_agent_loss_fn(current_q_values, target_q_values_full)
        
        self.opponent_agent_optimizer.zero_grad()
        loss.backward()
        self.opponent_agent_optimizer.step()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return loss.item()  