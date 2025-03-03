from Simulation.Utils.utils import setup_parser
from Simulation.CUAVs_sim import Simulation

def main():
    train_mode = True
    render = False
    train_episodes = 50

    sim = Simulation(args, train_mode, train_episodes, render)
    sim.run_simulation()
    sim.close_simulation()


if __name__ == '__main__':
    args = setup_parser()
    main()












#
# import torch.nn as nn
# import torch.optim as optim
# import random
# import numpy as np
# from collections import deque
#
#
# # Define the Q-network
# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_dim)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)
#
#
# # Hyperparameters
# gamma = 0.99
# learning_rate = 0.001
# epsilon = 1.0
# epsilon_decay = 0.995
# epsilon_min = 0.01
# batch_size = 32
# memory_size = 10000
# update_target = 1000
#
# # Initialize environment
# env = combat_plane_v2.env(game_version="bi-plane", guided_missile=True, render_mode="human")
# env.reset(seed=42)
# state_size = np.prod(env.observation_space(env.agents[0]).shape)
# action_size = env.action_space(env.agents[0]).n
#
# # Initialize DQN and optimizer
# policy_net = DQN(state_size, action_size)
# target_net = DQN(state_size, action_size)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
# optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
#
# # Experience replay memory
# memory = deque(maxlen=memory_size)
#
#
# # Training function
# def train():
#     if len(memory) < batch_size:
#         return
#
#     batch = random.sample(memory, batch_size)
#     states, actions, rewards, next_states, dones = zip(*batch)
#
#     states = torch.tensor(np.array(states), dtype=torch.float32)
#     actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
#     rewards = torch.tensor(rewards, dtype=torch.float32)
#     next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
#     dones = torch.tensor(dones, dtype=torch.float32)
#
#     q_values = policy_net(states).gather(1, actions).squeeze()
#     next_q_values = target_net(next_states).max(1)[0]
#     target_q_values = rewards + (gamma * next_q_values * (1 - dones))
#
#     loss = nn.functional.mse_loss(q_values, target_q_values.detach())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#
# # Main training loop
# episodes = 500
# target_update_counter = 0
#
# env.reset(seed=42)
#
# for episode in range(episodes):
#     #env.reset()
#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()
#         state = np.array(observation).flatten()
#
#         if termination or truncation:
#             action = None
#         else:
#             if random.random() < epsilon:
#                 action = env.action_space(agent).sample()
#             else:
#                 with torch.no_grad():
#                     q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
#                     action = torch.argmax(q_values).item()
#
#         env.step(action)
#         next_observation, _, next_termination, next_truncation, _ = env.last()
#         next_state = np.array(next_observation).flatten()
#         memory.append((state, action, reward, next_state, termination or truncation))
#         train()
#
#         target_update_counter += 5
#         if target_update_counter % update_target == 0:
#             target_net.load_state_dict(policy_net.state_dict())
#             target_update_counter = 0
#
#     epsilon = max(epsilon * epsilon_decay, epsilon_min)
#
# env.close()
