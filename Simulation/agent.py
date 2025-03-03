import json
import torch
import numpy as np

class DRLAgent:
    def __init__(self, env):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.env = env
        self.state_size = np.prod(env.observation_space(env.agents[0]).shape)
        self.action_size = env.action_space(env.agents[0]).n
        self.batch_size = self.params['BATCH_SIZE']

        self.model = None
        self.model_name = None
        self.model_filename = None
        self.train_data_filename = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'Info: GPU is available...')
        else:
            self.device = torch.device("cpu")
            print(f'Info: CPU is available...')
        print(f'-' * 93)

    def train_value_agent(self, episodes, render):
        print(f'Info: Agent Training has been started over Combat Environment...')
        print(f'-' * 93)

        step, time_steps, saved_model = 0, 0, False
        returns = {agent: 0 for agent in self.env.agents}
        for episode in range(episodes):
            observations = self.env.reset(seed=42)
            done = {agent: False for agent in self.env.agents}
            loses = {agent: 0.0 for agent in self.env.agents}
            actions = {agent: None for agent in self.env.agents}

            while True:
                time_steps += 1
                if time_steps % self.model.update_rate == 0:
                    self.model.update_target_network()

                for agent in self.env.agents:
                    state = np.array(observations[agent]).flatten()
                    actions = {agent: self.model.act(state, agent)}
                new_observations, rewards, terminations, truncations, infos = self.env.step(actions)

                for agent in self.env.agents:
                    returns[agent] += rewards[agent]
                    done[agent] = terminations[agent] or truncations[agent]
                    state = np.array(observations[agent]).flatten()
                    new_state = np.array(new_observations[agent]).flatten()
                    self.model.remember(state, actions[agent], rewards, new_state, terminations[agent])

                observations = new_observations
                step += 1

                if all(done.values()):
                    print(f"Episode {episode + 1}/{episodes} - Steps: {step}, Return: {returns:.2f}, Epsilon: "
                                      f"{self.model.epsilon:.3f}, Loss: {loses:0.4f}")
                    break

                if len(self.model.replay_buffer.buffer) > self.batch_size:
                    loses = self.model.train(self.batch_size)



