import json
import torch
import numpy as np
from .Utils.preprocessing import preprocess_state

class DRLAgent:
    def __init__(self, env):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.env = env
        self.state_size = np.prod(env.observation_space(env.agents[0]).shape)
        #self.input_channels = env.observation_space(env.agents[0]).shape[2]
        self.input_channels = 1
        self.action_size = env.action_space(env.agents[0]).n
        self.batch_size = self.params['BATCH_SIZE']
        self.max_step = self.params['MAX_STEP']

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
        
        time_steps, saved_model = 0, False
        returns_per_episode = np.zeros(episodes)
        epsilon_history = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        training_error = np.zeros(episodes)
        for episode in range(episodes):
            observations_frame, _ = self.env.reset()
            step = 0
            done = {agent: True for agent in self.env.agents}
            loses = {agent: 0.0 for agent in self.env.agents}
            actions = returns = {agent: 0 for agent in self.env.agents}

            while True:
                time_steps += 1
                if time_steps % self.model.update_rate == 0:
                    self.model.update_target_network()

                for agent in self.env.agents:
                    state = preprocess_state(observations_frame[agent])
                    actions[agent] = self.model.act(state, agent)
                new_observations_frame, rewards, terminations, truncations, infos = self.env.step(actions)

                for agent in self.env.agents:
                    returns[agent] += rewards[agent]
                    done[agent] = terminations[agent] or truncations[agent]
                
                self.model.remember(observations_frame, actions, rewards, new_observations_frame, terminations)
                observations_frame = new_observations_frame
                step += 1

                if any(done.values()):
                    print(f"Episode {episode + 1}/{episodes} - Steps: {step}, Return: {returns:.2f}, Epsilon: "
                          f"{self.model.epsilon:.3f}, Our Agent Loss: {loses[0]:0.4f}, "
                          f"Opponent Agent Loss: {loses[1]:0.4f}")
                    break

                if len(self.model.replay_buffer_our_agent.buffer) > self.batch_size:
                    for agent in self.env.agents:
                        if agent == self.env.agents[0]:
                            loses[agent] = self.model.train_our_agent(self.batch_size)
                        else:
                            loses[agent] = self.model.train_opponent_agent(self.batch_size)
            self.model.epsilon = max(self.model.epsilon * self.model.epsilon_decay, self.model.epsilon_min)
            returns_per_episode[episode] = returns
            epsilon_history[episode] = self.model.epsilon
            steps_per_episode[episode] = step
            training_error[episode] = loss
        
        print(f'-' * 93)
        if not saved_model:
            self.save_model()
        print(f'-' * 93)    
        return [returns_per_episode, epsilon_history, training_error, steps_per_episode, None]
        
    def save_model(self):
        torch.save(self.model.main_network_our_agent.state_dict(), self.model_save_path + self.model_filename)
        torch.save(self.model.main_network_opponent_agent.state_dict(), self.model_save_path + self.model_filename)
        print(f'Info: The model has been saved...')




