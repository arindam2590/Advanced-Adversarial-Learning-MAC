import json
import torch
import numpy as np
from model.preprocessing import FrameStack


class DRLAgent:
    def __init__(self, env):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.env = env
        self.action_size = env.action_space(env.possible_agents[0]).n
        self.state_size = self.params['STACK_SIZE']
        self.frame_stack = FrameStack(self.env.possible_agents, stack_size=self.state_size)
        self.batch_size = self.params['BATCH_SIZE']
        self.max_step = self.params['MAX_STEP']

        self.model = None
        self.model_name = None
        self.model_filename = None
        self.model_save_path = None
        self.train_data_filename = None

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            optimal_batch_size = min(64, int(gpu_memory // (84 * 84 * 1 * 4)))
            self.batch_size = optimal_batch_size
            self.device = torch.device("cuda")
            print(f'Info: GPU is available... and GPU memory: {gpu_memory} GB')
        else:
            self.device = torch.device("cpu")
            print(f'Info: CPU is available...')
        print(f'-' * 93)

    def train_value_agent(self, episodes):
        print(f'Info: Agent Training has been started over Combat Environment...')
        print(f'-' * 93)

        time_steps, saved_model = 0, False
        returns_per_episode = np.zeros((episodes, 2))
        training_error = np.zeros((episodes, 2))
        epsilon_history = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        
        for episode in range(episodes):
            observations_frame, _ = self.env.reset(seed=42)
            self.frame_stack.init_stack(observations_frame, self.env.agents)
            
            step, flag = 0, True
            done = {agent: True for agent in self.env.agents}
            state = {agent: None for agent in self.env.agents}
            loses = {agent: 0.0 for agent in self.env.agents}
            actions = returns = {agent: 0 for agent in self.env.agents}

            while True:
                time_steps += 1
                if time_steps % self.model.update_rate == 0:
                    self.model.update_target_network()

                for agent in self.env.agents:
                    state[agent] = self.frame_stack.get_state(agent)
                    actions[agent] = self.model.act(state[agent], agent)
                new_observations_frame, rewards, terminations, truncations, infos = self.env.step(actions)
                new_state = self.frame_stack.update_frame_stack(new_observations_frame)                
                for agent in self.env.agents:
                    returns[agent] += rewards[agent]
                    done[agent] = terminations[agent] or truncations[agent]
                
                self.model.remember(state, actions, rewards, new_state, terminations)
                observations_frame = new_observations_frame
                step += 1

                if any(done.values()) or step == self.max_step:
                    print(f"Episode {episode + 1}/{episodes} - Steps: {step}")
                    #print(f"Episode {episode + 1}/{episodes} - Steps: {step}, Return: {returns:.2f}, Epsilon: "
                     #     f"{self.model.epsilon:.3f}, Our Agent Loss: {loses[0]:0.4f}, "
                      #    f"Opponent Agent Loss: {loses[1]:0.4f}")
                    break

                if len(self.model.replay_buffer_our_agent.buffer) > self.batch_size:
                    for agent in self.env.agents:
                        if agent == self.env.agents[0]:
                            loses[agent] = self.model.train_our_agent(self.batch_size)
                        else:
                            loses[agent] = self.model.train_opponent_agent(self.batch_size)
            self.model.epsilon = max(self.model.epsilon * self.model.epsilon_decay, self.model.epsilon_min)
            for i, agent in enumerate(self.env.agents):
                returns_per_episode[episode, i] = returns[agent]
                training_error[episode, i] = loses[agent]
            steps_per_episode[episode] = step
            epsilon_history[episode] = self.model.epsilon           
        
        print(f'-' * 93)
        if not saved_model:
            self.save_model()
        print(f'-' * 93)  
        return [returns_per_episode, epsilon_history, training_error, steps_per_episode, None]
        
    def save_model(self):
        torch.save(self.model.main_network_our_agent.state_dict(), self.model_save_path + self.model_filename)
        torch.save(self.model.main_network_opponent_agent.state_dict(), self.model_save_path + self.model_filename)
        print(f'Info: The model has been saved...')




