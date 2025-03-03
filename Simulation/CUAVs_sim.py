import json
import os
import time
from pettingzoo.atari import combat_plane_v2
from .agent import DRLAgent
from model.dqntorch import DQNModel

class Simulation:
    def __init__(self, args, train_mode, train_episodes=100, render=False):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        print(f'\n' + '%' * 20 + ' Advanced Adversarial Learning in Multi Agent Combat ' + '%' * 20)
        self.data_dir = self.params['DATA_DIR']

        if not os.path.exists(self.data_dir):
            print(f'Exception: Data directory is not exists. Unable to load saved model!!')
            exit(0)

        self.env = combat_plane_v2.env(game_version="bi-plane", guided_missile=True, render_mode="human")
        self.env.reset(seed=42)
        self.agent = DRLAgent(self.env)

        self.render = render
        self.train_mode = train_mode
        self.is_trained = None
        self.train_episodes = train_episodes
        self.test_episodes = self.params['TEST_EPISODES']
        self.is_test_completed = False

        if args.dqn:
            self.agent.model_name = 'DQN'
        elif args.doubledqn:
            self.agent.model_name = 'Double DQN'
        elif args.dueldqn:
            self.agent.model_name = 'Dueling DQN'
        else:
            print(f'Exception: Model type and passed argument are not matched.')
            exit(0)

        self.train_start_time = None
        self.train_end_time = None
        self.sim_start_time = None
        self.sim_end_time = None
        self.is_env_initialized = False
        self.running = True

    def run_simulation(self):
        self.game_initialize()
        self.sim_start_time = time.time()

        while self.running:
            if self.train_mode and not self.is_trained:
                print(f'=' * 38 + ' Training Phase ' + '=' * 39)
                self.train_start_time = time.time()
                result = self.agent.train_value_agent(self.train_episodes, self.render)
                self.train_end_time = time.time()
                elapsed_time = self.train_end_time - self.train_start_time
                print(f'Info: Training has been completed...')
                print(f'Info: Training Completion Time: {elapsed_time:.2f} seconds')
                print(f'-' * 93)
                self.is_trained = True

                break

        self.sim_end_time = time.time()
        elapsed_time = self.sim_end_time - self.sim_start_time
        print(f'Info: Simulation Completion Time: {elapsed_time:.2f} seconds')

    def game_initialize(self):
        if self.train_mode:
            self.is_trained = False
        else:
            self.is_trained = True

        self.is_env_initialized = True
        if self.agent.model_name == 'DQN':
            print(f'Info: Selected Model is {self.agent.model_name}')
            self.agent.model = DQNModel(self.agent.state_size, self.agent.action_size, self.env, self.agent.device)
            print(f'Info: DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Double DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            print(f'Info: Double DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Dueling DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            print(f'Info: Dueling DQN Model is assigned for the Training and Testing of Agent...')

        self.agent.model_filename = self.agent.model_name + '_' + str(self.train_episodes) + '_ep_final.pt'
        self.agent.train_data_filename = self.agent.model_name + '_' + str(self.train_episodes) + '_training_data.xlsx'

    def close_simulation(self):
        self.env.close() if self.render else None
