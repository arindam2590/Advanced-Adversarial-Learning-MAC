import argparse


def setup_parser():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-d', '--dqn',
                        action='store_true',
                        help='Simulate the environment with DQN Model: use argument -d or --dqn')
    parser.add_argument('-q', '--doubledqn',
                        action='store_true',
                        help='Simulate the environment with Double DQN Model: use argument -q or --ddqn')
    parser.add_argument('-u', '--dueldqn',
                        action='store_true',
                        help='Simulate the environment with Dueling DQN Model: use argument -u or --dueldqn')
    args = parser.parse_args()
    return args
    
class DataVisualization:
    def __init__(self, episodes, result, model, train_data_filename):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)
        self.model = model
        self.fig_dir = self.params['DATA_DIR']
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.train_data_filename = train_data_filename
        self.n_episodes = episodes
        self.returns = result[0]
        self.epsilon_decay_history = result[1]
        self.training_error = result[2]
        self.steps = result[3]
        
            
    def save_data(self):
        filepath = self.fig_dir + self.train_data_filename
        df = pd.DataFrame({'Rewards': self.returns,
                           'Steps': self.steps,
                           'Epsilon Decay': self.epsilon_decay_history,
                           'Training Error': self.training_error})
        if not os.path.isfile(filepath):
            with pd.ExcelWriter(filepath, mode='w') as writer:
                df.to_excel(writer, sheet_name=self.model)
                
        else:
            with pd.ExcelWriter(filepath, mode='a') as writer:
                df.to_excel(writer, sheet_name=self.model)
