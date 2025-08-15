import json 
import os
import time
from pettingzoo.atari import combat_plane_v2
from .agent import DRLAgent
from model.dqntorch import DQNModel
from .Utils.utils import DataVisualization
from .curriculum_manager import CurriculumManager  
from .enemy_behavior import EnemyBehaviorController


class Simulation:
    def __init__(self, args, train_mode, train_episodes=1500, render=False):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        print(f'\n' + '%' * 20 + ' Advanced Adversarial Learning in Multi Agent Combat ' + '%' * 20)
        self.data_dir = self.params['DATA_DIR']

        if not os.path.exists(self.data_dir):
            print(f'Exception: Data directory is not exists. Unable to load saved model!!')
            exit(0)

        if render:
            self.env = combat_plane_v2.parallel_env(game_version="bi-plane", guided_missile=True, render_mode="human")
        else:
            self.env = combat_plane_v2.parallel_env(game_version="bi-plane", guided_missile=True, render_mode=None)
        self.render = render
        self.agent = DRLAgent(self.env)

        self.train_mode = train_mode
        self.is_trained = None
        self.train_episodes = train_episodes
        self.test_episodes = self.params['TEST_EPISODES']
        self.is_test_completed = False

        self.curriculum_manager = CurriculumManager()
        self.enemy_controller = None

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

        self.curriculum_data = {
            'level_history': [],
            'performance_history': [],
            'level_transitions': []
        }

    def run_simulation(self):
        self.game_initialize()
        self.sim_start_time = time.time()

        while self.running:
            if self.train_mode and not self.is_trained:
                print(f'=' * 38 + ' Training Phase ' + '=' * 39)
                
                if self.curriculum_manager.enabled:
                    print(f'CURRICULUM LEARNING ENABLED')
                    print(f'Current Level: {self.curriculum_manager.current_level}/{self.curriculum_manager.max_levels}')
                    print(f'Enemy Configuration: {self.curriculum_manager.get_enemy_config()}')
                    print(f'-' * 93)
                
                self.train_start_time = time.time()
                result = self.agent.train_value_agent(self.train_episodes, self.curriculum_manager)  # MODIFIED
                
                train_data_visual = DataVisualization(
                    self.train_episodes, 
                    result, 
                    self.agent.model_name, 
                    self.agent.train_data_filename
                )
                
                train_data_visual.save_data()
                train_data_visual.plot_returns()
                train_data_visual.plot_episode_length()
                train_data_visual.plot_training_error()
                train_data_visual.plot_epsilon_decay()
                train_data_visual.plot_comprehensive_summary()

                print("Generating performance metrics analysis...")
                train_data_visual.plot_performance_metrics()
                train_data_visual.plot_curriculum_performance_analysis()

                performance_summary = self.agent.performance_metrics.get_current_performance_summary()
                curriculum_stats = self.curriculum_manager.get_curriculum_stats()

                print(f'\n{"="*100}')
                print(f'COMPREHENSIVE TRAINING SUMMARY')
                print(f'{"="*100}')
                print(f'Model: {self.agent.model_name}')
                print(f'Training Episodes: {self.train_episodes}')
                print(f'Curriculum Learning: {"ENABLED" if self.curriculum_manager.enabled else "DISABLED"}')
                
                if self.curriculum_manager.enabled:
                    print(f'Final Curriculum Level: {curriculum_stats["current_level"]}/{self.curriculum_manager.max_levels}')
                    print(f'Level Transitions: {len(curriculum_stats["level_transitions"])}')
                
                print(f'\nPERFORMance METRICS:')
                print(f'├─ Final Combat Performance Score: {performance_summary.get("recent_avg_cps", 0):.3f}')
                print(f'├─ Final Success Rate: {performance_summary.get("recent_success_rate", 0):.3f}')
                print(f'├─ Final Avg Time to Capture: {performance_summary.get("recent_avg_capture_time", 0):.1f} steps')
                print(f'├─ Final Deception Resistance: {performance_summary.get("recent_deception_resistance", 0):.3f}')
                print(f'└─ Performance Data File: {self.agent.performance_metrics.excel_filename}')
                
                print(f'\nFILES GENERATED:')
                print(f'├─ Training Data: {self.agent.train_data_filename}')
                print(f'├─ Performance Metrics: {self.agent.performance_metrics.excel_filename}')
                print(f'├─ Model Checkpoint: {self.agent.model_filename}')
                print(f'└─ Visualization Plots: Multiple PNG files in {self.data_dir}')
                print(f'{"="*100}')
                
                
                self.train_end_time = time.time()
                elapsed_time = self.train_end_time - self.train_start_time
                print(f'Info: Training has been completed...')
                print(f'Info: Training Completion Time: {elapsed_time:.2f} seconds')
                
                if self.curriculum_manager.enabled:
                    self._display_curriculum_summary()
                
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
            self.agent.model = DQNModel(
                self.agent.state_size, 
                self.agent.action_size, 
                self.env, 
                self.agent.device, 
                self.agent.frame_stack
            )
            self.agent.model.curriculum_manager = self.curriculum_manager
            print(f'Info: DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Double DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            print(f'Info: Double DQN Model is assigned for the Training and Testing of Agent...')
        elif self.agent.model_name == 'Dueling DQN':
            print(f'Info: Model Selected: {self.agent.model_name}')
            print(f'Info: Dueling DQN Model is assigned for the Training and Testing of Agent...')

        self.agent.model_save_path = 'Data/'
        self.agent.model_filename = self.agent.model_name + '_' + str(self.train_episodes) + '_ep_final.pt'
        self.agent.train_data_filename = self.agent.model_name + '_' + str(self.train_episodes) + '_training_data.xlsx'

        if self.curriculum_manager.enabled:
            enemy_config = self.curriculum_manager.get_enemy_config()
            self.enemy_controller = EnemyBehaviorController(enemy_config)

    def _display_curriculum_summary(self):
        """Display curriculum learning summary"""
        stats = self.curriculum_manager.get_curriculum_stats()
        
        print(f'\n CURRICULUM LEARNING SUMMARY:')
        print(f'├─ Final Level Reached: {stats["current_level"]}/{self.curriculum_manager.max_levels}')
        print(f'├─ Episodes at Current Level: {stats["episodes_at_level"]}')
        print(f'├─ Current Performance: {stats["current_performance"]:.3f}')
        print(f'├─ Level Transitions: {len(stats["level_transitions"])}')
        
        if stats['level_transitions']:
            print(f'├─ Level Progression:')
            for i, (from_level, to_level, episodes) in enumerate(stats['level_transitions']):
                print(f'│  └─ Level {from_level} → {to_level} (after {episodes} episodes)')
        
        if stats['episodes_per_level']:
            avg_episodes = sum(stats['episodes_per_level']) / len(stats['episodes_per_level'])
            print(f'└─ Average Episodes per Level: {avg_episodes:.1f}')

    def update_enemy_behavior(self):
        """Update enemy behavior based on curriculum level"""
        if self.curriculum_manager.enabled:
            enemy_config = self.curriculum_manager.get_enemy_config()
            self.enemy_controller = EnemyBehaviorController(enemy_config)
            print(f'Enemy behavior updated to level {self.curriculum_manager.current_level}')

    def close_simulation(self):
        self.env.close() if self.render else None