import json
import torch
import numpy as np
from model.preprocessing import FrameStack
from .curriculum_manager import CurriculumManager
import gc
from .performance_metrics import PerformanceMetrics

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


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

        self.performance_metrics = PerformanceMetrics(
            config_path=param_dir + 'config.json',
            max_episode_steps=self.max_step
        )

        self.curriculum_manager = CurriculumManager()
        
        self.model = None
        self.model_name = None
        self.model_filename = None
        self.model_save_path = None
        self.train_data_filename = None

        self.curriculum_episode_rewards = []
        self.curriculum_episode_lengths = []
        self.curriculum_success_rates = []
        self.curriculum_level_transitions = []

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.batch_size = 4
            self.device = torch.device("cuda")
            print(f'Info: GPU is available... and GPU memory: {gpu_memory} GB')
            print(f'Info: Batch size set to: {self.batch_size} for memory safety')
        else:
            self.device = torch.device("cpu")
            print(f'Info: CPU is available...')
        print(f'-' * 93)

    def calculate_success_rate(self, episode_return, episode_length):
        """
        Calculate success rate based on episode performance
        You can customize this logic based on your specific success criteria
        """
        
        success_threshold = 0.0 
        length_bonus = min(episode_length / self.max_step, 1.0) 
        
        if episode_return > success_threshold:
            success_rate = min(1.0, (episode_return / 100.0) * length_bonus) 
        else:
            success_rate = 0.0
            
        return max(0.0, min(1.0, success_rate))

    def train_value_agent(self, episodes, curriculum_manager=None):
        print(f'Info: Agent Training has been started over Combat Environment...')
        print(f'Info: Curriculum Learning is {"ENABLED" if self.curriculum_manager.enabled else "DISABLED"}')
        print(f'Info: Performance Metrics System ENABLED')
        print(f'-' * 93)

        self.performance_metrics.total_episodes = episodes
        time_steps, saved_model = 0, False
        returns_per_episode = np.zeros((episodes, 2))
        training_error = np.zeros((episodes, 2))
        epsilon_history = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        
        curriculum_levels = np.zeros(episodes)
        curriculum_progressions = []
        
        for episode in range(episodes):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            current_level = self.curriculum_manager.current_level
            enemy_config = self.curriculum_manager.get_enemy_config()
            curriculum_levels[episode] = current_level

            self.performance_metrics.start_episode(episode + 1, current_level)

            if episode % 50 == 0 or (episode > 0 and curriculum_levels[episode] != curriculum_levels[episode-1]):
                print(f"Episode {episode + 1}: Curriculum Level {current_level} - Enemy Config: {enemy_config}")
            
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
                    actions[agent] = self.model.act(state[agent], agent, enemy_config if agent != self.env.agents[0] else None)
                    
                new_observations_frame, rewards, terminations, truncations, infos = self.env.step(actions)
                new_state = self.frame_stack.update_frame_stack(new_observations_frame)                
                
                for agent in self.env.agents:
                    returns[agent] += rewards[agent]
                    done[agent] = terminations[agent] or truncations[agent]

                mapped_actions = {}
                mapped_rewards = {}
                mapped_terminations = {}
                
                agent_list = list(self.env.agents)
                if len(agent_list) >= 2:
                    mapped_actions['agent0'] = actions[agent_list[0]]
                    mapped_actions['agent1'] = actions[agent_list[1]]
                    mapped_rewards['agent0'] = rewards[agent_list[0]]
                    mapped_rewards['agent1'] = rewards[agent_list[1]]
                    mapped_terminations['agent0'] = terminations[agent_list[0]]
                    mapped_terminations['agent1'] = terminations[agent_list[1]]
                
                self.performance_metrics.update_step(
                    step + 1, 
                    new_observations_frame, 
                    mapped_actions,  
                    mapped_rewards,  
                    mapped_terminations,  
                    infos
                )
                
                self.model.remember(state, actions, rewards, new_state, terminations)
                observations_frame = new_observations_frame
                step += 1

                if any(done.values()) or step == self.max_step:
                    our_agent_return = returns[self.env.agents[0]]
                    success_rate = self.calculate_success_rate(our_agent_return, step)
                    
                    level_advanced = self.curriculum_manager.update_performance(
                        our_agent_return, success_rate, step
                    )
                    
                    if level_advanced:
                        curriculum_progressions.append(episode)
                        print(f"\nCURRICULUM LEVEL ADVANCED at Episode {episode + 1}!")
                        print(f"   New Level: {self.curriculum_manager.current_level}")
                        print(f"   Success Rate: {success_rate:.3f}")

                        if self.curriculum_manager.save_checkpoints and hasattr(self.model, 'main_network_our_agent'):
                            self.curriculum_manager.save_checkpoint(self.model, self.model_save_path)
  
                    mapped_losses = {
                        'agent0': loses[self.env.agents[0]] if self.env.agents[0] in loses else 0,
                        'agent1': loses[self.env.agents[1]] if len(self.env.agents) > 1 and self.env.agents[1] in loses else 0
                    }

                    episode_metrics = self.performance_metrics.end_episode(
                        epsilon=self.model.epsilon,
                        training_losses=mapped_losses
                    )

                    if episode % 10 == 0 or level_advanced:
                        cps = episode_metrics.get('combat_performance_score', 0)
                        deception_resistance = episode_metrics.get('deception_resistance_score', 0)
                        print(f"Episode {episode + 1}/{episodes} - Level: {current_level} - "
                              f"Steps: {step} - Return: {our_agent_return:.2f} - "
                              f"Success Rate: {success_rate:.3f} - CPS: {cps:.3f} - "
                              f"Deception Resistance: {deception_resistance:.3f} - Epsilon: {self.model.epsilon:.3f}")
                    
                    break

                if len(self.model.replay_buffer_our_agent.buffer) > self.batch_size:
                    try:
                        for agent in self.env.agents:
                            if agent == self.env.agents[0]:
                                loses[agent] = self.model.train_our_agent(self.batch_size)
                            else:
                                loses[agent] = self.model.train_opponent_agent(self.batch_size)
                    except torch.cuda.OutOfMemoryError:
                        print(f"Warning: CUDA out of memory at episode {episode+1}, step {step}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        continue

            self.model.epsilon = max(self.model.epsilon * self.model.epsilon_decay, self.model.epsilon_min)

            for i, agent in enumerate(self.env.agents):
                returns_per_episode[episode, i] = returns[agent]
                training_error[episode, i] = loses[agent]
            steps_per_episode[episode] = step
            epsilon_history[episode] = self.model.epsilon

            self.curriculum_episode_rewards.append(returns[self.env.agents[0]])
            self.curriculum_episode_lengths.append(step)
            self.curriculum_success_rates.append(success_rate)

            if (episode + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"Episode {episode+1}: GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        

        print(f"Training completed. Saving final performance metrics...")
        self.performance_metrics.save_final_data()  
        print(f"Final performance metrics saved with {len(self.performance_metrics.episode_data)} episodes")
        

        self.print_curriculum_summary()
        self.print_performance_summary()
        
        print(f'-' * 93)
        if not saved_model:
            self.save_model()
        print(f'-' * 93)

        curriculum_data = {
            'levels': curriculum_levels,
            'progressions': curriculum_progressions,
            'success_rates': self.curriculum_success_rates,
            'final_stats': self.curriculum_manager.get_curriculum_stats()
        }
        
        return [returns_per_episode, epsilon_history, training_error, steps_per_episode, curriculum_data]


    def print_curriculum_summary(self):
        """Print comprehensive curriculum learning summary"""
        if not self.curriculum_manager.enabled:
            return
            
        stats = self.curriculum_manager.get_curriculum_stats()
        
        print(f'\n{"="*60}')
        print(f'CURRICULUM LEARNING SUMMARY')
        print(f'{"="*60}')
        print(f'Final Level Reached: {stats["current_level"]}/{self.curriculum_manager.max_levels}')
        print(f'Episodes at Current Level: {stats["episodes_at_level"]}')
        print(f'Current Performance: {stats["current_performance"]:.3f}')
        
        if stats['level_transitions']:
            print(f'\nLevel Transitions:')
            for i, (from_level, to_level, episodes) in enumerate(stats['level_transitions']):
                print(f'  Level {from_level} → {to_level}: {episodes} episodes')
        
        if stats['performance_per_level']:
            print(f'\nPerformance by Level:')
            for i, perf in enumerate(stats['performance_per_level']):
                print(f'  Level {i+1}: {perf:.3f} avg performance')
        
        if len(self.curriculum_success_rates) > 0:
            avg_success_rate = np.mean(self.curriculum_success_rates)
            final_success_rate = np.mean(self.curriculum_success_rates[-50:]) if len(self.curriculum_success_rates) >= 50 else avg_success_rate
            print(f'\nOverall Success Rate: {avg_success_rate:.3f}')
            print(f'Final 50 Episodes Success Rate: {final_success_rate:.3f}')
        
        print(f'{"="*60}\n')

    
    def print_performance_summary(self):
        """Print comprehensive performance metrics summary"""
        performance_summary = self.performance_metrics.get_current_performance_summary()
        
        print(f'\n{"="*60}')
        print(f'PERFORMANCE METRICS SUMMARY')
        print(f'{"="*60}')
        print(f'Total Episodes Completed: {performance_summary.get("total_episodes", 0)}')
        print(f'Recent Average CPS: {performance_summary.get("recent_avg_cps", 0):.3f}')
        print(f'Recent Success Rate: {performance_summary.get("recent_success_rate", 0):.3f}')
        print(f'Recent Avg Time to Capture: {performance_summary.get("recent_avg_capture_time", 0):.1f}')
        print(f'Recent Deception Resistance: {performance_summary.get("recent_deception_resistance", 0):.3f}')
        print(f'Excel File: {self.performance_metrics.excel_filename}')
        print(f'{"="*60}\n')


    def save_model(self):
        """Enhanced model saving with curriculum information"""
        torch.save(self.model.main_network_our_agent.state_dict(), self.model_save_path + self.model_filename)
        torch.save(self.model.main_network_opponent_agent.state_dict(), self.model_save_path + self.model_filename)
        
        if self.curriculum_manager.enabled:
            curriculum_model_data = {
                'our_agent_state_dict': self.model.main_network_our_agent.state_dict(),
                'opponent_agent_state_dict': self.model.main_network_opponent_agent.state_dict(),
                'curriculum_stats': self.curriculum_manager.get_curriculum_stats(),
                'curriculum_config': {
                    'max_levels': self.curriculum_manager.max_levels,
                    'current_level': self.curriculum_manager.current_level,
                    'progression_type': self.curriculum_manager.progression_type,
                    'threshold': self.curriculum_manager.threshold
                },
                'training_metrics': {
                    'episode_rewards': self.curriculum_episode_rewards,
                    'episode_lengths': self.curriculum_episode_lengths,
                    'success_rates': self.curriculum_success_rates
                }
            }
            
            curriculum_filename = self.model_name + '_curriculum_' + str(len(self.curriculum_episode_rewards)) + '_ep_final.pt'
            torch.save(curriculum_model_data, self.model_save_path + curriculum_filename)
            print(f'Info: Curriculum-enhanced model saved as {curriculum_filename}')
        
        print(f'Info: The model has been saved...')

    def load_curriculum_model(self, model_path):
        """Load a curriculum-enhanced model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'curriculum_stats' in checkpoint:
                print(f'Info: Loading curriculum-enhanced model...')
                self.model.main_network_our_agent.load_state_dict(checkpoint['our_agent_state_dict'])
                self.model.main_network_opponent_agent.load_state_dict(checkpoint['opponent_agent_state_dict'])
                
                curriculum_config = checkpoint.get('curriculum_config', {})
                print(f'Info: Model was trained with curriculum level {curriculum_config.get("current_level", "unknown")}')
                
                return checkpoint
            else:
                print(f'Info: Loading standard model...')
                self.model.main_network_our_agent.load_state_dict(checkpoint)
                return None
        except Exception as e:
            print(f'Error loading model: {e}')
            return None

    def check_memory_usage(self):
        """Helper method to check current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        else:
            print("GPU not available")

    def cleanup_memory(self):
        """Manual memory cleanup method"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("GPU memory cleaned up")