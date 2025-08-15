"""
ULTIMATE Performance Metrics System for Combat Environment
This version handles directory permission issues and provides multiple fallback options
"""

import numpy as np
import pandas as pd
import json
import os
import time
import stat
import tempfile
from collections import deque
from datetime import datetime
import torch


class PerformanceMetrics:
    def __init__(self, config_path='Simulation/Utils/config.json', max_episode_steps=200):
        with open(config_path, 'r') as file:
            self.params = json.load(file)
        
        self.max_episode_steps = max_episode_steps

        self.data_dir = self._setup_data_directory()

        self.cps_weights = {
            'alpha': self.params.get('CPS_ALPHA', 0.25),
            'beta': self.params.get('CPS_BETA', 0.25), 
            'gamma': self.params.get('CPS_GAMMA', 0.25),
            'delta': self.params.get('CPS_DELTA', 0.25)
        }

        self.success_threshold = self.params.get('SUCCESS_THRESHOLD', 0.0)
        self.capture_reward_threshold = self.params.get('CAPTURE_REWARD_THRESHOLD', 10.0)

        self.episode_data = []
        self.current_episode_data = {}

        self.episode_count = 0
        self.total_episodes = 0

        self.window_size = self.params.get('PERFORMANCE_WINDOW_SIZE', 50)
        self.recent_performance = deque(maxlen=self.window_size)

        self.excel_filename = None
        self.excel_path = None
        self.backup_csv_path = None
        self.temp_json_path = None
        self.initialize_file_structure()
        
        print("Performance Metrics System Initialized")
        print(f"Data Directory: {self.data_dir}")
        print(f"CPS Weights: α={self.cps_weights['alpha']}, β={self.cps_weights['beta']}, "
              f"γ={self.cps_weights['gamma']}, δ={self.cps_weights['delta']}")

    def _setup_data_directory(self):
        """Setup data directory with multiple fallback options"""
        possible_dirs = [
            self.params.get('DATA_DIR', 'Data/'),  
            './Data/',                             
            './output/',                          
            './results/',                         
            tempfile.gettempdir(),                
            os.path.expanduser('~/performance_metrics/')  
        ]
        
        for directory in possible_dirs:
            try:
                directory = os.path.expanduser(directory)
                if not os.path.exists(directory):
                    os.makedirs(directory, mode=0o755)

                test_file = os.path.join(directory, 'write_test.tmp')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                
                print(f"Using data directory: {directory}")
                return directory
                
            except (PermissionError, OSError) as e:
                print(f"Cannot use directory {directory}: {e}")
                continue

        current_dir = os.getcwd()
        print(f"WARNING: Using current directory as last resort: {current_dir}")
        return current_dir

    def initialize_file_structure(self):
        """Initialize file structure with multiple format support"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"combat_performance_metrics_{timestamp}"

        self.excel_filename = f"{base_filename}.xlsx"
        self.excel_path = os.path.join(self.data_dir, self.excel_filename)
        
        self.backup_csv_path = os.path.join(self.data_dir, f"{base_filename}_backup.csv")
        self.temp_json_path = os.path.join(self.data_dir, f"{base_filename}_temp.json")

        self._initialize_json_backup()
        
        print(f"File paths initialized:")
        print(f"  Excel: {self.excel_filename}")
        print(f"  CSV Backup: {os.path.basename(self.backup_csv_path)}")
        print(f"  JSON Temp: {os.path.basename(self.temp_json_path)}")

    def _initialize_json_backup(self):
        """Initialize JSON backup file (always works)"""
        try:
            initial_data = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'config': self.params,
                    'cps_weights': self.cps_weights
                },
                'episodes': []
            }
            with open(self.temp_json_path, 'w') as f:
                json.dump(initial_data, f, indent=2)
            print(f"JSON backup initialized: {os.path.basename(self.temp_json_path)}")
        except Exception as e:
            print(f"Warning: Could not initialize JSON backup: {e}")

    def start_episode(self, episode_num, curriculum_level=1):
        """Start tracking a new episode"""
        self.episode_count = episode_num
        self.current_episode_data = {
            'episode': episode_num,
            'curriculum_level': curriculum_level,
            'start_time': datetime.now(),
            'steps_taken': 0,
            'actions_taken': {'agent0': [], 'agent1': []},
            'rewards_received': {'agent0': [], 'agent1': []},
            'deception_events': [],
            'capture_events': [],
            'episode_return': {'agent0': 0, 'agent1': 0},
            'terminated': False,
            'success_achieved': False,
            'capture_time': None
        }

        print(f"DEBUG: Started episode {episode_num}, curriculum level {curriculum_level}")

    def update_step(self, step_num, observations, actions, rewards, terminations, infos=None):
        """Update metrics for current step"""
        if not self.current_episode_data:
            print(f"WARNING: update_step called but no episode started!")
            return
        
        self.current_episode_data['steps_taken'] = step_num

        if isinstance(actions, dict):
            for agent_key, action in actions.items():
                if agent_key in self.current_episode_data['actions_taken']:
                    self.current_episode_data['actions_taken'][agent_key].append(action)
                else:
                    if 'agent0' not in self.current_episode_data['actions_taken']:
                        mapped_agent = 'agent0'
                    else:
                        mapped_agent = 'agent1'
                    self.current_episode_data['actions_taken'][mapped_agent].append(action)
        
        if isinstance(rewards, dict):
            for agent_key, reward in rewards.items():
                if agent_key in self.current_episode_data['rewards_received']:
                    self.current_episode_data['rewards_received'][agent_key].append(reward)
                    self.current_episode_data['episode_return'][agent_key] += reward
                else:
                    if 'agent0' not in self.current_episode_data['rewards_received'] or len(self.current_episode_data['rewards_received']['agent0']) == 0:
                        mapped_agent = 'agent0'
                    else:
                        mapped_agent = 'agent1'
                    self.current_episode_data['rewards_received'][mapped_agent].append(reward)
                    self.current_episode_data['episode_return'][mapped_agent] += reward

        if isinstance(terminations, dict):
            if any(terminations.values()):
                self.current_episode_data['terminated'] = True
        elif terminations:  
            self.current_episode_data['terminated'] = True

        self._detect_deception_events(step_num, actions, rewards)

        self._detect_capture_events(step_num, rewards, terminations)

        if step_num % 50 == 0:
            agent0_return = self.current_episode_data['episode_return'].get('agent0', 0)
            print(f"DEBUG: Episode {self.episode_count}, Step {step_num}, Agent0 Return: {agent0_return}")

    def _detect_deception_events(self, step_num, actions, rewards):
        """Detect potential deception events (simplified heuristic)"""
        try:
            if len(self.current_episode_data['actions_taken']['agent0']) > 5:
                recent_actions = self.current_episode_data['actions_taken']['agent0'][-5:]
                action_changes = sum(1 for i in range(1, len(recent_actions)) 
                                   if recent_actions[i] != recent_actions[i-1])

                agent0_reward = 0
                agent1_reward = 0
                
                if isinstance(rewards, dict):
                    for key, reward in rewards.items():
                        if 'agent0' in str(key).lower() or key == list(rewards.keys())[0]:
                            agent0_reward = reward
                        else:
                            agent1_reward = reward
                
                if (action_changes >= 3 and agent0_reward < 0 and agent1_reward > 0):
                    self.current_episode_data['deception_events'].append({
                        'step': step_num,
                        'type': 'direction_confusion',
                        'severity': action_changes / 5.0
                    })
        except Exception as e:
            pass

    def _detect_capture_events(self, step_num, rewards, terminations):
        """Detect capture/intercept events"""
        try:
            agent0_reward = 0
            if isinstance(rewards, dict):
                for key, reward in rewards.items():
                    if 'agent0' in str(key).lower() or key == list(rewards.keys())[0]:
                        agent0_reward = reward
                        break
            
            terminated = False
            if isinstance(terminations, dict):
                terminated = any(terminations.values())
            elif terminations:
                terminated = True
            
            if (agent0_reward > self.capture_reward_threshold and 
                terminated and not self.current_episode_data['success_achieved']):
                
                self.current_episode_data['capture_events'].append({
                    'step': step_num,
                    'type': 'successful_capture',
                    'reward': agent0_reward
                })
                self.current_episode_data['success_achieved'] = True
                self.current_episode_data['capture_time'] = step_num
        except Exception as e:
            pass

    def end_episode(self, epsilon=None, training_losses=None):
        """End current episode and calculate all metrics"""
        if not self.current_episode_data:
            print(f"WARNING: end_episode called but no episode data!")
            return {}

        if 'episode_return' not in self.current_episode_data:
            self.current_episode_data['episode_return'] = {'agent0': 0, 'agent1': 0}

        metrics = self._calculate_episode_metrics()

        if epsilon is not None:
            metrics['epsilon'] = epsilon
        
        if training_losses:
            metrics['training_loss_agent0'] = training_losses.get('agent0', 0)
            metrics['training_loss_agent1'] = training_losses.get('agent1', 0)
        else:
            metrics['training_loss_agent0'] = 0
            metrics['training_loss_agent1'] = 0

        self.episode_data.append(metrics)

        self.recent_performance.append(metrics['combat_performance_score'])

        save_frequency = self.params.get('EXCEL_SAVE_FREQUENCY', 10)
        if self.episode_count % save_frequency == 0:
            self.save_data_multi_method()
            print(f"Performance metrics saved at episode {self.episode_count}")

        print(f"DEBUG: Episode {self.episode_count} completed - CPS: {metrics['combat_performance_score']:.3f}, Success: {metrics['success_rate']:.3f}")

        self.current_episode_data = {}
        
        return metrics

    def _calculate_episode_metrics(self):
        """Calculate all performance metrics for current episode"""
        episode_return_agent0 = self.current_episode_data['episode_return'].get('agent0', 0)
        episode_return_agent1 = self.current_episode_data['episode_return'].get('agent1', 0)
        episode_length = self.current_episode_data.get('steps_taken', 0)

        success_rate = self._calculate_success_rate()

        time_to_capture = self._calculate_time_to_capture()

        deception_resistance = self._calculate_deception_resistance()

        normalized_return = self._normalize_return(episode_return_agent0)

        generalization_score = self._calculate_generalization_score()

        cps = self._calculate_cps(normalized_return, success_rate, 
                                deception_resistance, generalization_score)

        agent0_wins = 1 if episode_return_agent0 > episode_return_agent1 else 0
        agent1_wins = 1 if episode_return_agent1 > episode_return_agent0 else 0
        draws = 1 if episode_return_agent0 == episode_return_agent1 else 0
        
        return {
            'episode': self.current_episode_data['episode'],
            'curriculum_level': self.current_episode_data['curriculum_level'],
            'episode_return_agent0': episode_return_agent0,
            'episode_return_agent1': episode_return_agent1,
            'episode_length': episode_length,
            'success_rate': success_rate,
            'time_to_capture': time_to_capture,
            'deception_resistance_score': deception_resistance,
            'combat_performance_score': cps,
            'normalized_return': normalized_return,
            'generalization_score': generalization_score,
            'agent0_wins': agent0_wins,
            'agent1_wins': agent1_wins,
            'draws': draws,
            'avg_episode_return': (episode_return_agent0 + episode_return_agent1) / 2
        }

    def _calculate_success_rate(self):
        """Calculate success rate for current episode"""
        episode_return_agent0 = self.current_episode_data['episode_return'].get('agent0', 0)
        episode_length = self.current_episode_data.get('steps_taken', 0)
        
        positive_return = episode_return_agent0 > self.success_threshold
        no_timeout = episode_length < self.max_episode_steps
        capture_success = len(self.current_episode_data['capture_events']) > 0
        
        success_indicators = [positive_return, no_timeout, capture_success]
        success_rate = sum(success_indicators) / len(success_indicators)
        
        return success_rate

    def _calculate_time_to_capture(self):
        """Calculate time to capture/intercept"""
        if self.current_episode_data.get('capture_time'):
            return self.current_episode_data['capture_time']
        else:
            return self.current_episode_data.get('steps_taken', self.max_episode_steps)

    def _calculate_deception_resistance(self):
        """Calculate deception resistance score"""
        if not self.current_episode_data['deception_events']:
            return 1.0  

        total_deception_severity = sum(event['severity'] 
                                     for event in self.current_episode_data['deception_events'])

        episode_length = max(self.current_episode_data.get('steps_taken', 1), 1)
        deception_density = total_deception_severity / episode_length

        resistance_score = max(0.0, 1.0 - deception_density)
        
        return resistance_score

    def _normalize_return(self, episode_return):
        """Normalize episode return for CPS calculation"""
        return 1 / (1 + np.exp(-episode_return / 50.0))

    def _calculate_generalization_score(self):
        """Calculate generalization score (simplified)"""
        if len(self.recent_performance) < 5:
            return 0.5  

        recent_scores = list(self.recent_performance)[-10:] 
        if len(recent_scores) < 2:
            return 0.5

        variance = np.var(recent_scores)
        generalization = max(0.0, 1.0 - variance)
        
        return generalization

    def _calculate_cps(self, normalized_return, success_rate, deception_resistance, generalization):
        """Calculate Combat Performance Score (CPS)"""
        cps = (self.cps_weights['alpha'] * normalized_return +
               self.cps_weights['beta'] * success_rate +
               self.cps_weights['gamma'] * deception_resistance +
               self.cps_weights['delta'] * generalization)
        
        return cps

    def save_data_multi_method(self):
        """Save data using multiple methods as fallbacks"""
        if not self.episode_data:
            print("No episode data to save")
            return
        
        saved_count = 0

        if self._save_to_json():
            saved_count += 1

        if self._save_to_csv():
            saved_count += 1

        if self._save_to_excel():
            saved_count += 1
        
        if saved_count == 0:
            print("WARNING: Could not save data with any method!")
            self._save_to_memory_dump()
        else:
            print(f"Data saved using {saved_count}/3 methods successfully")

    def _save_to_json(self):
        """Save to JSON format (most reliable)"""
        try:
            data_to_save = {
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'total_episodes': len(self.episode_data),
                    'config': self.params,
                    'cps_weights': self.cps_weights
                },
                'episodes': self.episode_data
            }
            
            with open(self.temp_json_path, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            
            print(f"✓ JSON saved: {os.path.basename(self.temp_json_path)}")
            return True
        except Exception as e:
            print(f"✗ JSON save failed: {e}")
            return False

    def _save_to_csv(self):
        """Save to CSV format"""
        try:
            df = pd.DataFrame(self.episode_data)

            required_columns = [
                'episode', 'curriculum_level', 'episode_return_agent0', 'episode_return_agent1',
                'episode_length', 'success_rate', 'time_to_capture', 'deception_resistance_score',
                'combat_performance_score', 'normalized_return', 'generalization_score',
                'agent0_wins', 'agent1_wins', 'draws'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            df.to_csv(self.backup_csv_path, index=False)
            print(f"✓ CSV saved: {os.path.basename(self.backup_csv_path)}")
            return True
        except Exception as e:
            print(f"✗ CSV save failed: {e}")
            return False

    def _save_to_excel(self):
        """Save to Excel format (with error handling)"""
        try:
            df = pd.DataFrame(self.episode_data)

            required_columns = [
                'episode', 'curriculum_level', 'episode_return_agent0', 'episode_return_agent1',
                'episode_length', 'success_rate', 'time_to_capture', 'deception_resistance_score',
                'combat_performance_score', 'normalized_return', 'generalization_score',
                'agent0_wins', 'agent1_wins', 'draws'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            with pd.ExcelWriter(self.excel_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name='Performance_Data', index=False)
                self._create_summary_sheet(writer, df)
                if 'curriculum_level' in df.columns:
                    self._create_curriculum_sheet(writer, df)
            
            print(f"✓ Excel saved: {self.excel_filename}")
            return True
        except Exception as e:
            print(f"✗ Excel save failed: {e}")
            return False

    def _save_to_memory_dump(self):
        """Last resort: save raw data to a text file"""
        try:
            dump_path = os.path.join(self.data_dir, f"performance_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(dump_path, 'w') as f:
                f.write(f"Performance Metrics Memory Dump\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Episodes: {len(self.episode_data)}\n\n")
                
                for i, episode in enumerate(self.episode_data):
                    f.write(f"Episode {i+1}: {episode}\n")
            
            print(f"Emergency dump saved: {os.path.basename(dump_path)}")
        except Exception as e:
            print(f"Even emergency dump failed: {e}")

    def _create_summary_sheet(self, writer, df):
        """Create summary statistics sheet"""
        try:
            summary_stats = {
                'Metric': [
                    'Total Episodes', 'Average Episode Return (Agent 0)', 'Average Episode Return (Agent 1)',
                    'Average Episode Length', 'Overall Success Rate', 'Average Time to Capture',
                    'Average Deception Resistance', 'Average Combat Performance Score',
                    'Agent 0 Win Rate', 'Agent 1 Win Rate', 'Draw Rate'
                ],
                'Value': [
                    len(df),
                    df['episode_return_agent0'].mean(),
                    df['episode_return_agent1'].mean(),
                    df['episode_length'].mean(),
                    df['success_rate'].mean(),
                    df['time_to_capture'].mean(),
                    df['deception_resistance_score'].mean(),
                    df['combat_performance_score'].mean(),
                    df['agent0_wins'].sum() / len(df) if len(df) > 0 else 0,
                    df['agent1_wins'].sum() / len(df) if len(df) > 0 else 0,
                    df['draws'].sum() / len(df) if len(df) > 0 else 0
                ]
            }
            
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        except Exception as e:
            print(f"Error creating summary sheet: {e}")

    def _create_curriculum_sheet(self, writer, df):
        """Create curriculum-specific analysis sheet"""
        try:
            curriculum_analysis = df.groupby('curriculum_level').agg({
                'episode_return_agent0': ['mean', 'std', 'count'],
                'success_rate': ['mean', 'std'],
                'time_to_capture': ['mean', 'std'],
                'deception_resistance_score': ['mean', 'std'],
                'combat_performance_score': ['mean', 'std']
            }).round(4)

            curriculum_analysis.columns = ['_'.join(col).strip() for col in curriculum_analysis.columns]
            curriculum_analysis = curriculum_analysis.reset_index()
            
            curriculum_analysis.to_excel(writer, sheet_name='Curriculum_Analysis', index=False)
        except Exception as e:
            print(f"Error creating curriculum sheet: {e}")

    def get_current_performance_summary(self):
        """Get current performance summary"""
        if not self.episode_data:
            return {
                'recent_avg_cps': 0,
                'recent_success_rate': 0,
                'recent_avg_capture_time': 0,
                'recent_deception_resistance': 0,
                'total_episodes': 0
            }
        
        recent_data = self.episode_data[-10:] if len(self.episode_data) >= 10 else self.episode_data
        df = pd.DataFrame(recent_data)
        
        return {
            'recent_avg_cps': df['combat_performance_score'].mean(),
            'recent_success_rate': df['success_rate'].mean(),
            'recent_avg_capture_time': df['time_to_capture'].mean(),
            'recent_deception_resistance': df['deception_resistance_score'].mean(),
            'total_episodes': len(self.episode_data)
        }

    def save_final_data(self):
        """Force save all data at the end of training"""
        print("Saving final performance data...")
        self.save_data_multi_method()

        self._create_final_report()
        
        print(f"Final data saved. Total episodes: {len(self.episode_data)}")

    def _create_final_report(self):
        """Create a final text summary report"""
        try:
            report_path = os.path.join(self.data_dir, f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            performance_summary = self.get_current_performance_summary()
            
            with open(report_path, 'w') as f:
                f.write("COMBAT TRAINING FINAL REPORT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Training completed at: {datetime.now().isoformat()}\n")
                f.write(f"Total episodes: {performance_summary['total_episodes']}\n")
                f.write(f"Final Combat Performance Score: {performance_summary['recent_avg_cps']:.3f}\n")
                f.write(f"Final Success Rate: {performance_summary['recent_success_rate']:.3f}\n")
                f.write(f"Final Avg Time to Capture: {performance_summary['recent_avg_capture_time']:.1f}\n")
                f.write(f"Final Deception Resistance: {performance_summary['recent_deception_resistance']:.3f}\n\n")
                
                f.write("Files generated:\n")
                f.write(f"- JSON: {os.path.basename(self.temp_json_path)}\n")
                f.write(f"- CSV: {os.path.basename(self.backup_csv_path)}\n")
                f.write(f"- Excel: {self.excel_filename}\n")
                f.write(f"- Report: {os.path.basename(report_path)}\n")
            
            print(f"Final report created: {os.path.basename(report_path)}")
        except Exception as e:
            print(f"Could not create final report: {e}")

    def update_cps_weights(self, alpha=None, beta=None, gamma=None, delta=None):
        """Update CPS weights"""
        if alpha is not None:
            self.cps_weights['alpha'] = alpha
        if beta is not None:
            self.cps_weights['beta'] = beta
        if gamma is not None:
            self.cps_weights['gamma'] = gamma
        if delta is not None:
            self.cps_weights['delta'] = delta
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.cps_weights.values())
        for key in self.cps_weights:
            self.cps_weights[key] /= total_weight
        
        print(f"CPS weights updated: α={self.cps_weights['alpha']:.3f}, "
              f"β={self.cps_weights['beta']:.3f}, γ={self.cps_weights['gamma']:.3f}, "
              f"δ={self.cps_weights['delta']:.3f}")