"""
Curriculum Manager - Controls progression through difficulty levels
"""
import json
import numpy as np
from collections import deque
import torch
import os

class CurriculumManager:
    def __init__(self, config_path='Simulation/Utils/config.json'):
        with open(config_path, 'r') as file:
            self.params = json.load(file)
        
        self.enabled = self.params.get('CURRICULUM_ENABLED', False)
        self.max_levels = self.params.get('CURRICULUM_LEVELS', 5)
        self.progression_type = self.params.get('CURRICULUM_PROGRESSION', 'threshold')
        self.threshold = self.params.get('CURRICULUM_THRESHOLD', 0.05)
        self.eval_episodes = self.params.get('CURRICULUM_EVALUATION_EPISODES', 50)
        self.min_episodes = self.params.get('CURRICULUM_MIN_EPISODES', 300)
        self.fixed_interval = self.params.get('CURRICULUM_FIXED_INTERVAL', 300)
        self.save_checkpoints = self.params.get('CURRICULUM_SAVE_CHECKPOINTS', True)
        
        self.current_level = 1
        self.episodes_at_level = 0
        self.performance_history = deque(maxlen=self.eval_episodes)
        self.curriculum_stats = {
            'level_transitions': [],
            'episodes_per_level': [],
            'performance_per_level': []
        }

        self.enemy_configs = {
            1: {'type': 'straight', 'speed': 0.5, 'maneuver_freq': 0},
            2: {'type': 'random_turn', 'speed': 0.6, 'maneuver_freq': 0.1},
            3: {'type': 'zigzag', 'speed': 0.7, 'maneuver_freq': 0.3},
            4: {'type': 'reactive', 'speed': 0.8, 'maneuver_freq': 0.5},
            5: {'type': 'adversarial', 'speed': 0.9, 'maneuver_freq': 0.7}
        }
        
        print(f"Curriculum Manager initialized: Level {self.current_level}/{self.max_levels}")
        print(f"Progression type: {self.progression_type}")
        
    def get_enemy_config(self):
        """Get current enemy configuration"""
        return self.enemy_configs[self.current_level]
    
    def update_performance(self, episode_return, success_rate=None, steps_taken=None):
        """Update performance tracking"""
        if not self.enabled:
            return False
            
        self.episodes_at_level += 1

        if success_rate is None:
            success_rate = 1.0 if episode_return > 0 else 0.0
        
        self.performance_history.append(success_rate)

        should_advance = self._should_advance()
        
        if should_advance:
            self._advance_level()
            return True
        
        return False
    
    def _should_advance(self):
        """Determine if we should advance to next level"""
        if self.current_level >= self.max_levels:
            return False
        
        if self.episodes_at_level < self.min_episodes:
            return False
        
        if self.progression_type == 'threshold':
            if len(self.performance_history) >= self.eval_episodes:
                avg_performance = np.mean(self.performance_history)
                return avg_performance >= self.threshold
        
        elif self.progression_type == 'fixed_interval':
            return self.episodes_at_level >= self.fixed_interval
        
        return False
    
    def _advance_level(self):
        """Advance to next curriculum level"""
        prev_level = self.current_level
        avg_performance = np.mean(self.performance_history) if self.performance_history else 0

        self.curriculum_stats['level_transitions'].append(
            (prev_level, self.current_level + 1, self.episodes_at_level)
        )
        self.curriculum_stats['episodes_per_level'].append(self.episodes_at_level)
        self.curriculum_stats['performance_per_level'].append(avg_performance)

        self.current_level += 1
        self.episodes_at_level = 0
        self.performance_history.clear()
        
        print(f"\n{'='*50}")
        print(f"CURRICULUM LEVEL ADVANCED: {prev_level} → {self.current_level}")
        print(f"Previous level performance: {avg_performance:.3f}")
        print(f"New enemy configuration: {self.get_enemy_config()}")
        print(f"{'='*50}\n")
    
    def get_curriculum_stats(self):
        """Get curriculum learning statistics"""
        return {
            'current_level': self.current_level,
            'episodes_at_level': self.episodes_at_level,
            'current_performance': np.mean(self.performance_history) if self.performance_history else 0,
            'level_transitions': self.curriculum_stats['level_transitions'],
            'episodes_per_level': self.curriculum_stats['episodes_per_level'],
            'performance_per_level': self.curriculum_stats['performance_per_level']
        }
    
    def save_checkpoint(self, model, save_path):
        """Save curriculum checkpoint"""
        if not self.save_checkpoints:
            return
            
        checkpoint = {
            'current_level': self.current_level,
            'episodes_at_level': self.episodes_at_level,
            'performance_history': list(self.performance_history),
            'curriculum_stats': self.curriculum_stats,
            'model_state': model.main_network_our_agent.state_dict()
        }
        
        checkpoint_path = os.path.join(save_path, f'curriculum_checkpoint_level_{self.current_level}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Curriculum checkpoint saved: {checkpoint_path}")
