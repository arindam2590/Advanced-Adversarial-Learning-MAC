"""
Enemy Behavior Controller - Implements different enemy tactics based on curriculum level
"""
import numpy as np
import random

class EnemyBehaviorController:
    def __init__(self, config):
        self.config = config
        self.behavior_type = config['type']
        self.speed = config['speed']
        self.maneuver_freq = config['maneuver_freq']
        self.step_count = 0
        self.last_action = None
        
    def get_action(self, agent_position=None, enemy_position=None, action_space=None):
        """
        Get enemy action based on current curriculum level
        
        Args:
            agent_position: Position of pursuing agent (for reactive behavior)
            enemy_position: Current enemy position
            action_space: Available actions
        """
        self.step_count += 1
        
        if self.behavior_type == 'straight':
            return self._straight_behavior(action_space)
        elif self.behavior_type == 'random_turn':
            return self._random_turn_behavior(action_space)
        elif self.behavior_type == 'zigzag':
            return self._zigzag_behavior(action_space)
        elif self.behavior_type == 'reactive':
            return self._reactive_behavior(agent_position, enemy_position, action_space)
        elif self.behavior_type == 'adversarial':
            return self._adversarial_behavior(agent_position, enemy_position, action_space)
        else:
            return action_space.sample()
    
    def _straight_behavior(self, action_space):
        """Level 1: Straight line movement"""
        return 1 
    
    def _random_turn_behavior(self, action_space):
        """Level 2: Random turns at low frequency"""
        if random.random() < self.maneuver_freq:
            return action_space.sample()
        return 1 
    
    def _zigzag_behavior(self, action_space):
        """Level 3: Zigzag pattern"""
        cycle_length = 20
        phase = (self.step_count % cycle_length) / cycle_length
        
        if phase < 0.25:
            return 2
        elif phase < 0.5:
            return 1 
        elif phase < 0.75:
            return 3 
        else:
            return 1 
    
    def _reactive_behavior(self, agent_pos, enemy_pos, action_space):
        """Level 4: React to pursuing agent"""
        if agent_pos is None or enemy_pos is None:
            return self._random_turn_behavior(action_space)

        dx = enemy_pos[0] - agent_pos[0]
        dy = enemy_pos[1] - agent_pos[1]
        
        if random.random() < self.maneuver_freq:
            if abs(dx) > abs(dy):
                return 2 if dx > 0 else 3  
            else:
                return 1 if dy > 0 else 4  
        
        return 1 
    
    def _adversarial_behavior(self, agent_pos, enemy_pos, action_space):
        """Level 5: Advanced adversarial behavior with decoys"""
        
        if random.random() < 0.3:  
            return action_space.sample()
        else:
            return self._reactive_behavior(agent_pos, enemy_pos, action_space)