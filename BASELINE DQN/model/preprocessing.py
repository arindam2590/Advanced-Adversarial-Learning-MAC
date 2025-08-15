import cv2
import numpy as np
from collections import deque


def preprocess_state(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    normalized_frame = resized / 255.0
    return normalized_frame


class FrameStack:
    def __init__(self, agents, stack_size=4):
        self.agents = agents       
        self.stack_size = stack_size
        self.frames_our_agent = deque(maxlen=stack_size)
        self.frames_opponent_agent = deque(maxlen=stack_size)

    def init_stack(self, frame, agents):
        for agent in self.agents:
            processed = preprocess_state(frame[agent])  
            if agent == self.agents[0]:
                for _ in range(self.stack_size):
                    self.frames_our_agent.append(processed)
            else:
                for _ in range(self.stack_size):
                    self.frames_opponent_agent.append(processed) 

    def update_frame_stack(self, frame):
        state = {agent: None for agent in self.agents}
        for agent in self.agents:
            processed = preprocess_state(frame[agent])
            if agent == self.agents[0]:
                self.frames_our_agent.append(processed)
            else:
                self.frames_opponent_agent.append(processed)
            state[agent] = self.get_state(agent)
        return state
        
    def get_state(self, agent):
        if agent == self.agents[0]:
            return np.stack(self.frames_our_agent, axis=0)
        else:
            return np.stack(self.frames_opponent_agent, axis=0)

