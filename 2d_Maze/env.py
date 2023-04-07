import gymnasium as gym
import numpy as np
from IPython.display import clear_output
from IPython import display
import matplotlib.pylab as plt
from typing import Tuple
from gym import spaces


class Grid(gym.Env):


    def __init__(self, size=4):
        self.row = size
        self.col = size
        self.startpos = (0, 0)
        self.state = self.startpos
        self.goal = (self.row - 1, self.col - 1)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([size, size])

    def reset(self):
        self.current_pos = self.startpos
        return np.array(self.current_pos)

    def step(self, action: int):

        reward = self.compute_reward(self.state, action)
        self.state = self._get_next_state(self.state, action)
        done = self.state == self.goal
        info = {}
        return self.state, reward, done, info
    
    def simulate_step(self, state, action):
        reward = self.compute_reward(state,action)
        next_state = self._get_next_state(state,action)
        done = next_state == self.goal
        info = {}
        return next_state, reward, done, info


    def compute_reward(self, state, action):
        if state == self.goal:
            return 1
        else:
            return 0

    def _get_next_state(self, state, action):
        if action == 0:
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            next_state = (state[0], state[1] + 1)
        elif action == 2:
            next_state = (state[0] + 1, state[1])
        elif action == 3:
            next_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Action value not supported:", action)
        if next_state in self.observation_space:
            return next_state
        else: 
            return state
        

def test_agent(env: gym.Env, policy: callable, epsidoes:int = 10)-> None:
    for epidode in range(epsidoes):
        agent_pos = env.reset()
        done = False

        while not done: 
            p = policy(agent_pos)
            action = np.random.choice(4, p = p)

            next_state, _, done, _ = env.step(action)
            agent_pos = next_state
    