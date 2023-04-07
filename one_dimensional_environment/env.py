from typing import Tuple, Dict, Optional, Iterable
import gymnasium as gym
from gym import spaces
import pygame 
import numpy as np 

class LineWorldEnv(gym.Env):

    def __init__(self, render_mode = None, size: int = 15) -> None:
        self.size = size #No of positions in the line 

        #observation space 1-D array of values 
        self.observation_space = spaces.Discrete(size - 1, start = 0)

        #actionspace: we have only 2 actions corresponding to left and right
        self.action_space = spaces.Discrete(n = 2)

        #0 corresponds to left and 1 corresponds to right 
        self._action_to_direction = {
            0: np.array([0]),
            1: np.array([1])
        }
        self.goal = size - 1
        self.maze = self._create_maze(size=size)

    def reset(self) -> int:
        """
        Reset the environment to execute a new episode.

        Returns: State representing the initial position of the agent.
        """

        self.state = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Take an action in the environment and observe the next transition.

        Args:
            action: An indicator of the action to be taken.

        Returns:
            The next transition.
        """
        reward = self.compute_reward(self.state,action)
        self.state = self._get_next_state(self.state, action)
        done = self.state == self.goal
        info ={}
        return self.state, reward, done, info
    
    def compute_reward(self, state: int, action: int) -> float:
        """
        Compute the reward attained by taking action 'a' at state 's'.

        Args:
            state: the state of the agent prior to taking the action.
            action: the action taken by the agent.

        Returns:
            A float representing the reward signal received by the agent.

        """
        next_state = self._get_next_state(state, action)
        return -float(state != self.goal)
    
    def _get_next_state(self, state: int, action: int) -> int:
        """
        Gets the next state after the agent performs action 'a' in state 's'. If there is a
        wall in the way, the next state will be the same as the current.

        Args:
            state: current state (before taking the action).
            action: move performed by the agent.

        Returns: a State instance representing the new state.
        """
        if action == 0: 
            next_state = state - 1
        elif action == 1:
            next_state = state + 1
        else:
            raise ValueError("Action value not supported:", action)
        if next_state in self.maze[state]:
            return next_state
        return state

    def simulate_step(self, state: int, action: int):
        """

        Simulate (without taking) a step in the environment.

        Args:
            state: the state of the agent prior to taking the action.
            action: the action to simulate the step with.

        Returns:
            The next transition.

        """
        reward = self.compute_reward(state, action)
        next_state = self._get_next_state(state, action)
        done = next_state == self.goal
        info = {}
        return next_state, reward, done, info
    
    @staticmethod
    def _create_maze(size:int) -> Dict[int, Iterable]
        

    







env = LineWorldEnv()