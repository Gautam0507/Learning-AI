import gymnasium as gym  
import pygame 
import numpy as np
from IPython.display import clear_output
from IPython import display
import random
import matplotlib.pylab as plt
import copy
import time

class SimpleCorridor(gym.Env):
    def __init__(self, size = 15):
        self.length = size
        
        self.start_pos = 0
        self.end_pos = self.length -1
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, self.length, shape=(2,), dtype= int)


env = SimpleCorridor()
print()