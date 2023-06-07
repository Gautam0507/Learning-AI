import gymnasium as gym
import numpy as np 
import pygame
import matplotlib.pyplot as plt

class SimpleCorridor(gym.Env):
    def __init__(self, size=15, render_mode = "rgb_array"):
        self.length = size
        self.start_pos = 0
        self.end_pos = 14
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            0.0, self.length, shape=(2,), dtype=int)
        self.mode = render_mode
        self.width = 20

    def reset(self):
        self.current_pos = self.start_pos
        return np.array([self.current_pos, self.end_pos])

    def step(self, action):
        if action == 0 and self.current_pos > 0:
            self.current_pos -= 1
        if action == 1 and self.current_pos < self.length - 1:
            self.current_pos += 1
        done = self.current_pos == self.end_pos
        reward = 1 if done else -1
        info = {}
        return np.array([self.current_pos, self.end_pos]), reward, done, info

    def simulate_step(self, tpos, action):
        if action == 0 and tpos > 0:
            tpos -= 1
        if action == 1 and tpos < self.length -1:
            tpos += 1
            
        done = tpos == self.end_pos
        reward = 1 if done else -1
        info = {}
        return tpos, reward, done, info
        
    def render(self, mode="rgb_array"):
        canvas = pygame.Surface((self.width, self.width * self.length + 1))
        canvas.fill((255,255,255))
        for i in range(self.length+1):
            pygame.draw.line(canvas, 0, (0, 20*(i)), (20, 20*(i)), width=1)
        pygame.draw.line(canvas, 0, (0,0), (0, 20*(i)), width=1)
        pygame.draw.line(canvas, 0, (19,0), (19, 20*(i)), width=1)
        pygame.draw.circle(canvas, (0, 0, 255), (10,self.current_pos*20+10), 7)
        pygame.draw.circle(canvas, (255, 0, 0), (10,(self.end_pos)*20+10), 7)
        plArray = np.array(pygame.surfarray.pixels3d(canvas))
        plt.imshow(plArray)        
        plt.axis("off")