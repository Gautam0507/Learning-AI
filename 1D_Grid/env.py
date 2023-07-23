import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display


class SimpleGridWorld(gym.Env):
    metadata = {'render_modes': ["human", 'rgb_array'], 'render_fps': 4}

    def __init__(self, size: int = 5, start: int = 0, end: int = 4):
        self.size = size
        self.observation_space = spaces.Discrete(n=self.size)
        self.action_space = spaces.Discrete(n=2)
        self.START_POS = start
        self.END_POS = end
        self.state = self.START_POS
        self.target = self.END_POS
        self.action_to_direction = {
            0: -1,
            1: 1
        }

        # Pygame declarations
        self.pix_sq_size = 32  # Size of each square in the pygame window

    def reset(self):
        self.state = self.START_POS
        self.target = self.END_POS
        return self.state, self.target

    def step(self, action: int):
        direction = self.action_to_direction[action]
        self.state = np.clip(self.state + direction, 0, self.size - 1)
        reward = 0 if self.state == self.target else -1
        done = True if self.state == self.END_POS else False
        return self.state, reward, done

    def simulate_step(self, state: int, action: int):
        direction = self.action_to_direction[action]
        state = np.clip(state + direction, 0, self.size - 1)
        reward = 0 if state == self.target else -1
        done = True if state == self.END_POS else False
        return state, reward, done

    def _render_frame(self):
        backgorund_surface = pygame.Surface(
            (self.size * self.pix_sq_size, self.pix_sq_size))
        backgorund_surface.fill((255, 255, 255))
        pygame.draw.rect(backgorund_surface, (255, 0, 0), pygame.Rect(
            self.pix_sq_size * self.target, 0, 32, 32),)
        pygame.draw.circle(backgorund_surface, (0, 0, 255),
                           ((self.state + 0.5) * self.pix_sq_size, self.pix_sq_size/2), self.pix_sq_size/2)
        for x in range(0, self.size + 1):
            pygame.draw.line(backgorund_surface, color=(0, 0, 0), start_pos=(
                x * self.pix_sq_size, 0), end_pos=(x * self.pix_sq_size, self.pix_sq_size), width=2)
        pygame.draw.line(backgorund_surface, 0, (0, 0),
                         (self.END_POS * self.pix_sq_size, 0), width=3)
        pygame.draw.line(backgorund_surface, 0, (0, self.pix_sq_size),
                         (self.END_POS * self.pix_sq_size, self.pix_sq_size), width=3)
        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(backgorund_surface)), axes=(1, 0, 2))
        plt.axis('off')
        plt.imshow(frame)
        display.display(plt.gcf())
        display.clear_output(wait=True)

    def test_agent(self, policy, episodes):
        for episode in range(1, episodes+1):
            done = False
            state, target = self.reset()
            while not done:
                action = policy(state)
                state, reward, done = self.step(action)
                self.render()

    def render(self):
        return self._render_frame()


if __name__ == "__main__":
    pass
