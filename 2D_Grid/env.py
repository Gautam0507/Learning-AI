import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import matplotlib.pyplot as plt


class Grid_World(gym.Env):
    def __init__(self, size=4, type="fixed", render_mode="human") -> None:
        self.size = size  # The size of the environment
        self.type = type  # Defines if the environment is fixed or random
        self.window_size = 512  # Pygame rendering window size
        self.window = None  # pygame window
        self.clock = None  # reference to the clock that is used
        self.render_mode = render_mode  # pygame or rgb array

        # We have 4 actions corresponding to 'keft', 'up', 'right', 'down'
        self.action_space = spaces.Discrete(4)

        # Defining the observation space
        self.observation_space = spaces.Box(0, size-1, shape=(2,), dtype=int)

        # Dictionary defining the action to what happens to the position of the agent if action is taken
        self.action_to_direction = {
            0: np.array([-1, 0]),  # left
            1: np.array([0, -1]),  # up
            2: np.array([1, 0]),  # right
            3: np.array([0, 1])  # down
        }
        # When it is in fixed mode the agent is at the starting position
        if self.type == "fixed":
            self.start_position = (0, 0)
            self.target = (size - 1, size - 1)
        # When the type is random the postions of the agent and the target are randomized
        elif self.type == "random":
            while True:  # Cycle random positions as long as start and target are not the same
                self.start_position = self.observation_space.sample()
                self.target = self.observation_space.sample()
                if not (np.array_equal(self.start_position, self.target)):
                    break

    def reset(self):
        # Returning the agent to the starting position
        self.agent = self.start_position

        if self.render_mode == "human":
            self._render_frame()

        return (self.agent, self.target)

    def step(self, action):
        # mapping the direction to the array that needs to be added to position if the action is taken
        direction = self.action_to_direction[action]
        # making sure that the values stay within the borders of the maze
        self.agent = np.clip(self.agent + direction, 0, self.size - 1)
        # Computing the value of done by ckecking is agent pos == target pos
        done = np.array_equal(self.agent, self.target)
        reward = 1 if done else 0  # 1 if agent reaches goal else gets a 0
        info = []  # Any extra information to be given back

        if self.render_mode == "human":
            self._render_frame()

        return self.agent, reward, done, info

    def simulate_step(self, state, action):
        #getting the direction to add to the postion if action is taken
        direction = self.action_to_direction[action]
        #calculating the next step
        next_state = np.clip(state + direction, 0, self.size - 1)
        done = np.array_equal(self.agent, self.target)
        reward = 1 if done else 0
        info = []

        if self.render_mode == "human":
            self._render_frame()

        return next_state, reward, done, info

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Size of 1 square grid
        pix_square_size = (self.window_size / self.size)
        # Drawing the target
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(
            (pix_square_size * self.target[0], pix_square_size * self.target[1], pix_square_size, pix_square_size),),)
        # Drawing the agent
        pygame.draw.circle(canvas, (0, 0, 255), ((
            self.agent[0] + 0.5) * pix_square_size, (self.agent[1] + 0.5)*pix_square_size), pix_square_size/3)

        # Adding the gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
        # Pygame render updates
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(4)
            return None

        # else rgb array
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def py_init(self):  # Used running it multiple times after closing the window
        self.window = None
        self.clock = None

    def close(self):  # Finally closes the opened pygame window
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
