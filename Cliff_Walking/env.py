import gymnasium as gym
import numpy as np
import pygame


class CliffWalking(gym.Env):
    def __init__(self, render_mode="human"):
        # observation space
        self.observation_space = gym.spaces.MultiDiscrete([3, 11])
        # action space
        self.action_space = gym.spaces.Discrete(4)
        # start end and holes constant positions
        self.START_POS = [3, 0]
        self.ENDPOS = [3, 11]
        self.HOLES = []
        for i in range(1, 11):
            self.HOLES.append([3, i])

        # Basic reset
        self.state = self.START_POS
        self.target = self.ENDPOS
        # up, right, down, left
        self.action_to_direction = {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1]
        }
        self.square_size = 64
        self.window_size = (self.square_size * 12, self.square_size * 4)
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self):
        self.state = self.START_POS.copy()
        self.target = self.ENDPOS
        return self._calc_state(self.state[0], self.state[1])

    # Checks if the state is a hole
    def _check_holes(self, state):
        if state[0] == 3 and state[1] > 0 and state[1] < 11:
            return True
        else:
            return False

    def _calc_next_state(self, state, action):
        direction = self.action_to_direction[action]
        next_state = [0, 0]
        next_state[0] = np.clip(state[0] + direction[0],
                                0,3)
        next_state[1] = np.clip(state[1] + direction[1],
                                0, 11)
        return next_state

    def _calc_reward(self, state):
        if np.array_equal(state, self.target):
            return 0
        elif self._check_holes(state):
            return -100
        else:
            return -1

    def _calc_done(self, state):
        return np.array_equal(state, self.target)

    def _calc_terminated(self, state):
        return self.state in self.HOLES

    def _calc_state(self, row, col):
        state = (row * 12) + col
        return state

    def _calc_row_col(self, state):
        row, col = divmod(state, 12)
        row_col_state = [row, col]
        return row_col_state

    def step(self, action):
        next_state = self._calc_next_state(self.state, action)
        self.state = np.copy(next_state)
        reward = self._calc_reward(self.state)
        done = self._calc_done(self.state)
        terminated = self._check_holes(self.state)
        return self._calc_state(self.state[0], self.state[1]), reward, done, terminated

    def simulate_step(self, state_num, action):
        state = self._calc_row_col(state_num)
        next_state = self._calc_next_state(state, action)
        state = np.copy(next_state)
        reward = self._calc_reward(state)
        done = self._calc_done(state)
        terminated = self._check_holes(state)
        return self._calc_state(next_state[0], next_state[1]), reward, done, terminated

    def render(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        pygame.draw.circle(canvas, (255, 0, 0), ((
            self.target[1] + 0.5)*self.square_size, (self.target[0] + 0.5) * self.square_size), self.square_size/3)
        pygame.draw.circle(canvas, (0, 0, 255), ((
            self.state[1] + 0.5)*self.square_size, (self.state[0] + 0.5) * self.square_size), self.square_size/3)

        for x in range(13):
            pygame.draw.line(canvas, (0, 0, 0), (self.square_size * x, 0),
                             (self.square_size * x, self.window_size[0]), width=3)
        for x in range(5):
            pygame.draw.line(canvas, 0, (0, self.square_size * x),
                             (self.window_size[0], self.square_size * x), 3)
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(4)
            return None
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    def pygame_init(self):
        self.window = None
        self.clock = None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()