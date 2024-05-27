r"""°°°
## Creating the environement
°°°"""

# |%%--%%| <LPsx8O9au7|EfHSyufvSz>

import statistics

import gymnasium as gym
import matplotlib.pylab as plt
import numpy as np
import pygame
from IPython import display
from IPython.display import clear_output

from env import SimpleCorridor

# |%%--%%| <EfHSyufvSz|BnKbgj0puj>
r"""°°°
## Test Agent Function 
°°°"""
# |%%--%%| <BnKbgj0puj|0v7QHErL6L>


def test_agent(env: gym.Env, policy: callable, episodes: int = 10) -> None:
    for episode in range(episodes):
        temp = env.reset()
        tagent_pos = temp[0]
        ttarget_pos = temp[1]
        done = False

        env.render(mode="rgb_array")
        while not done:
            action = policy(tagent_pos, 0.1)

            next_state, _, done, _ = env.step(action)
            env.render()
            plt.axis("off")
            display.display(plt.gcf())
            display.clear_output(wait=True)

            tagent_pos = next_state[0]


# |%%--%%| <0v7QHErL6L|5nwzYEIp0J>
r"""°°°
## Initialising the environment 
°°°"""
# |%%--%%| <5nwzYEIp0J|TgPzPWonJf>

env = SimpleCorridor()
env.reset()
env.render()


# |%%--%%| <TgPzPWonJf|CU4dsJfEzX>
r"""°°°
## Q Value Table
°°°"""
# |%%--%%| <CU4dsJfEzX|YVFwY9uHzA>

action_values = np.zeros(shape=(15, 2))
print(action_values)

# |%%--%%| <YVFwY9uHzA|j7qgEeYhlK>
r"""°°°
### Creating the Policy
°°°"""
# |%%--%%| <j7qgEeYhlK|YWHrtX0c0K>


def policy(state, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.choice(2)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))


# |%%--%%| <YWHrtX0c0K|ykdfruagrZ>

action = policy(0, epsilon=0.5)
print(f"The action taken in state 0 is {action}")

# |%%--%%| <ykdfruagrZ|hKJN62Wsnn>
r"""°°°
#### Testing the policy
°°°"""
# |%%--%%| <hKJN62Wsnn|aUGXvzUKEL>
r"""°°°
# Implementing the algorithm
°°°"""
# |%%--%%| <aUGXvzUKEL|QK5RxLYJ6O>


def on_policy_mc_control(policy, action_values, episodes=1000, epsilon=0.1, gamma=0.99):
    sa_returns = np.empty((15, 2), dtype=object)

    for episode in range(0, episodes + 1):
        state, target = env.reset()
        done = False
        transitions = []

        while not done:
            action = policy(state, epsilon)
            (next_state, target), reward, done, info = env.step(action)
            transitions.append([state, action, reward])
            state = next_state

        G = 0

        for state_t, action_t, reward_t in reversed(transitions):
            G = reward_t + gamma * G
            if sa_returns[state_t, action_t] == None:
                sa_returns[state_t, action_t] = [G]
            else:
                sa_returns[state_t, action_t].append(G)
            action_values[state_t, action_t] = statistics.mean(
                sa_returns[state_t, action_t]
            )


# |%%--%%| <QK5RxLYJ6O|kpQ7BW0OvR>

on_policy_mc_control(policy, action_values)

# |%%--%%| <kpQ7BW0OvR|f6Ex2vJpJv>

test_agent(env, policy, episodes=10)
