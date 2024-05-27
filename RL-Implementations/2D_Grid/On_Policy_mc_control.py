# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # On Policy Monte-Carlo control

# %% [markdown]
# # Importing all the libraries
# ## Hello

# %%

import random
import time
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from IPython.display import clear_output

# %%
from env import Grid_World

# %% [markdown]
# # Creating the environment

# %%
envsize = 4
Render_mode = "rgb_array"
env = Grid_World(size=envsize, type="fixed", render_mode=Render_mode)

# %%
agent, target = env.reset()
if env.render_mode == "rgb_array":
    frame = env.render()
    plt.imshow(frame)
    plt.axis = "off"
elif env.render_mode == "human":
    env.py_init()
    env.render()
    time.sleep(5)
    env.close()


# %% [markdown]
# # Creating the Q(s|a) value table

# %%
action_values = np.random.rand(envsize, envsize, 4)

# %% [markdown]
# ### Testing the value with state (0,0)

# %%
ap = action_values[0, 0]
np.argmax(ap)


# %% [markdown]
# # Defining the policy


# %%
def policy(state, epsilon=0.1):
    action_probablities = action_values[state[0], state[1]]
    if random.uniform(0, 1) < epsilon:
        # Choosing a random action
        action = np.random.choice(4)
    else:
        # Choosing action according to the q-values
        action = np.argmax(action_probablities)
    return action


# %% [markdown]
# ### Testing the policy with position (0,0)

# %%
action_values[0, 0]

# %% [markdown]
# # Implementing the algorithm

# %%
env.render_mode = "rgb_array"


# %%
def on_policy_mc_control(
    policy: callable,
    action_values,
    episodes: int = 10000,
    gamma: int = 0.9,
    epsilon: int = 0.1,
) -> None:

    sa_returns = np.empty(shape=(envsize, envsize, 4), dtype=object)

    for episode in range(0, episodes + 1):
        trajectory = []
        state, _ = env.reset()
        done = False
        while not done:
            action = policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            trajectory.append([state, action, reward])
            state = next_state

        G = 0

        for state_t, action_t, reward_t in reversed(trajectory):
            G = reward_t + gamma * G
            if sa_returns[state_t[0], state_t[1], action_t] == None:
                sa_returns[state_t[0], state_t[1], action_t] = [G]
            else:
                sa_returns[state_t[0], state_t[1], action_t].append(G)
            action_values[state_t[0], state_t[1], action_t] = mean(
                sa_returns[state_t[0], state_t[1], action_t]
            )

        print(episode)  # Only for tracking progress


# %%
on_policy_mc_control(policy, action_values)

# %%
print(action_values)


# %%
def test_agent(policy, episodes=1, epsilon=0):
    env.render_mode = Render_mode
    if env.render_mode == "human":
        env.py_init()

    for episode in range(0, episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy(state, epsilon)
            state, _, done, _ = env.step(action)
            frame = env.render()
            if env.render_mode == "rgb_array":
                plt.imshow(frame)
                plt.axis = "off"
                display.display(plt.gcf())
                display.clear_output(wait=True)
    if env.render_mode == "human":
        env.close()


# %%
test_agent(policy, episodes=5)
