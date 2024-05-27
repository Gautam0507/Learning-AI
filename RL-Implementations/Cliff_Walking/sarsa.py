r"""°°°
# Importing all the libraries
°°°"""

# |%%--%%| <miMi46uRGL|EwH6BVebB4>

import matplotlib.pyplot as plt
import numpy as np

from env import CliffWalking

# |%%--%%| <EwH6BVebB4|vYl2a4EQ6A>
r"""°°°
# Creating the environment
°°°"""
# |%%--%%| <vYl2a4EQ6A|bvDEJFkFJO>

env = CliffWalking()
env.reset()
env.render()

# |%%--%%| <bvDEJFkFJO|kIobK3X5Zz>
r"""°°°
# Creating the Q(s,a) table
°°°"""
# |%%--%%| <kIobK3X5Zz|dlpev8L7NU>

action_values = np.zeros((48, 4))

# |%%--%%| <dlpev8L7NU|ejKjacumDB>

print(action_values)

# |%%--%%| <ejKjacumDB|Ma5zSYRPmB>
r"""°°°
# Creating the policy
°°°"""
# |%%--%%| <Ma5zSYRPmB|SU6oomGCXQ>


def policy(state, epsilon=0.2):
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        av = action_values[state]
        return np.argmax(av)


# |%%--%%| <SU6oomGCXQ|4wSUaeysVD>

print(f"The action chosen at state 0 is {policy(0)}")

# |%%--%%| <4wSUaeysVD|pmTASoBw1Q>
r"""°°°
# Implementing the algorithm
°°°"""
# |%%--%%| <pmTASoBw1Q|0D8pRLkAyi>


def sarsa(action_values, policy, episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.2):
    for episode in range(1, episodes + 1):
        state = env.reset()
        done, terminated = False, False
        action = policy(state, epsilon)
        print(episode)
        while not done or terminated:
            next_state, reward, done, terminated = env.step(action)
            next_action = policy(next_state, epsilon)

            qsa = action_values[state][action]
            next_qsa = action_values[next_state][next_action]

            action_values[state][action] = qsa + alpha * (
                reward + gamma * next_qsa - qsa
            )
            state, action = next_state, next_action


# |%%--%%| <0D8pRLkAyi|KQMK4DbTRE>

sarsa(action_values, policy)

# |%%--%%| <KQMK4DbTRE|VofMGXvf0r>
r"""°°°
# Showing the results
°°°"""
# |%%--%%| <VofMGXvf0r|CWKlZDqump>

print(action_values)

# |%%--%%| <CWKlZDqump|GhA9vmtOPs>


def test_agent(policy, episodes=1, epsilon=0):
    env.pygame_init()
    for episode in range(episodes):
        state = env.reset()
        done, terminated = False, False
        while not (done or terminated):
            action = policy(state, epsilon)
            next_state, reward, done, terminated = env.step(action)
            frame = env.render()
            state = next_state
        print(episode + 1)


# |%%--%%| <GhA9vmtOPs|ssOJOpTSWV>

test_agent(policy)

# |%%--%%| <ssOJOpTSWV|nW46fzoX42>

env.close()
