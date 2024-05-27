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

# %%
import time

# %% jukit_cell_id="1HU8GD8cBx"
import gymnasium
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

# %% jukit_cell_id="HbZrFmdJCn"
env = gymnasium.make("Taxi-v3")
env.reset()
plt.imshow(env.render(mode="rgb_array"))

# %% jukit_cell_id="XuP39XTgFx"
