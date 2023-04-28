#%%
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import os
import gym
import copy
import random
import datetime
import gym_super_mario_bros

from collections import deque
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from typing import Tuple

from module import (
    initialize_mario,
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation
)



#%%
'''version check and initiailize env'''

if __name__ == "__main__":
    env = initialize_mario()
    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    env = SkipFrame(env, skip_num=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

