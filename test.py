#%%
import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
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
    ResizeObservation,
    Mario,
    MetricLogger,
)


#%%
"""GRAY ENV"""
if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

env = JoypadSpace(env, [["right"], ["right", "A"]])
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)

if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

"""COLOR ENV"""
if gym.__version__ < '0.26':
    color_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    color_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

color_env = JoypadSpace(color_env, [["right"], ["right", "A"]])
color_env = SkipFrame(color_env, skip=4)

if gym.__version__ < '0.26':
    color_env = FrameStack(color_env, num_stack=4, new_step_api=True)
else:
    color_env = FrameStack(color_env, num_stack=4)


use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

# save_dir = Path("assets/checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# save_dir.mkdir(parents=True)
weights_dir = "/Users/wonhyung64/Github/reinforce/assets/checkpoints/2023-12-14T21-48-10"

out = cv2.VideoWriter(f"{weights_dir}/color.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 60, (256, 240))
out_gray = cv2.VideoWriter(f"{weights_dir}/gray.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 60, (84, 84), False)

# for ckpt_name in ["mario_net_1", "mario_net_6", "mario_net_10", "mario_net_13"]:
#     mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=weights_dir, load_dir=f"{weights_dir}/{ckpt_name}.chkpt")
for ckpt_name in range(1,14):
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=weights_dir, load_dir=f"{weights_dir}/mario_net_{ckpt_name}.chkpt")


    episodes = 1
    for e in range(episodes):

        color_state = color_env.reset()
        state = env.reset()

        for f in color_state[0]._frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        for f in state[0]._frames:
            out_gray.write((f.numpy()*255).astype(np.uint8))

        # 게임을 실행시켜봅시다!
        while True:
            action = mario.act(state)

            # 현재 상태에서 에이전트 실행하기
            color_state, color_reward, color_done, color_trunc, color_info = color_env.step(action)
            state, reward, done, trunc, info = env.step(action)

            for f in color_state._frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            for f in state._frames:
                out_gray.write((f.numpy()*255).astype(np.uint8))

            if done or info["flag_get"]:
                break

out.release()
out_gray.release()

# %%
