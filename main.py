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


def arr2model_input(state):
    transforms = T.Compose(
        [T.Grayscale(), T.Resize((84, 84)), T.Normalize(0, 255)]
    )

    obs = np.transpose(state, (2, 0, 1))
    obs = torch.tensor(obs.copy(), dtype=torch.float) 
    obs = transforms(obs).squeeze(0)

    return obs

#%%
if __name__ == "__main__":
    org_env = initialize_mario()
    
    env = SkipFrame(org_env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    save_dir = Path("assets/checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, load_dir="/Users/wonhyung64/Downloads/trained_mario.chkpt")

    # logger = MetricLogger(save_dir)

    state, _ = org_env.reset()
    height, width, layers = state.shape

    out = cv2.VideoWriter(f"{save_dir}/episode.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))
    out.write(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))

    for _ in range(10):
        state, _ = org_env.reset()
        height, width, layers = state.shape
        while True:
            action = mario.act(arr2model_input(state))
            next_state, reward, done, trunc, info = org_env.step(action=action)
            out.write(cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR))
            state = next_state
            if done or info["flag_get"]:
                break
    out.release()

    state = env.reset()
env.reset()
        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

# %%

episodes = 50000
for e in range(episodes):
    # if e // 500 == 0:
    #     state, _ = org_env.reset()
    #     height, width, layers = state.shape
    #     out = cv2.VideoWriter(f"{save_dir}/episode_{e+1}.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))
    #     out.write(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))

    #     for _ in range(10):
    #         state, _ = org_env.reset()
    #         height, width, layers = state.shape
    #         while True:
    #             action = mario.act(arr2model_input(state))
    #             next_state, reward, done, trunc, info = org_env.step(action=action)
    #             out.write(cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR))
    #             state = next_state
    #             if done or info["flag_get"]:
    #                 break
    #     out.release()

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()
    logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

# %%

#%%
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 10
for e in range(episodes):

    state = env.reset()

if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
else:
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb', apply_api_compatibility=True)

# 상태 공간을 2가지로 제한하기
#   0. 오른쪽으로 걷기
#   1. 오른쪽으로 점프하기
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")