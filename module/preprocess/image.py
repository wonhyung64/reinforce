import gym
import torch
import numpy as np
from gym.spaces import Box
from torchvision import transforms as T
from nes_py.wrappers import JoypadSpace
from typing import Tuple, List, Union


class SkipFrame(gym.Wrapper):
    def __init__(self, env: JoypadSpace, skip_num: int) -> None:
        """
        Return only every 'skip_num'-th frame

        Args:
            env (JoypadSpace): environment object
            skip_num (int): number of frames to skip
        """
        super().__init__(env)
        self._skip_num = skip_num

    def skip(self, action: int) -> Tuple:
        """
        Repeat the same action and accumulate rewards

        Args:
            action (int): selected action

        Returns:
            Tuple: last transtion after 'skip_num' skipping
        """
        total_reward = 0.
        for _ in range(self._skip_num):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: JoypadSpace):
        """
        Convert rgb to gray scale img

        Args:
            env (JoypadSpace): environment object
        """
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def _numpy2torch(self, obs: np.ndarray) -> torch.Tensor:
        """
        Permute [H,W,C] to [C,H,W] tensor and convert to torch.Tensor

        Args:
            obs (np.ndarray): image state

        Returns:
            torch.Tensor: converted image state
        """
        obs = np.transpose(obs, (2, 0, 1))
        obs = torch.tensor(obs.copy(), dtype=torch.float) 
        return obs

    def convert(self, obs: np.ndarray) -> torch.Tensor:
        """
        Convert [H,W,C] to [C,H,W], nd.array to torch.Tensor, RGB to Gray

        Args:
            obs (np.ndarray): RGB image ndarray 

        Returns:
            torch.Tensor: converted image
        """
        obs = self._numpy2torch(obs)
        transform = T.Grayscale()
        obs = transform(obs)
        return obs


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: JoypadSpace, shape: Union[int, Tuple, List]):
        """
        Resize img

        Args:
            env (JoypadSpace): envionment object
            shape (Union[int, Tuple, List]): shape to resize
        """
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def convert(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Convert image size

        Args:
            obs (torch.Tensor): image to resize

        Returns:
            torch.Tensor: resized image
        """

        transforms = T.Compose(
            [T.resize(self.shape), T.Normalize(0, 255)]
        )
        obs = transforms(obs).squeeze(0)
        return obs
