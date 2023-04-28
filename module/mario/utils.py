import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace


def initialize_mario() -> JoypadSpace:
    """
    Initialize Mario environment with version check

    Returns:
        JoypadSpace: environment object
    """
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3", render_mode="rbg", apply_api_compatibility=True)
    if gym.__version__ < "0.26":
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    return env
