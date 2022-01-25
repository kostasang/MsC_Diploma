import gym
from gym import spaces
import numpy as np

class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            pass
        else:
            reward = 0
        return reward

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [-0.6, 0.6]
        # Turn right
        elif action == 1:
            vels = [0.6, -0.6]
        # Go forward
        elif action == 2:
            vels = [0.7, 0.7]
        else:
            assert False, "unknown action"
        return np.array(vels)

    def reverse_action(self, action):
        raise NotImplementedError()