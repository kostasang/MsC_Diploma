
import gym, torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from typing import Union
from gym_duckietown.envs import DuckietownEnv

class PPO():

    """
    Implements PPO algorithm

    """

    def __init__(self, 
                 environment : Union[str, object],
                 model : nn.Module
                 ):

        if isinstance(environment, str):
            self.env = gym.make(environment)
        else:
            self.env = environment

        self.model = model

