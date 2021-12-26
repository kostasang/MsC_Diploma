from typing import Union
from gym_duckietown.envs import DuckietownEnv

import torch, gym, copy
import torch.nn as nn


def test_env(env : Union[DuckietownEnv, str],
             model : nn.Module,
             vis : bool = False) -> float :

    """
    Used for testing the so far trained model. It calculates the total reward 
    received when model operates on the environment.

    """

    if isinstance(env, str):
        env = gym.make(env)
    else:
        env = copy.deepcopy(env)
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, _ = model(state)
        dist = torch.distributions.Categorical(logits=dist)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    env.close()
    return total_reward