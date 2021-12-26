from typing import Union

import torch, gym, copy
import torch.nn as nn

@torch.inference_mode()
def test_env(env : Union[object, str],
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
        action = model.infer_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if vis: env.render()
        total_reward += reward
    env.close()
    return total_reward