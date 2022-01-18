from typing import Union

import torch, gym, copy
import torch.nn as nn

@torch.inference_mode()
def test_env(env : Union[object, str],
             model : nn.Module,
             vis : bool = False) -> float :

    """
    Used for testing the so far trained model. It calculates the total reward 
    received when model operates on the environment and the total number of frames.

    """

    if isinstance(env, str):
        env = gym.make(env)
    else:
        #env = copy.deepcopy(env)
        pass
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    total_frames = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action = model.infer_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if vis: env.render()
        total_reward += reward
        total_frames += 1
    env.close()
    return total_reward, total_frames