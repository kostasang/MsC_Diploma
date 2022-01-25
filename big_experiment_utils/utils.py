from gym_duckietown.simulator import Simulator
from big_experiment_utils.wrappers import *
from typing import Union
from collections import deque

import torch
import torch.nn as nn

@torch.inference_mode()
def test_duckietown(env : Union[object, str],
                    model : nn.Module,
                    vis : bool = False,
                    stacked_frames : int = 5) -> float :
    
    
    env = Simulator(
        seed=None,  # random seed
        map_name="loop_empty",
        max_steps=10_000,  # we don't want the gym to reset itself
        domain_rand=False,
        distortion=False,
        camera_width=80,
        camera_height=60
    )
    #env = DtRewardWrapper(env)
    env = DiscreteWrapper(env)

    frame = env.reset()
    stacked_frames = deque([torch.zeros(size=frame.shape).unsqueeze(0)]*stacked_frames,
                           maxlen=stacked_frames)

    if vis: env.render()
    done = False
    total_reward = 0
    total_frames = 0
    while not done:
        frame = torch.FloatTensor(frame).unsqueeze(0)
        stacked_frames.append(frame)
        state = torch.cat(tuple(stacked_frames), dim=-1)
        action = model.infer_action(state)
        next_frame, reward, done, _ = env.step(action)
        frame = next_frame
        if vis: env.render()
        total_reward += reward
        total_frames += 1
    env.close()
    return total_reward, total_frames