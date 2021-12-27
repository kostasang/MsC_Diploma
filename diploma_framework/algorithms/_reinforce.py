import torch, gym, copy, math, random
import numpy as np

from collections import deque
from typing import Union
import torch.nn as nn
import torch.optim as optim

from diploma_framework.evaluation import test_env

class Reinforce():

    """
    Class that implements REINFORCE algorithm

    """

    def __init__(self,
                environment : Union[object, str],
                model : nn.Module,
                lr : float = 1e-03,
                max_episodes : int = 150_000,
                num_steps : int = 150,
                gamma : float = 0.99) -> None :

        
        if isinstance(environment, str):
            self.env = gym.make(environment)
        else:
            self.env = environment
        
        self.model = model
        self.max_episodes = max_episodes
        self.num_steps = num_steps
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def run(self, 
            eval_window : int = 1000,
            n_evaluations : int = 10,
            early_stopping : bool = True) -> list :

        """
        Run REINFORCE algorithm with hyperparameters specified in arguments.
        Returns list of test rewards throughout the agent's training loop.

        """

        test_rewards = []
        frame_idx = 0
        early_stop = False

        while frame_idx < self.max_episodes and not early_stop:
            
            curr_state = self.env.reset()
            done = False
            transitions = []
            cumulative_reward = 0

            for _ in range(self.num_steps):
                
                act_prob = self.model(torch.from_numpy(curr_state).float().unsqueeze(dim=0))
                action = np.random.choice(np.array([0,1]), p=act_prob.squeeze(dim=0).data.numpy())
                prev_state = curr_state
                curr_state, reward, done, _ = self.env.step(action)
                frame_idx += 1
                cumulative_reward = reward + self.gamma * cumulative_reward
                transitions.append((prev_state, action, cumulative_reward))

                if frame_idx % eval_window == 0:
                    test_reward = np.mean([test_env(self.env, self.model, vis=False) for _ in range(n_evaluations)])
                    test_rewards.append(test_reward)
                    print(f'Frame : {frame_idx} - Test reward : {test_reward}')
                    if test_reward > 195.2 and early_stopping: early_stop = True

                if done:
                    break
            
            # Do not update is criterions for early stop are met
            if not (early_stopping and early_stop):
                
                # Get batches
                returns_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))
                returns_batch /= returns_batch.max()
                # List of numpy arrays to numpy and hen to Tensor for performance boost
                state_batch = torch.Tensor(np.asarray([s for (s,a,r) in transitions]))
                action_batch = torch.Tensor([a for (s,a,r) in transitions])
                pred_batch = self.model(state_batch)
                prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()
                # Calculate loss based on log prob and discounted rewards
                loss = self.criterion(prob_batch, returns_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return test_rewards
        
    def criterion(self, 
                  predicted_probs_batch : torch.Tensor,
                  returns_batch : torch.Tensor) -> float :
        """
        Reinforce algorithm loss function 

        """

        return -1 * torch.sum(returns_batch*torch.log(predicted_probs_batch))

