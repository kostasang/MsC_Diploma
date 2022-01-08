import torch, gym, logging
import numpy as np

from tqdm import tqdm
from typing import Union
import torch.nn as nn
import torch.optim as optim

from diploma_framework.algorithms._generic import DeepRLAlgorithm
from diploma_framework.evaluation import test_env

logger = logging.getLogger('deepRL')

class Reinforce(DeepRLAlgorithm):

    """
    Class that implements REINFORCE algorithm

    """

    def __init__(self,
                environment : Union[object, str],
                model : nn.Module,
                lr : float = 1e-03,
                max_frames : int = 150_000,
                num_steps : int = 150,
                gamma : float = 0.99) -> None :

        
        if isinstance(environment, str):
            self.env = gym.make(environment)
        else:
            self.env = environment
        
        self.model = model
        self.max_frames = max_frames
        self.num_steps = num_steps
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def run(self, 
            eval_window : int = 1000,
            n_evaluations : int = 10,
            early_stopping : bool = True,
            reward_threshold : float = 197.5) -> list :

        """
        Run REINFORCE algorithm with hyperparameters specified in arguments.
        Returns list of test rewards throughout the agent's training loop.

        """

        logger.info('Initializing training')
        test_rewards = []
        frame_idx = 0
        early_stop = False

        with tqdm(total = self.max_frames) as pbar:
            while frame_idx < self.max_frames and not early_stop:
                
                curr_state = self.env.reset()
                done = False
                transitions = []
                cumulative_reward = 0

                for _ in range(self.num_steps):
                    
                    act_dist, action = self.model.infer_step(torch.from_numpy(curr_state).float().unsqueeze(dim=0))
                    prev_state = curr_state
                    curr_state, reward, done, _ = self.env.step(action)
                    frame_idx += 1
                    cumulative_reward = reward + self.gamma * cumulative_reward
                    transitions.append((prev_state, action, cumulative_reward))

                    if frame_idx % eval_window == 0:
                        test_reward = np.mean([test_env(self.env, self.model, vis=False) for _ in range(n_evaluations)])
                        test_rewards.append(test_reward)
                        pbar.update(eval_window)
                        pbar.set_description(f'Cumulative reward {test_reward}')
                        if test_reward > reward_threshold and early_stopping: 
                            early_stop = True
                            logger.info('Early stopping criteria met')

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
                    pred_dist_batch = self.model.infer_batch(state_batch)
                    prob_batch = pred_dist_batch.log_prob(action_batch)

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
        # log is already applied to predicted_probs_batch
        return -1 * torch.sum(returns_batch*predicted_probs_batch)

