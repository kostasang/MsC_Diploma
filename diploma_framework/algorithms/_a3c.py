from multiprocessing import process
import torch, gym, copy
import numpy as np, multiprocessing as mp

from typing import Union
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer
from torch.nn import functional as F 

from diploma_framework.evaluation import test_env

test_lock = mp.Lock()


class A3C():

    """
    Implements A3C algorithm

    """

    def __init__(self,
                 environment : Union[object, str],
                 model : nn.Module,
                 n_workers : int = 8,
                 lr : float = 1e-03,
                 max_frames : int = 150000,
                 num_steps : int = 200,
                 actor_weight : float = 1,
                 critic_weight : float = 0.1,
                 gamma : float = 0.99) -> None :

        if isinstance(environment, str):
            self.env = gym.make(environment)
        else:
            self.env = environment

        self.model = model
        # Move model parameters to shared memory
        self.model.share_memory()

        self.n_workers = n_workers
        self.lr = lr
        self.max_frames = max_frames
        self.num_steps = num_steps
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.gamma = gamma

        # Shared variable to indicate early stopping
        self.early_stop = mp.Value('B', False)
    
    def run(self,
            eval_window : int = 1000,
            n_evaluations : int = 10,
            early_stopping : bool = True,
            reward_threshold : float = 197.5) -> list :

        """
        Run A3C algorithm with hyperparameters specified in arguments.
        Returns list of test rewards throughout the agent's training loop.

        """

        manager = mp.Manager()
        test_rewards = manager.list()
        actor_loss = manager.list()
        critic_loss = manager.list()


        processes = []
        frame_counter = mp.Value('Q', 0)
        for i in range(self.n_workers):
            p = mp.Process(target=self._worker, args=(i, test_rewards, actor_loss, critic_loss, 
                                                      frame_counter, eval_window, n_evaluations, early_stopping,
                                                      reward_threshold))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
        
        return test_rewards

    def _worker(self,
                worker_id : int,
                test_rewards : list,
                actor_losses : list,
                critic_losses : list,
                frame_counter : mp.Value,
                eval_window : int,
                n_evaluations : int,
                early_stopping : bool,
                reward_threshold : float):
        
        """
        Process performed per different worker

        """

        env = copy.deepcopy(self.env)
        env.reset()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.early_stop.value = False
        
        # Loop for epochs
        while frame_counter.value < self.max_frames and not self.early_stop.value:
            
            optimizer.zero_grad()
            values, logprobs, rewards = [], [], []
            done = False
            
            state_ = env.reset()
            state = torch.from_numpy(state_).float().unsqueeze(0)
            step_counter = 0

            bootstraping_value = torch.Tensor([0])
            # Loop for steps in episode
            while step_counter < self.num_steps and not done:
                
                policy, value = self.model(state)
                values.append(value)
                logits = policy.view(-1)
                action_dist = torch.distributions.Categorical(logits=logits)
                action = action_dist.sample()
                logprob_ = policy.view(-1)[action]
                logprobs.append(logprob_)
                state_, reward, done, _ = env.step(action.detach().numpy())
                state = torch.from_numpy(state_).float().unsqueeze(0)
                if done:
                    env.reset()
                if step_counter == self.num_steps and not done:
                    # Get value of next state
                    _, value = self.model(state)
                    bootstraping_value = value.detach()
                rewards.append(reward)
                
                step_counter +=1
                frame_counter.value += 1
                
                test_lock.acquire()
                if frame_counter.value % eval_window == 0:
                    counter = frame_counter.value
                    test_reward = np.mean([test_env(self.env, self.model, vis=False) for _ in range(n_evaluations)])
                    test_rewards.append(test_reward)
                    print(f'Frame : {counter} - Test reward : {test_reward}')
                    if test_reward > reward_threshold and early_stopping: self.early_stop.value = True
                test_lock.release()


            # Do not update is criterions for early stop are met
            test_lock.acquire()
            if not (early_stopping and self.early_stop.value):
                actor_loss, critic_loss = self._update_params(optimizer, values, rewards, logprobs, bootstraping_value)
                # Only worker 0 logs losses
                if worker_id == 0:
                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
            test_lock.release()
   
    def _update_params(self,
                       optimizer : torch.optim,
                       values : list,
                       rewards : list,
                       logprobs : list,
                       boostraping_value : torch.Tensor
                       ) -> tuple :
        
        """
        Implemenetaion of model's parameter update step

        """

        rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
        values = torch.stack(values).flip(dims=(0,)).view(-1)

        returns = []

        ret_ = boostraping_value
        for i in range(rewards.shape[0]):
            ret_ = rewards[i] + self.gamma * ret_
            returns.append(ret_)
        returns = torch.stack(returns).view(-1)
        returns = F.normalize(returns, dim=0)
        loss, actor_loss, critic_loss = self._criterion(logprobs, values, returns)
        loss.backward()
        optimizer.step()
        return actor_loss, critic_loss

    def _criterion(self, 
                   logprobs : torch.Tensor,
                   values : torch.Tensor,
                   returns : torch.Tensor) -> tuple :
        
        """
        Loss function for the A3C algorithm

        """

        actor_loss = -1*logprobs*(returns-values.detach())
        critic_loss = torch.pow(values-returns, 2)
        loss =  self.actor_weight*actor_loss.sum() + self.critic_weight*critic_loss.sum()
        
        return loss, actor_loss.sum(), critic_loss.sum()


