
import gym, torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from typing import Union
from diploma_framework.evaluation import test_env


class PPO():

    """
    Implements PPO algorithm

    """

    def __init__(self, 
                 environment : Union[str, object],
                 model : nn.Module,
                 lr : float = 1e-03,
                 batch_size : int = 32,
                 epochs : int = 4,
                 max_frames  : int = 150_000,
                 num_steps : int = 100,
                 clip_param : float = 0.2,
                 gamma : float = 0.99,
                 lamb : float = 1
                ):

        if isinstance(environment, str):
            self.env = gym.make(environment)
        else:
            self.env = environment

        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_frames = max_frames
        self.num_steps = num_steps
        self.clip_param = clip_param
        self.gamma = gamma
        self.lamb = lamb

    def run(self, 
            eval_window : int = 1000,
            n_evaluations : int = 10,
            early_stopping : bool = True) -> list :

        """
        Run the PPO algorithm with hyperparameters specified in arguments.
        Returns list of test rewards throughout the agent's training loop.

        eval_window : number of frames between each evaluation

        """

        test_rewards = []

        frame_idx = 0
        early_stop = False

        while frame_idx < self.max_frames and not early_stop:

            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0
            
            state = self.env.reset()
            for _ in range(self.num_steps):

                state = torch.FloatTensor(state).unsqueeze(0)
                action_log_probs, value = self.model(state)

                dist = torch.distributions.Categorical(logits=action_log_probs)
                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action.item())
                entropy += dist.entropy().mean()

                log_probs.append(action_log_probs)
                values.append(value)
                rewards.append(reward)
                masks.append(1-done)

                states.append(state)
                actions.append(action)
                
                state = next_state
                frame_idx += 1
                if frame_idx % eval_window == 0:
                    test_reward = np.mean([test_env(self.env, self.model, vis=False) for _ in range(n_evaluations)])
                    test_rewards.append(test_reward)
                    print(f'Frame : {frame_idx} - Test reward : {test_reward}')
                    if test_reward > 195.2 and early_stopping: early_stop = True

                if done: break

            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = self.model(next_state)
            
            returns = self._compute_returns(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states, dim=0)
            actions   = torch.LongTensor(actions)
            advantage = returns -  values

            self._update_params(states, actions, log_probs, returns, advantage)

        return test_rewards

    def _compute_returns(self, 
                    next_value : float,
                    rewards : list,
                    masks : list,
                    values : list) -> list :

        """
        Calculates return at each time step. Uses delta presented in PPO paper.

        """

        values = values + [next_value]
        gae = 0
        returns = []

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lamb * masks[i] * gae
            returns.insert(0, gae + values[i])

        return returns

    def _get_batch(self,
              states : np.ndarray,
              actions : np.ndarray,
              log_probs : np.ndarray,
              returns : np.ndarray,
              advantage : np.ndarray) -> tuple:

        """
        Responsible for sampling a random batch out of the total saved data.
        Returns sampled states, actions, log_probs, returns and advantages
        """

        total_experiences = states.size(0)
        for _ in range(total_experiences // self.batch_size):
            selections = np.random.randint(0, total_experiences, self.batch_size)
            yield states[selections,:], actions[selections], log_probs[selections,:], returns[selections,:], advantage[selections, :]

    def _update_params(self, 
                  states : np.ndarray, 
                  actions : np.ndarray,
                  log_probs : np.ndarray,
                  returns : np.ndarray,
                  advantages : np.ndarray) -> None:
    
        """
        Performs the basic parameter update of PPO algorithm
        
        """

        for _ in range(self.epochs):
            for state_batch, action_batch, old_log_probs_batch, return_batch, advantage_batch in self._get_batch(states, actions, log_probs, returns, advantages):

                new_log_probs_batch, value_batch = self.model(state_batch)
                dist = torch.distributions.Categorical(logits=new_log_probs_batch)
                entropy = dist.entropy().mean()
                new_log_probs_batch = new_log_probs_batch.gather(dim=-1, index=action_batch.unsqueeze(1))
                old_log_probs_batch = old_log_probs_batch.gather(dim=-1, index=action_batch.unsqueeze(1))

                ratio = (new_log_probs_batch - old_log_probs_batch).exp()
                surr1 = ratio*advantage_batch
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage_batch

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_batch - value_batch).pow(2).mean()

                loss = 0.5*critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()