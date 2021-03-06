import gym, torch, logging
import torch.optim as optim
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from typing import Union
from diploma_framework.algorithms._generic import DeepRLAlgorithm

logger = logging.getLogger('deepRL')

class PPO(DeepRLAlgorithm):

    """
    Implements PPO algorithm

    """

    def __init__(self, 
                 environment: Union[str, object],
                 model: nn.Module,
                 lr: float = 1e-03,
                 batch_size: int = 32,
                 epochs: int = 4,
                 max_frames: int = 150_000,
                 num_steps: int = 100,
                 clip_param: float = 0.2,
                 gamma: float = 0.99,
                 lamb: float = 1.0,
                 actor_weight: float = 1.0,
                 critic_weight: float = 0.5,
                 entropy_weight: float = 0.001
                ) -> None:

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
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.entropy_weight = entropy_weight

    def run(self, 
            eval_window: int = 1000,
            n_evaluations: int = 10,
            early_stopping: bool = True,
            reward_threshold: float = 197.5) -> list:
        """
        Run the PPO algorithm with hyperparameters specified in arguments.
        Returns list of test rewards throughout the agent's training loop.
        
        eval_window : number of frames between each evaluation
        """
        logger.info('Initializing training')

        test_rewards = []
        test_frames = []
        frame_idx = 0
        early_stop = False

        with tqdm(total = self.max_frames) as pbar:
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
                    dist, action, value = self.model.infer_step(state)

                    next_state, reward, done, _ = self.env.step(action)
                    entropy += dist.entropy().mean()
                    
                    action_log_probs = dist.log_prob(torch.Tensor([action]))
                    log_probs.append(action_log_probs)
                    values.append(value)
                    rewards.append(reward)
                    masks.append(1-done)

                    states.append(state)
                    actions.append(action)
                    
                    state = next_state
                    frame_idx += 1
                    if frame_idx % eval_window == 0:
                        reward_metric, frame_metric = self.evaluate(n_evaluations)
                        test_rewards.append(reward_metric)
                        test_frames.append(frame_metric)
                        pbar.update(eval_window)
                        pbar.set_description(f'Reward {reward_metric} - Frames {frame_metric}')
                        if reward_metric > reward_threshold and early_stopping: 
                            early_stop = True
                            logger.info('Early stopping criteria met')


                    if done: break

                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                _, _, next_value = self.model.infer_step(next_state)
                
                returns = self._compute_returns(next_value, rewards, masks, values)

                returns   = torch.cat(returns).detach()
                log_probs = torch.cat(log_probs).detach()
                values    = torch.cat(values).detach()
                states    = torch.cat(states, dim=0)
                actions   = torch.LongTensor(actions)
                advantage = returns -  values

                self._update_params(states, actions, log_probs, returns, advantage)

        return test_rewards, test_frames

    def _compute_returns(self, 
                    next_value: float,
                    rewards: list,
                    masks: list,
                    values: list) -> list:
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
              states: np.ndarray,
              actions: np.ndarray,
              log_probs: np.ndarray,
              returns: np.ndarray,
              advantage: np.ndarray) -> tuple:
        """
        Responsible for sampling a random batch out of the total saved data.
        Returns sampled states, actions, log_probs, returns and advantages
        """
        total_experiences = states.size(0)
        for _ in range(total_experiences // self.batch_size):
            selections = np.random.randint(0, total_experiences, self.batch_size)
            yield states[selections,:], actions[selections], log_probs[selections], returns[selections,:], advantage[selections, :]

    def _update_params(self, 
                  states: np.ndarray, 
                  actions: np.ndarray,
                  log_probs: np.ndarray,
                  returns: np.ndarray,
                  advantages: np.ndarray) -> None:
        """
        Performs the basic parameter update of PPO algorithm
        """
        for _ in range(self.epochs):
            for state_batch, action_batch, old_log_probs_batch, return_batch, advantage_batch in self._get_batch(states, actions, log_probs, returns, advantages):

                dist_batch, value_batch = self.model.infer_batch(state_batch)
                entropy = dist_batch.entropy().mean()
                
                new_log_probs_batch = dist_batch.log_prob(action_batch)

                ratio = (new_log_probs_batch - old_log_probs_batch).exp()
                surr1 = ratio*advantage_batch
                surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage_batch

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_batch - value_batch).pow(2).mean()

                loss = self.critic_weight*critic_loss + self.actor_weight*actor_loss - self.entropy_weight * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()