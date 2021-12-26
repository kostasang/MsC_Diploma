import torch, gym, copy, math, random
import numpy as np

from collections import deque
from typing import Union
import torch.nn as nn
import torch.optim as optim

from diploma_framework.evaluation import test_env

class DeepQLearning():

    """
    Implements Deep Q-Learning algorithm

    """
    def __init__(self,
                 environment : Union[object, str],
                 model : nn.Module,
                 sync_freq : int = 1000,
                 lr : float = 1e-03,
                 memory_size : int = 2000,
                 batch_size : int = 128,
                 max_frames : int = 150_000,
                 epsilon_start : float = 1.0,
                 epsilon_end : float = 0.0,
                 epsilon_decay : int = 250,
                 gamma = 0.9) -> None :
        
        if isinstance(environment, str):
            self.env = gym.make(environment)
        else:
            self.env = environment
        
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.sync_freq = sync_freq
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=memory_size)
        

    def run(self, 
            eval_window : int = 1000,
            early_stopping : bool = True) -> list:
        
        """
        Run the DQN algorithm with hyperparameters specified in arguments.
        Returns list of test rewards throughout the agent's training loop.

        """

        test_rewards = []
        frame_idx = 0
        early_stop = False

        while frame_idx < self.max_frames and not early_stop:
            
            state0_ = self.env.reset()
            state0 = torch.from_numpy(state0_).float().unsqueeze(dim=0)
            done =  False

            # In episode :
            while not done and not early_stop:

                epsilon = self.epsilon_end + (self.epsilon_start-self.epsilon_end)*math.exp(-1 *frame_idx / self.epsilon_decay)
                frame_idx += 1
                qval = self.model(state0).data.numpy()

                # Explore or exploit
                if (random.random() < epsilon):
                    action = np.random.randint(0,self.env.action_space.n)
                else:
                    action = np.argmax(qval)
                
                # Make the action
                state1_, reward, done, _ = self.env.step(action)
                state1 = torch.from_numpy(state1_).float().unsqueeze(dim=0)

                # Store experience
                exp = (state0, action, reward, state1, done)
                self.replay_buffer.append(exp)
                state0 = state1

                # Train network after new experience is added
                if len(self.replay_buffer) > self.batch_size:

                    batch = random.sample(self.replay_buffer, self.batch_size)
                    state0_batch, action_batch, reward_batch, state1_batch, done_batch = self._get_batch_data(batch)

                    Q1 = self.model(state0_batch)
                    with torch.inference_mode():
                        Q2 = self.target_model(state1_batch)
                    
                    # Bellman criterion
                    Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2, dim=1)[0])
                    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = self.criterion(X, Y.detach())

                    # Perform update
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update target network every sync_freq steps
                    if (frame_idx % self.sync_freq == 0):
                        self.target_model.load_state_dict(self.model.state_dict())

                if frame_idx % eval_window == 0:
                    test_reward = np.mean([test_env(self.env, self.model, vis=False) for _ in range(10)])
                    test_rewards.append(test_reward)
                    print(f'Frame : {frame_idx} - Test reward : {test_reward}')
                    if test_reward > 195.2 and early_stopping: early_stop = True

        return test_rewards

    def _get_batch_data(self, batch : tuple) -> tuple:

        """
        Given a batch of tuples of the form :
        (s0, a1, r1, s1, done), returns batches of the five didfferet components 
        of the initial batch

        """
        
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        done_batch = []

        for experience in batch:
            state0, action, reward, state1, done = experience
            state0_batch.append(state0)
            action_batch.append(action)
            reward_batch.append(reward)
            state1_batch.append(state1)
            done_batch.append(done)

        state0_batch = torch.cat(state0_batch)
        action_batch = torch.Tensor(action_batch)
        reward_batch = torch.Tensor(reward_batch)
        state1_batch = torch.cat(state1_batch)
        done_batch = torch.Tensor(done_batch)

        return state0_batch, action_batch, reward_batch, state1_batch, done_batch