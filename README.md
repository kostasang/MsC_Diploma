# Master Thesis

## Installation 

Run the following in order to use the framework:

```bash
git clone https://github.com/kostasang/MsC_Diploma.git
cd MsC_Diploma
conda create -y --name deepRL python==3.8
conda activate deepRL
pip3 install -r requirements.txt
```
For recreating the results for duckieTown environment install the corresponding repository following instructions from [here](https://github.com/duckietown/gym-duckietown).

## Learning & Reinforcement Learning Framework

As part of my MsC's thesis, a framework that implements Deep Reinforcement Learning algorithms (DQN, REINFORCE, A3C, PPO) has been developed. Its goal is to enable fast experimentation with different neural network models and different environments without the need of re-implementing the code for each deep reinforcement learning algorithm all over again. The framework is build on Pytorch and any environment following the OpenAI gym API paradigm can be used.

### Usage 

In the following code section, an example for running PPO algorithm on CartPole environment using a self-defined Actor-Critic model is illustrated. The model must implement an `infer_step()`, `infer_batch()` and `infer_action()` methods that output thepolicy distributions, values and action calculated by the model for a single or batch input. Each algorithm expects the input model to implement these three methods.

```python
from diploma_framework.algorithms import PPO
import gym, torch
import torch.nn as nn
from torch.nn import functional as F 


class ActorCritic(nn.Module):
    
    # Define double-headed model, one 
    # head for actor, another one for critic

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50,2)
        self.l3 = nn.Linear(50,25)
        self.critic_lin1 = nn.Linear(25,1)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor =  F.log_softmax(self.actor_lin1(y), dim=1)
        c = F.relu(self.l3(y.detach()))
        critic = self.critic_lin1(c)
        return actor, critic

    def infer_step(self, x):
        action_probs, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=action_probs)
        action = dist.sample().item()
        return dist, action, value

    def infer_batch(self, x):
        action_probs, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=action_probs)
        return dist, value

    def infer_action(self, x):
        dist, _ = self.forward(x)
        dist = torch.distributions.Categorical(logits=dist)
        return dist.sample().cpu().numpy()[0]

env = gym.make('CartPole-v0')
model = ActorCritic()

ppo= PPO(env, model, lr=1e-03, num_steps=150, max_frames=500_000, batch_size=4)
rewards, n_frames, _, _ = ppo.run(early_stopping=False)
```
