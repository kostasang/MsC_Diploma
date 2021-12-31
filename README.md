# Reinforcement Learning Framework

As part of my MsC's thesis, a framework that implements Deep Reinforcement Learning algorithms (DQN, REINFORCE, A3C, PPO) has been developed. Its goal is to enable fast experimentation with different neural network models and different environments without the need of re-implementing the code for each deep reinforcement learning algorithm all over again. The framework is build on Pytorch and any environment following the OpenAI gym API paradigm can be used.


## Installation 

Run the following in order to use the framework:

```bash
git clone https://github.com/kostasang/MsC_Diploma.git
cd MsC_Diploma
conda create -y --name deepRL python==3.8
conda activate deepRL
pip3 install -r requirements.txt
```

## Usage 

In the following code section, an example for running PPO algorithm on CartPole environment using a self-defined Actor-Critic model is illustrated. The model must implement an `infer_action()` method outputs the model's final action decision implementing the corresponging sampling with the proper probabilities calculated by the model itself. Each algorithm expects the input model to implement certain methods with similar functionalities. More details can be found in the documentation section bellow.

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

    def infer_action(self, x):
        dist, _ = self.forward(x)
        dist = torch.distributions.Categorical(logits=dist)
        return dist.sample().cpu().numpy()[0]

env = gym.make('CartPole-v0')
model = ActorCritic()

ppo= PPO(env, model, lr=1e-03, num_steps=150, max_frames=500_000, batch_size=4)
rewards = ppo.run(early_stopping=False)
```

## Documentation

Complete documentation for each algorithm can be found bellow:

* [DQN](#dqn)  
* [Reinforce](#reinforce) 
* [A3C](#a3c)
* [PPO](#ppo)

In order to create a model that is usable from the the framework, the model object must implement two methods, `infer_action()` and `infer_all()`. Each method should return the following :

* `infer_action()` : Return the plain action that the model performs given the state. The returned action must be compatible with `env.step()` method.

* `infer_all()` : Return every usefull information from the model. For example, a Q-network's `infer_all()` method should return `q_value` and `action` while an Actor-Critic model's `infer_all()` method should return `action_probs`, `action` and `value`. In order for the model to return the action, sampling from the probabilities calculated by the model is performer inside `infer_all()` method. This way, the framework can still operate on continious action spaces without the need to change the algorithms.

<a name="dqn"></a>
### `diploma_framework.algorithms.DQN` :

<details>
<summary>Parameters</summary>

* `environment` : str or object 

    Either the name of an gym environment or an environment object exposing the same API as openAI gym.

* `model` : pytorch model

    Pytorch network that must implement `infer_all()` and `infer_action()` methods. `infer_all()` must return tuple of `q_value`, `action` and `infer_action()` must return `action`.

* `sync_freq` : int, default = 1000

    Number of episodes after which Q-network's parameters will be copied to target Q-network.

* `lr` : float, default = 1e-03 

    Learning rate used by the optimizer when performing gradient descent.

* `memory_size` : int, default = 2000

    Size of experience replay buffer

* `batch_size` : int, default = 128

    Batch size used in the optimization process.

* `max_frames` : int, default = 150000

    Maximum number of frames seen during the agent's training.

* `epsilon_start` : float, default = 1.0

    Initial probability of exploration.

* `epsilon_end` : float, default = 0.0

    Terminal probability of exploration. Probability will not be reduced bellow this threshold.

* `epsilon_decay` : int, default = 250

    Exploration probability decay parameter. The higher the value, the faster the decay.

* `gamma` : float , default = 0.9

    Discount factor for future rewards.
</details>

<details>
<summary>Methods</summary>

* `run()`: 

    * Parameters :
    
        - `eval_window` : int, default = 1000

        Number of frames between each evaluation.

        - `n_evaluations` : int, default = 10

            Number of evaluation runs perform at each evaluation step. 

        - `early_stopping` : bool, default = True

            Whether the training is terminated upon the reward_threshold is achieved.

        - `reward_threshold` : float, default = 197.5

            The reward threshold above which the training is terminated.
    
    * Returns :

        - `test_rewards` : list 

            List of calculated average rewards at each evaluation step.
</details>

<a name="reinforce"></a>

### `diploma_framework.algorithms.Reinforce` :

<details>
<summary>Parameters</summary>

* `environment` : str or object 

    Either the name of an gym environment or an environment object exposing the same API as openAI gym.

* `model` : pytorch model

    Pytorch network that must implement `infer_all()` and `infer_action()` methods. `infer_all()` must return tuple of `action_probabilities`, `action` and `infer_action()` must return `action`.

* `lr` : float, default = 1e-03

    Learning rate used by the optimizer when performing gradient descent.

* `max_frames` : int, default = 150000

    Maximum number of frames seen during the agent's training.

* `num_steps` : int, default = 150

    Maximum number of steps in an episode. If this number of steps is reach, episode ends and optimization process is performed.

* `gamma` : float , default = 0.9

    Discount factor for future rewards.
</details>

<details>
<summary>Methods</summary>

* `run()`: 

    * Parameters :
    
        - `eval_window` : int, default = 1000

        Number of frames between each evaluation.

        - `n_evaluations` : int, default = 10

            Number of evaluation runs perform at each evaluation step. 

        - `early_stopping` : bool, default = True

            Whether the training is terminated upon the reward_threshold is achieved.

        - `reward_threshold` : float, default = 197.5

            The reward threshold above which the training is terminated.
    
    * Returns :

        - `test_rewards` : list 

            List of calculated average rewards at each evaluation step.
</details>

<a name="a3c"></a>

### `diploma_framework.algorithms.A3C` :
<details>
<summary>Parameters</summary>

* `environment` : str or object 

    Either the name of an gym environment or an environment object exposing the same API as openAI gym.

* `model` : pytorch model

    Pytorch network that must implement `infer_all()` and `infer_action()` methods. `infer_all()` must return tuple of `action_probabilities`, `action`, `value` and `infer_action()` must return `action`.

* `n_workers` : int, default = 8

    Number of parallel working agents.
    
* `lr` : float, default = 1e-03

    Learning rate used by the optimizer when performing gradient descent.

* `max_frames` : int, default = 150000

    Maximum number of frames seen during the agent's training.

* `num_steps` : int, default = 150

    Maximum number of steps in an episode. If this number of steps is reach, episode ends and optimization process is performed.

* `actor_weight` : float, default = 1.0

    Weight applied to actor loss.

* `critic_weight` : float, default = 0.1

    Weight applied to critic loss.

* `gamma` : float , default = 0.9

    Discount factor for future rewards.

</details>
<details>
<summary>Methods</summary>

* `run()`: 

    * Parameters :
    
        - `eval_window` : int, default = 1000

            Number of frames between each evaluation.

        - `n_evaluations` : int, default = 10

            Number of evaluation runs perform at each evaluation step. 

        - `early_stopping` : bool, default = True

            Whether the training is terminated upon the reward_threshold is achieved.

        - `reward_threshold` : float, default = 197.5

            The reward threshold above which the training is terminated.
    
    * Returns :

        - `test_rewards` : list 

            List of calculated average rewards at each evaluation step.

        - `actor_loss` : list

            List of actor losses calculated by one worker during the agent's training.

        - `critic_loss` : list

            List of critic losses calculated by one worker during the agent's training.


</details>

<a name="ppo"></a>

### `diploma_framework.algorithms.PPO` :

<details>
<summary>Parameters</summary>

* `environment` : str or object 

    Either the name of an gym environment or an environment object exposing the same API as openAI gym.

* `model` : pytorch model

    Pytorch network that must implement `infer_all()` and `infer_action()` methods. `infer_all()` must return tuple of `action_probabilities`, `action`, `value` and `infer_action()` must return `action`.
    
* `lr` : float, default = 1e-03

    Learning rate used by the optimizer when performing gradient descent.

* `batch_size` : int, default = 32

    Batch size used in optimization process.
    
* `epochs` : int, default = 4

    Number of epochs during the optimization process.

* `max_frames` : int, default = 150000

    Maximum number of frames seen during the agent's training.

* `num_steps` : int, default = 150

    Maximum number of steps in an episode. If this number of steps is reach, episode ends and optimization process is performed.

* `clip_param` : float, default = 0.2

    Clip parameter of PPO algorithm.

* `gamma` : float , default = 0.9

    Discount factor for future rewards.

* `lamb` : float, default = 1.0

    Lambda used when discounting rewards.

</details>

<details>
<summary>Methods</summary>

* `run()`: 

    * Parameters :
    
        - `eval_window` : int, default = 1000

            Number of frames between each evaluation.

        - `n_evaluations` : int, default = 10

            Number of evaluation runs perform at each evaluation step. 

        - `early_stopping` : bool, default = True

            Whether the training is terminated upon the reward_threshold is achieved.

        - `reward_threshold` : float, default = 197.5

            The reward threshold above which the training is terminated.
    
    * Returns :

        - `test_rewards` : list 

            List of calculated average rewards at each evaluation step.

</details>