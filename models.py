import torch
import numpy as np
import torch.nn as nn, torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F 
from PIL import Image

class CNNActorCritic(nn.Module):

    def __init__(self, n_output, device):

        super(CNNActorCritic, self).__init__()

        self.device = device
        self.transform = T.Compose([
                    T.Normalize(mean=[0]*15, std=[255]*15),   # Turn input to 0-1 range from 0-255
                    ])

        self.conv_core = nn.Sequential(
            nn.Conv2d(15, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.3)
        ).to(self.device)
        
        self.actor_head = nn.Sequential(
            nn.Linear(2560,128),
            nn.ReLU(),
            nn.Linear(128, n_output)
        ).to(device=self.device)

        self.critic_head = nn.Sequential(
            nn.Linear(2560, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device=self.device)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))  # Place channel axis in correct position
        x = self.transform(x)               # Apply transform
        x = x.to(device=self.device)
        visual_repr = self.conv_core(x).squeeze(-1).squeeze(-1)   # Calculate ResNet output
        return F.log_softmax(self.actor_head(visual_repr), dim=1).to('cpu'), self.critic_head(visual_repr.detach()).to('cpu') # Calculate policy probs and value

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

class SimpleDQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super(SimpleDQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,n_actions)
        )
    
    def infer_step(self, x):
        qval = self.model(x)
        action = qval.detach().argmax().item()
        return qval, action

    def infer_batch(self, x):
        return self.model(x)

    def infer_action(self, x):
        return self.model(x).detach().argmax().item()

class SimpleReinforce(nn.Module):

    def __init__(self):
        super(SimpleReinforce, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,2),
            nn.Softmax(dim=1)
        )
    
    def infer_step(self, x):
        act_prob = self.model.forward(x)
        dist = torch.distributions.Categorical(probs=act_prob)
        action = dist.sample().item()
        return dist, action

    def infer_batch(self, x):
        act_prob = self.model.forward(x)
        dist = torch.distributions.Categorical(probs=act_prob)
        return dist
    
    def infer_action(self, x):   
        act_prob = self.model.forward(x)
        dist = torch.distributions.Categorical(probs=act_prob)
        action = dist.sample().item()
        return action
                
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

