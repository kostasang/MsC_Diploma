import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.nn import functional as F 

class CNNActorCritic(nn.Module):

    def __init__(self, n_output, device):

        super(CNNActorCritic, self).__init__()

        self.device = device
        self.transform = T.Compose([
                    T.Normalize(mean=[0]*15, std=[255]*15),   # Turn input to 0-1 range from 0-255
                    ])

        self.conv_core = nn.Sequential(
            nn.Conv2d(15, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        ).to(self.device)
        
        self.actor_head = nn.Sequential(
            nn.Linear(1536,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128, n_output)
        ).to(device=self.device)

        self.critic_head = nn.Sequential(
            nn.Linear(1536,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        ).to(device=self.device)
        self.apply(self.init_weights)

    def init_weights(self, m):
        return 
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0., std=0.5)
            nn.init.constant_(m.bias, 0.5)

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))  # Place channel axis in correct position
        x = self.transform(x)               # Apply transform
        x = T.functional.crop(x, top=20, left=0, height=40, width=80)
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