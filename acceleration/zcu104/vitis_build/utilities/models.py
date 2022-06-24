import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(torch.nn.Module):
    #Implements the inference of only the actor model coming from the acotr critic object

    def __init__(self):
        #Initilizes actor by copying necesseary layers
        super(Actor, self).__init__()
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
            nn.Flatten()
        )
        self.actor_head = nn.Sequential(
            nn.Linear(1536,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )

        #self.output_act = nn.LogSoftmax(dim=-1)
        self.output_act = nn.Softmax(dim=-1)

    def forward(self, x):
        #Implements forward pass of model
        #x = torch.permute(x, (0, 3, 1, 2))   # Place channel axis in correct position
        x = x.permute((0, 3, 1, 2))
        #x = self.transform(x)               # Apply transform
        x = x / 255
        #x = T.functional.crop(x, top=20, left=0, height=40, width=80)
        x = x[:,:,20:,:]
        out = self.conv_core(x)
        #out = self.actor_head(out)
        #out = self.output_act(out)
        return out

    def infer_action(self, x):
        # Utilizes torch distributions to return an action
        dist_probs = self.forward(x)
        dist = torch.distributions.Categorical(logits=dist_probs)
        return dist.sample().cpu().numpy()[0]
    