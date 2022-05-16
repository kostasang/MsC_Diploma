import torch, onnxruntime
import numpy as np
import torch.nn.functional as F
from torch import nn

class ONNXActor():
    # Implements actor using ONNX runtime
    
    def __init__(self, onnx_path, providers):
        # Initiliaze model
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.ort_session.disable_fallback()
    
    def forward(self, x):
        # Implements forward pass of model
        output = self.ort_session.run(None, {'input' : x.numpy().astype(np.float32)})[0]
        return torch.Tensor(output)
    
    def infer_action(self, x):
        # Utilizes torch distributions to return an action
        dist_probs = self.forward(x)
        dist = torch.distributions.Categorical(logits=dist_probs)
        return dist.sample().numpy()[0]

class Actor(torch.nn.Module):
    #Implements the inference of only the actor model coming from the acotr critic object

    def __init__(self, device = 'cpu'):
        #Initilizes actor by copying necesseary layers
        super(Actor, self).__init__()
        self.device = device
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
            nn.Linear(128, 3)
        ).to(device=self.device)

    def forward(self, x):
        #Implements forward pass of model
        x = x.permute((0, 3, 1, 2))   # Place channel axis in correct position
        #x = self.transform(x)               # Apply transform
        x = x / 255
        #x = T.functional.crop(x, top=20, left=0, height=40, width=80)
        x = x[:,:,20:,:]
        x = x.to(device=self.device)
        visual_repr = self.conv_core(x).squeeze(-1).squeeze(-1)
        dist = F.log_softmax(self.actor_head(visual_repr), dim=1)
        return dist

    def infer_action(self, x):
        # Utilizes torch distributions to return an action
        dist_probs = self.forward(x)
        dist = torch.distributions.Categorical(logits=dist_probs)
        return dist.sample().cpu().numpy()[0]

