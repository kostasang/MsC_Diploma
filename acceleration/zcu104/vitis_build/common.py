'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Common functions for simple PyTorch MNIST example
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import torch, joblib
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset
from utilities.models import Actor

class StatesDataset(Dataset):

    def __init__(self, path, golden_model_path='models/actor_state_dict.pt'):

        states = joblib.load(path)
        model = Actor()
        model.load_state_dict(state_dict=torch.load(golden_model_path))
        self.processed_states = []
        for state in states:
            #state = state.permute((0, 3, 1, 2))
            #state = state / 255
            #state = state[:,:,20:,:]
            self.processed_states.append(state)
        self.target = [model.forward(state) for state in self.processed_states]

    def __len__(self):
        return len(self.processed_states)
    
    def __getitem__(self, index):
        return self.processed_states[index].squeeze(0), self.target[index].squeeze(0)


def test(model, device, test_loader):
    '''
    test the model
    '''
    avg_error = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            batch_avg_error = torch.abs(output - target).flatten().mean().item()
            avg_error += (batch_avg_error * output.shape[0]) 
            total_samples += output.shape[0]
    avg_error /= total_samples
    print(f'Average absolute error {avg_error}')
    return avg_error


