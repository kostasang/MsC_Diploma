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

from ctypes import *
from typing import List
import numpy as np
import vart
import os
import pathlib
import xir
import time
import sys
import json
import argparse
import vitis_ai_library

_divider = '-------------------------------'


class Timer():
    """
    Class that implements a timer
    """

    def __init__(self) :
        """Initilize times"""
        self.t0 = None
        self.total_time = 0
        self.n_laps = 0
    
    def start(self):
        """"Start timer"""
        self.t0 = time.perf_counter()
    
    def stop(self):
        """Stops timer"""
        dt = (time.perf_counter() - self.t0)
        self.total_time += dt
        self.n_laps += 1
        return dt

    def reset(self):
        """Resets timer"""
        self.t0 = None
        self.total_time = 0
        self.n_laps = 0
    
    def get_average_time(self):
        """Returns average lap time"""
        return self.total_time / self.n_laps

    def get_laps(self):
        """Returns number of laps"""
        return self.n_laps

def log_results(durations_list, dest_file):
    """Log durations in json"""
    with open(dest_file, 'w') as f:
        json.dump(durations_list, f)

class FPGAActor():

    def __init__(self, dpu_runner):

        self.dpu_runner = dpu_runner
        self.numpy_actor_head = []

        input_fixpos = self.dpu_runner.get_input_tensors()[0].get_attr("fix_point")
        print(input_fixpos)
        self.input_scale = 2**input_fixpos

        output_fixpos = self.dpu_runner.get_output_tensors()[0].get_attr("fix_point")
        self.output_scale = 1 / (2**output_fixpos)

        input_tensors = self.dpu_runner.get_input_tensors()
        output_tensors = self.dpu_runner.get_output_tensors()
        self.input_ndim = tuple(input_tensors[0].dims)
        self.output_ndim = tuple(output_tensors[0].dims)
        self.load_numpy_head()

        print('Input tensor name: ', input_tensors[0].name)
        print('Input tensor dim: ', input_tensors[0].dims)
        print('Input tensor dtype: ', input_tensors[0].dtype)
        print('Output tensor name: ', output_tensors[0].name)
        print('Output tensor dim: ', output_tensors[0].dims)
        print('Output tensor dtype: ', output_tensors[0].dtype)

    def preprocess_fn(self, state):
        """Preprocessing of state"""
        state = state[:,20:,:,:] * (1/255) * (self.input_scale)
        return state.astype(np.int8)

    def load_numpy_head(self):
        # load linear numpy head
        for layer_idx in range(0,10,2):
            weights = np.load(f'model_linear_layers/weights_{layer_idx}.npy')
            bias = np.expand_dims(np.load(f'model_linear_layers/bias_{layer_idx}.npy'),1)
            self.numpy_actor_head.append((weights, bias)) 

    def forward(self, x):
        # Implements forward pass using numpy linear layer
        
        x = self.preprocess_fn(x)
        visual_repr = np.empty(self.output_ndim, dtype=np.int8, order="C")
        job_id = self.dpu_runner.execute_async(x, visual_repr)
        self.dpu_runner.wait(job_id)
        visual_repr = visual_repr * self.output_scale

        # layer 1
        weights, bias = self.numpy_actor_head[0]
        out = weights @ visual_repr.T + bias
        out = np.where(out > 0, out, out * 0.01)
        # layer 2
        weights, bias = self.numpy_actor_head[1]
        out = weights @ out + bias
        out = np.where(out > 0, out, out * 0.01)
        # layer 3
        weights, bias = self.numpy_actor_head[2]
        out = weights @ out + bias
        out = np.where(out > 0, out, out * 0.01)
        # layer 4
        weights, bias = self.numpy_actor_head[3]
        out = weights @ out + bias
        out = np.where(out > 0, out, out * 0.01)
        # layer 5
        weights, bias = self.numpy_actor_head[4]
        out = (weights @ out + bias).T
        out = np.exp(out) / np.sum(np.exp(out), axis=1)
        #out = np.log(out)
        return out

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def app(states_path, model):

    states = np.load(states_path)
    graph = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(graph)
    dpu_runner = vart.Runner.create_runner(subgraphs[0], "run")
    model = FPGAActor(dpu_runner)
    timer = Timer()
    outputs = np.empty(shape=(states.shape[0], 3))
    durations = []
    for i, state in enumerate(states):
        timer.start()
        out = model.forward(state)
        outputs[i] = out
        dt = timer.stop()
        durations.append(dt)
        if i == 0:
            print(f'Model output shape : {out.shape}')

    print(f'Averege inference time {timer.get_average_time() * 1000}ms on {timer.get_laps()} states.')
    golden_outputs = np.load('golden_out.npy')
    error = np.abs(outputs - golden_outputs).flatten()
    max_error = np.max(error)
    mean_error = np.mean(error)
    print(f'Max absolute error: {max_error} Average absolute error: {mean_error}')
    log_results(durations_list=durations, dest_file='zcu_fpga.json')

# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()  
  ap.add_argument('-d', '--states_path', type=str, default='test_set_small.npy', help='Path to joblibfile of states')  
  ap.add_argument('-m', '--model',     type=str, default='Actor_zcu104.xmodel', help='Path of xmodel. Default is Actor_zcu104.xmodel')
  args = ap.parse_args()  
  
  print ('Command line options:')
  print (' -states_path : ', args.states_path)
  print (' -model   : ', args.model)

  app(args.states_path, args.model)

if __name__ == '__main__':
  main()

