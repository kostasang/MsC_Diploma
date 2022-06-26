import time, sys, torch, json, numpy as np
from collections import deque

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

def collect_random_states(n_states, dtype=torch.float32):
    """Returns list of random states"""
    return [torch.randint(0,256, size=(1,60,80,15), dtype=dtype, ) for _ in range(n_states)]

def create_batches(states, batch_size):
    """Turn list of states to list of batches of states"""
    batches = []
    for i in range(0, len(states)//batch_size):
        batch = states[i*batch_size:(i+1)*batch_size]
        batch = torch.cat(batch, axis=0)
        batches.append(batch)
    return batches
    
@torch.inference_mode()
def time_inference(states, model, batch_size=1):
    """Calculate average inference time given list of collected states"""
    if batch_size > 1:
        states = create_batches(states, batch_size)
    durations = []
    print(f'Input shape: {states[0].shape}')
    timer = Timer()
    for state in states:
        timer.start()
        _ = model.infer_action(state)
        dt = timer.stop()
        durations.append(dt)

    print(f'Average inference time {timer.get_average_time()*1000}ms calculated on {timer.get_laps()} frames')
    return durations, timer.get_average_time()

def log_runs(durations_list, dest_file):
    """Save list of run durations into json"""
    with open(dest_file, 'w') as f:
        json.dump(durations_list, f)
