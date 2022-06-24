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

def time_inference(states, model):
    """Calculate average inference time given list of collected states"""
    runs = []
    timer = Timer()
    with torch.no_grad():
        for state in states:
            timer.start()
            _ = model.infer_action(state)
            dt = timer.stop()
            runs.append(dt)
    print(f'Average inference time {timer.get_average_time()*1000}ms calculated on {timer.get_laps()} frames')
    return runs


def log_runs(durations_list, dest_file):
    """Save list of run durations into json"""
    with open(dest_file, 'w') as f:
        json.dump(durations_list, f)
