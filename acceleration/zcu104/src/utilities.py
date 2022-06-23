import time, sys, torch, numpy as np
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
        self.total_time += (time.perf_counter() - self.t0)
        self.n_laps += 1

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
    timer = Timer()
    with torch.no_grad():
        for state in states:
            timer.start()
            _ = model.infer_action(state)
            timer.stop()
    print(f'Average inference time {timer.get_average_time()*1000}ms calculated on {timer.get_laps()} frames')
    return timer.get_average_time(), timer.get_laps()

