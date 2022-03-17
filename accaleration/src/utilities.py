import time, sys
from gym_duckietown.simulator import Simulator

sys.argv.append('../../learning/src')
from big_experiment_utils.wrappers import DiscreteWrapper, DtRewardWrapper

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

def environment_creator():
    """Return environment for experiments"""
    env = Simulator(
    seed=None,  # random seed
    map_name="loop_empty",
    max_steps=3_500,  
    domain_rand=False,
    distortion=False,
    camera_width=80,
    camera_height=60,
    draw_curve=True,
    accept_start_angle_deg=4
    )
    env = DtRewardWrapper(env) 
    env = DiscreteWrapper(env)
    return env