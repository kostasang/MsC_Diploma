import time, sys, torch, numpy as np
from collections import deque
#from gym_duckietown.simulator import Simulator

#sys.argv.append('../../learning/src')
#from big_experiment_utils.wrappers import DiscreteWrapper, DtRewardWrapper

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
'''
def create_environment():
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

def collect_states(env, 
                   golden_model_path, 
                   max_steps=1000):
    """
    Given the golden model and the environment, this function 
    collects states and returns them to a list for later use
    """
    golden_model = joblib.load(golden_model_path)
    states = []
    frame = env.reset()
    stacked_frames = deque([torch.zeros(size=frame.shape).unsqueeze(0)]*5,
                            maxlen=5)
    env.render()

    done = False
    n_steps = 0
    while not done and n_steps < max_steps:    
        frame = torch.FloatTensor(frame).unsqueeze(0)
        stacked_frames.append(frame)
        state = torch.cat(tuple(stacked_frames), dim=-1)
        action = golden_model.infer_action(state)
        states.append(state)
        next_frame, reward, done, _ = env.step(action)
        env.render()
        frame = next_frame
        n_steps += 1
    return states
'''

def collect_random_states(n_states, dtype=torch.float32):
    """Returns list of random states"""
    return [torch.randint(0,256, size=(1,60,80,15), dtype=dtype, ) for _ in range(n_states)]

@torch.inference_mode()
def time_inference(states, model):
    """Calculate average inference time given list of collected states"""
    timer = Timer()
    for state in states:
        timer.start()
        _ = model.infer_action(state)
        timer.stop()
        
    print(f'Average inference time {timer.get_average_time()*1000}ms calculated on {timer.get_laps()} frames')
    return timer.get_average_time(), timer.get_laps()

