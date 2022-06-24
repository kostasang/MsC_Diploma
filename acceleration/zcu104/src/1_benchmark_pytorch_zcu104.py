import torch, joblib
from utilities import collect_random_states, time_inference, log_runs
from model import Actor

if __name__ == "__main__":
    

    states = joblib.load('results/test_set_small.joblib')
    #states = collect_random_states(n_states=3500)

    # Benchmark cpu inference
    model = Actor('cpu')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    runs = time_inference(states=states, model=model)
    log_runs(durations_list=runs, dest_file='results/zcu_pytorch.json')

