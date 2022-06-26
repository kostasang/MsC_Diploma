import torch, joblib
from utilities import collect_random_states, time_inference, log_runs
from model import Actor

if __name__ == "__main__":
    

    states = joblib.load('results/test_set_3.joblib')
    #states = collect_random_states(n_states=3500)

    # Benchmark cpu inference
    model = Actor('cpu')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    dts, _ = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_pytorch_cpu.json')

    # Benchmark cuda inference
    model = Actor('cuda')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    dts, _ = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_pytorch_gpu.json')

    # Benchmark half precission model

    states = collect_random_states(3500, dtype=torch.float16)

    # Benchmark cuda inference
    model = Actor('cuda')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    model = model.half()
    dts, _ = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_pytorch_gpu_fp16.json')
