import torch, joblib
from utilities import collect_random_states, time_inference, log_runs
from model import Actor

if __name__ == "__main__":
    

    states = joblib.load('results/test_set_3.joblib')
    states_fp16 = collect_random_states(3500, dtype=torch.float16)

    # Benchmark cpu inference
    model = Actor('cpu')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    for batch_size in range(10,11):
        print(f'Batch size {batch_size}')
        dts, avg_time = time_inference(states=states, model=model, batch_size=batch_size)
        log_runs(durations_list=dts, dest_file=f'results/batch_pytorch_cpu_{batch_size}.json')
    
    # Benchmark cuda inference
    model = Actor('cuda')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    for batch_size in range(2,21):
        print(f'Batch size {batch_size}')
        dts, avg_time = time_inference(states=states, model=model, batch_size=batch_size)
        log_runs(durations_list=dts, dest_file=f'results/batch_pytorch_gpu_{batch_size}.json')

    # Benchmark half precission model
    model = Actor('cuda')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    model = model.half()
    for batch_size in range(2,21):
        print(f'Batch size {batch_size}')
        dts, avg_time = time_inference(states=states_fp16, model=model, batch_size=batch_size)
        log_runs(durations_list=dts, dest_file=f'results/batch_pytorch_gpu_fp16_{batch_size}.json')

