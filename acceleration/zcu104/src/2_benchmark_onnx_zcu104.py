import torch, os, joblib
from utilities import collect_random_states, time_inference, log_runs
from model import ONNXActor

if __name__ == "__main__":
    

    #states = collect_random_states(n_states=3500)
    states = joblib.load('results/test_set_small.joblib')
    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CPUExecutionProvider'])
    runs = time_inference(states=states, model=model)
    log_runs(durations_list=runs, dest_file='results/zcu_onnx.json')
