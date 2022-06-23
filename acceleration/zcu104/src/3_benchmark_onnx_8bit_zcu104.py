import torch, os, joblib
from utilities import collect_random_states, time_inference
from model import ONNXActor

if __name__ == "__main__":
    

    states = joblib.load('results/test_set_small.joblib')

    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['CPUExecutionProvider'])
    time_inference(states=states, model=model)
