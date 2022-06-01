import torch, os, joblib
from utilities import collect_random_states, time_inference
from model import ONNXActor

if __name__ == "__main__":
    

    #states = collect_random_states(n_states=3500)
    states = joblib.load('results/test_set_3.joblib')
    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CPUExecutionProvider'])
    time_inference(states=states, model=model)

    # Benchmark cuda inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CUDAExecutionProvider'])
    time_inference(states=states, model=model)
    
    # Benchmark tensorRT inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['TensorrtExecutionProvider'])
    time_inference(states=states, model=model)



