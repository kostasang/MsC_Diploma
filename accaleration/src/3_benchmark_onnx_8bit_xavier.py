import torch, os
from utilities import collect_random_states, time_inference
from model import ONNXActor

if __name__ == "__main__":
    

    states = collect_random_states(n_states=3500)

    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['CPUExecutionProvider'])
    time_inference(states=states, model=model)

    # Benchmark cuda inference
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['CUDAExecutionProvider'])
    time_inference(states=states, model=model)
    
    # Benchmark tensorRT inference
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['TensorrtExecutionProvider'])
    time_inference(states=states, model=model)



