import torch, os, joblib
from utilities import collect_random_states, time_inference, log_runs
from model import ONNXActor

if __name__ == "__main__":
    

    states = joblib.load('results/test_set_3.joblib')

    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['CPUExecutionProvider'])
    dts, _ = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_onnx8bit_cpu.json')

    # Benchmark cuda inference
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['CUDAExecutionProvider'])
    dts, _ = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_onnx8bit_gpu.json')
    
    # Benchmark tensorRT inference
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['TensorrtExecutionProvider'])
    dts, _ = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_onnx8bit_tensorrt.json')
