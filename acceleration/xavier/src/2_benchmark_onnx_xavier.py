import torch, os, joblib
from utilities import collect_random_states, time_inference, log_runs
from model import ONNXActor

if __name__ == "__main__":
    

    #states = collect_random_states(n_states=3500)
    states = joblib.load('results/test_set_3.joblib')
    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CPUExecutionProvider'])
    dts = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_onnx_cpu.json')

    # Benchmark cuda inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CUDAExecutionProvider'])
    dts = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_onnx_gpu.json')

    # Benchmark tensorRT inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['TensorrtExecutionProvider'])
    dts = time_inference(states=states, model=model)
    log_runs(durations_list=dts, dest_file='results/xavier_onnx_tensorrt.json')


