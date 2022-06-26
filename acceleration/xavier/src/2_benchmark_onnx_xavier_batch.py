import torch, os, joblib
from utilities import collect_random_states, time_inference, log_runs
from model import ONNXActor

if __name__ == "__main__":
    

    states = joblib.load('results/test_set_3.joblib')
    
    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CPUExecutionProvider'])
    for batch_size in range(2,21):
        print(f'Batch size {batch_size}')
        dts, _ = time_inference(states=states, model=model, batch_size=batch_size)
        log_runs(durations_list=dts, dest_file=f'results/batch_onnx_cpu_{batch_size}.json')

    # Benchmark cuda inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CUDAExecutionProvider'])
    for batch_size in range(2,21):
        print(f'Batch size {batch_size}')
        dts, _ = time_inference(states=states, model=model, batch_size=batch_size)
        log_runs(durations_list=dts, dest_file=f'results/batch_onnx_gpu_{batch_size}.json')

    # Benchmark tensorRT inference
    #model = ONNXActor(onnx_path='models/actor.onnx', providers=['TensorrtExecutionProvider'])
    #dts = time_inference(states=states, model=model)
    #log_runs(durations_list=dts, dest_file='results/xavier_onnx_tensorrt.json')


