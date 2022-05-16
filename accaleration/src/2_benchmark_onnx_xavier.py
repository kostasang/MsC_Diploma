import torch, os
from utilities import collect_random_states, time_inference
from model import ONNXActor

if __name__ == "__main__":
    

    states = collect_random_states(n_states=3500)

    # Benchmark cpu inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CPUExecutionProvider'])
    time_inference(states=states, model=model)

    # Benchmark cuda inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CUDAExecutionProvider'])
    time_inference(states=states, model=model)
    
    os.environ['ORT_TENSORRT_FP16_ENABLE'] = '1'
    os.environ['ORT_TENSORRT_MAX_WORKSPACE_SIZE'] = f"{3 * 1024 * 1024 * 1024}"
    # Benchmark tensorRT inference
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['TensorrtExecutionProvider'])
    time_inference(states=states, model=model)



