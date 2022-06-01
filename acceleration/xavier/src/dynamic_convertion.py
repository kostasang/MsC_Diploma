import onnx
from onnxruntime.quantization import quantize_dynamic


if __name__ == "__main__":
    quantize_dynamic('models/actor.onnx', 'models/actor_8bit_dynamic.onnx')
