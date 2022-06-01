import torch, joblib, numpy as np
from utilities import collect_random_states
from model import Actor, ONNXActor

@torch.inference_mode()
def check_fp16(target_model, test_model, states, decimal):
    """Check if test model yields the same results as target """
    golden_outputs = [target_model.debug_forward(state).cpu().numpy() for state in states]
    test_outputs = [test_model.debug_forward(state.to(torch.float16)).cpu().numpy() for state in states]
    
    print(calculate_stats(golden_outputs, test_outputs))
    
    np.testing.assert_array_almost_equal(golden_outputs, test_outputs, decimal=decimal)

@torch.inference_mode()
def check_onnx(target_model, test_model, states, decimal):
    """Compare golden model with onnx int8"""
    golden_outputs = [target_model.debug_forward(state).cpu().numpy() for state in states]
    test_outputs = [test_model.debug_forward(state).cpu().numpy() for state in states]
    
    print(calculate_stats(golden_outputs, test_outputs))

    np.testing.assert_array_almost_equal(golden_outputs, test_outputs, decimal=decimal)


def calculate_stats(golden, test):
    golden = np.array(golden)
    test = np.array(test)
    error = np.abs(golden-test).flatten()
    return {'avg_abs_error' : np.mean(error), 'max_abs_error' : np.max(error)}

if __name__ == "__main__":
    

    #states = collect_random_states(n_states=100)
    states = joblib.load('results/test_set_1.joblib')
    golden_model = Actor('cpu')
    golden_model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))

    #model = Actor('cuda')
    #model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    #model = model.half()
    #check_fp16(golden_model, model, states, decimal=5)
    
    model = ONNXActor(onnx_path='models/actor.onnx', providers=['CUDAExecutionProvider'])
    check_onnx(golden_model, model, states, decimal=7)

