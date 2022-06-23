import torch, joblib, numpy as np
from utilities import collect_random_states
from model import Actor, ONNXActor

def check_onnx(target_model, test_model, states, decimal):
    """Compare golden model with onnx int8"""
    with torch.no_grad():
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
    states = joblib.load('results/test_set_small.joblib')
    golden_model = Actor('cpu')
    golden_model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))

    
    model = ONNXActor(onnx_path='models/actor_8bit_dynamic.onnx', providers=['CPUExecutionProvider'])
    check_onnx(golden_model, model, states, decimal=7)

