import torch, numpy as np
from utilities import collect_random_states
from model import Actor

@torch.inference_mode()
def check_results(target_model, test_model, states, decimal=1):
    """Check if test model yields the same results as target """
    golden_outputs = [target_model.forward(state).cpu().numpy() for state in states]
    test_outputs = [test_model.forward(state.to(torch.float16)).cpu().numpy() for state in states]
    np.testing.assert_array_almost_equal(golden_outputs, test_outputs, decimal=decimal)


if __name__ == "__main__":
    

    states = collect_random_states(n_states=100)

    golden_model = Actor('cpu')
    
    model = Actor('cuda')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    model = model.half()
    
    check_results(golden_model, model, states, decimal=4)


