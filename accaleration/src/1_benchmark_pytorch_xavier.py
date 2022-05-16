import torch
from utilities import collect_random_states, time_inference
from model import Actor

if __name__ == "__main__":
    

    states = collect_random_states(n_states=3500)

    # Benchmark cpu inference
    model = Actor('cpu')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    time_inference(states=states, model=model)

    # Benchmark cuda inference
    model = Actor('cuda')
    model.load_state_dict(state_dict=torch.load('models/actor_state_dict.pt'))
    time_inference(states=states, model=model)



