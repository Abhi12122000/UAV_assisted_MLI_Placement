import torch
import utils

def policy(state, q_networks, env, epsilon=0., UAV_index=0, num_actions=9, ret_tr=False):
    t_act = utils.cz3a(env, ret_tr)
    if t_act is not None:
        return t_act[UAV_index]
    
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    else:
        flattened_state = torch.flatten(state, end_dim=-1)
        # print(flattened_state.shape)
        av = q_networks[UAV_index](flattened_state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)