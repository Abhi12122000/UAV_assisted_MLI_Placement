import torch
from torch import nn as nn
import copy

def create_q_networks(state_dims, num_actions, env):
    q_networks = [nn.Sequential(
        nn.Linear(state_dims, 64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64, num_actions)   
    ) for i in range(env.UAV_count)]
    
    target_q_networks = [copy.deepcopy(q_networks[i]).eval() for i in range(env.UAV_count)]
    
    return q_networks, target_q_networks