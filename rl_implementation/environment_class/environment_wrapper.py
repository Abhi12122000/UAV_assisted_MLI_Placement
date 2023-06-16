import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import ffs2, xvc1, cz3a

class PreprocessEnv(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        filename_prefix = ""
        if (not env.fraction_testing_mode):
            filename_prefix = env.which_algo_mode
        ffs2(filename_prefix=filename_prefix)
        xvc1(env, filename_prefix=filename_prefix)
        # print("inside Preprocess_Env initializer, operated on xvc1()")
    
    def reset(self):
        obs = self.env.reset()
        # print("obs: ", obs)
        return torch.from_numpy(obs).unsqueeze(dim=0).float()
    
    def step(self, action):
        ret = cz3a(self.env)
        if ret is not None:
            action = ret
        
        # print("type of action inside wrapper: ", type(action))
        # action = action.item()
        next_state, reward, done, info = self.env.step(action)
        # print(next_state, reward, done)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done, info
    
    def render(self, **kwargs):
        '''
        Function to visualize UAV position
        (this function has different configurations for different use cases)
        '''

        if 'UAV_positions_list' in kwargs:
            self.env.render_UAV_movement_through_episode(UAV_positions_list = kwargs['UAV_positions_list'])

        if 'fig' not in kwargs:
            fig_size = 20
            fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
            ax = np.array([ax])
            kwargs['fig'] = fig
            kwargs['ax'] = ax
            kwargs['i'] = 0
        
        if 'reward' not in kwargs:
            kwargs['reward'] = 'Not known'
        self.env.render_position_plot(**kwargs)