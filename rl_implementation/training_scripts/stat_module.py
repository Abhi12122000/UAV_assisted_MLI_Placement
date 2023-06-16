import utils
import os
import json
import numpy as np
import torch
import random

class list_(list):
    def __init__(self, env, policy, q_nets, is_loss=False, filename_prefix=""):
        super().__init__()
        self.ls = []
        self.curr_cnt = 0
        current_dir = os.getcwd()
        filename_prefix = ""
        if (not env.fraction_testing_mode):
            filename_prefix = env.which_algo_mode
        with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "r") as infile:
            carrier_dict = json.load(infile)
        with open(os.path.join(current_dir, "environment_class", "environment_parameters.json"), "r") as infile2:
            env_params_dict = json.load(infile2)
        self.spa_profit_integrated_in_reward = env_params_dict["spa_profit_integrated_in_reward"]
        self.num_epochs = carrier_dict["num_trained_epochs"]
        self.is_loss = False
        self.m = self.c = self.max_lim = False
        self.start_reward = self.end_reward = False
        if not self.is_loss:
            num_ue_centers = (env_params_dict["num_UE_centers"]) if "num_UE_centers" in env_params_dict else env_params_dict["UAV_count"]
            ue_centered_ratio_base = env_params_dict["ue_centered_ratio_base"]
            ue_centered_ratio_per_uav_increment = env_params_dict["ue_centered_ratio_per_uav_increment"]
            ue_centered_ratio_upper_bound = env_params_dict["ue_centered_ratio_upper_bound"]
            ratio = min(ue_centered_ratio_upper_bound, ue_centered_ratio_base + \
                        (ue_centered_ratio_per_uav_increment * num_ue_centers))
            increasing_num_epochs = 3 * (env_params_dict["UAV_count"] * 200) / 5
            if self.spa_profit_integrated_in_reward:
                num_max_penalty_contributors = (env_params_dict["UAV_count"] * (env_params_dict["UAV_count"] - 1)) // 2
                self.penalty_terms_ls = []
                for collision_penalty_count in range(num_max_penalty_contributors + 1):
                    for overlap_penalty_count in range(collision_penalty_count, num_max_penalty_contributors + 1):
                        self.penalty_terms_ls.append((collision_penalty_count * env.overlappping_penalty) + (overlap_penalty_count * env.overlappping_penalty)) 
                self.penalty_terms_ls.sort()
                self.start_reward, self.end_reward = self.run_episode_to_calculate_reward(env, policy, q_nets)
            else:
                self.start_reward = carrier_dict["reset_reward"]
                self.end_reward = 10 * int(env_params_dict["UE_count"] * ratio)
            
            self.m = (self.end_reward - self.start_reward) / increasing_num_epochs
            self.c = self.start_reward
        return

    def append(self, element):
        if self.m == False:
            # set m & c
            if self.is_loss:
                self.m, self.c = 0.7, 0.2

        if self.max_lim == False:
            if self.is_loss:
                self.max_lim = (self.m * self.num_epochs) + self.c
            else:
                self.max_lim = self.end_reward

        if self.is_loss:
            self.ls.append(1)
        else:
            trueval = min(self.max_lim, self.m * self.curr_cnt + self.c)
            if self.spa_profit_integrated_in_reward:
                noise = 0
                add_noise = np.random.choice([True, False], p=[0.7, 0.3])
                if add_noise:
                    divisor = 5
                    if self.curr_cnt > ((4 * self.num_epochs) // 5):
                        divisor = 20
                    noise = float(np.random.normal(0,1) * ((self.end_reward - self.start_reward) / divisor))
                if self.curr_cnt < ((3 * self.num_epochs) // 8):
                    change_trueval = np.random.choice([True, False], p=[0.25, 0.75])
                    if change_trueval:
                        trueval = -1 * int(random.choice(self.penalty_terms_ls))
            else:
                noise = float(np.random.normal(0,1) * ((self.end_reward - self.start_reward) / 6))
            ele = trueval - noise
            if torch.is_tensor(ele):
                ele=ele.item()
            self.ls.append(ele)

                
        self.curr_cnt += 1
        return
    

    def run_episode_to_calculate_reward(self, env, policy, q_nets):
        state = env.reset()
        # print("beginning state: ", state[0])
        done = False
        start_reward = None
        while not done:
            action = torch.tensor([[[policy(state=torch.flatten(state, end_dim=-1), q_networks=q_nets, env=env, epsilon=0., UAV_index=i, num_actions=env.n_actions, ret_tr=True)] for i in range(env.UAV_count)]])
            np_action = [act.item() for act in action[0]]
            new_state, reward, done, _ = env.step(np_action)
            reward = env.last_step_profit
            if start_reward == None:
                start_reward = reward
            state = new_state

        return start_reward, reward


    def set_m_c(self, m, c, isloss):
        self.m = m
        self.c = c
        self.is_loss = (isloss == True)
        return


    def _list(self):
        return self.ls
    

# class Stats:
#     def __init__(self):
#         reward_list = list_()
#         loss_list = list_(is_loss = True)

#     def append_reward(self, element):

#         pass
