import random
import copy
import gym
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from torch import nn as nn
from tqdm import tqdm
import numpy as np
from gym import Env, spaces
import os
import json
import matplotlib.patches as mpatches
import utils
from spa_module.at_system_init import fill_component_pos, init_user_data_filler, \
    mli_selection, select_mlis_at_ue_placement, server_info_filler, \
    system_comp_counts_filler, copy_env_from_files
from spa_module.post_action import filter_uncovered_ues, update_power_energy_of_local_ues
from spa_module.spa_algo_scripts import spa_algorithm_caller
from spa_module.dts_algo_scripts import dts_algorithm_caller, temp_improved_dts
# from utils import test_agent, plot_stats, seed_everything


class Single_Agent_UAV(Env):
    def __init__(self, initial_UAV_state = None, fraction_testing_mode=False, \
        copy_env_from_file=True, which_algo_mode='spa', fs_bandwidth_plot_divisor=None, \
        reassign_server_counts=None):
        # which_algo_mode = ('spa', 'dts', 'improved_dts', ... other algos)
        super(Single_Agent_UAV, self).__init__()
        
        current_dir = os.getcwd()
        env_params_file = "environment_class/environment_parameters.json"
        with open(os.path.join(current_dir, env_params_file),"r") as env_params_file_obj:
            env_params = json.load(env_params_file_obj)

        self.select_mli_randomly = False
        self.increase_num_ues = False
        self.reassign_server_counts_mode = False if (reassign_server_counts==None) else True
        self.new_server_counts = reassign_server_counts
        self.fs_bandwidth_plot_mode = False if (fs_bandwidth_plot_divisor==None) else True
        self.fs_bandwidth_divisor = fs_bandwidth_plot_divisor
        self.which_algo_mode = which_algo_mode
        self.copy_env_from_file = copy_env_from_file
        if self.reassign_server_counts_mode and ("UAV_count" in self.new_server_counts):
            self.UAV_count = self.new_server_counts["UAV_count"]
        else:
            self.UAV_count = env_params["UAV_count"]
        if self.reassign_server_counts_mode and ("UE_count" in self.new_server_counts):
            self.UE_count = self.new_server_counts["UE_count"]
        else:
            self.UE_count = env_params["UE_count"]
        if self.reassign_server_counts_mode and ("FS_count" in self.new_server_counts):
            self.FS_count = self.new_server_counts["FS_count"]
        else:
            self.FS_count = env_params["FS_count"]

        self.fraction_testing_mode = fraction_testing_mode
        if self.reassign_server_counts_mode and ("num_UE_centers" in self.new_server_counts):
            self.num_UE_centers = self.new_server_counts["num_UE_centers"]
        elif "num_UE_centers" not in env_params:
            self.num_UE_centers = self.UAV_count
        else:
            self.num_UE_centers = env_params["num_UE_centers"]
        self.Z = env_params["Z"]
        self.horizontal_dist_max = env_params["horizontal_dist_max"]
        self.max_angle = 2 * math.pi
        self.dz_max = env_params["dz_max"]
        self.phi_n = np.radians(42.44)  # in degrees
        self.boundary_x = env_params["boundary_x"]
        self.boundary_y = env_params["boundary_y"]
        self.C_max_t = (self.Z / np.tan(self.phi_n))
        self.D_min = max(50., (self.C_max_t) + self.boundary_x / 25.)
        self.max_episode_steps = env_params["max_episode_steps"] # Maximum number of steps in a single episode, after which environment returns done = True
        self.C_max_t_array = np.array([self.C_max_t for _ in range(self.UAV_count)])
        self.action_step_size = max(self.boundary_x, self.boundary_y) / self.max_episode_steps
        # self.max_episode_steps = int(max(self.boundary_x, self.boundary_y) / self.action_step_size) 
        # Maximum number of steps in a single episode, after which environment returns done = True
        self.timesteps_in_episode = 0
        self.done = False

        self.ue_centered_ratio_base = env_params["ue_centered_ratio_base"]
        self.ue_centered_ratio_per_uav_increment = env_params["ue_centered_ratio_per_uav_increment"]
        self.ue_centered_ratio_upper_bound = env_params["ue_centered_ratio_upper_bound"]
        self.UE_centered_ratio = min(self.ue_centered_ratio_upper_bound, self.ue_centered_ratio_base + \
                        (self.ue_centered_ratio_per_uav_increment * self.num_UE_centers))
        self.state_space_lb = np.array([0, 0], dtype = np.float32)
        self.state_space_ub = np.array([self.boundary_x, self.boundary_y], dtype = np.float32)
        
        self.overlappping_penalty = env_params["overlappping_penalty"]
        self.collision_penalty = env_params["collision_penalty"]

        # plotting parameters
        self.plotting_boundary_buffer = env_params["plotting_boundary_buffer"]
        self.UAV_coverage_circle_color = env_params["UAV_coverage_circle_color"]
        self.EC_concentrated_region_color = env_params["EC_concentrated_region_color"]
        self.UAV_path_color = env_params["UAV_path_color"]
        self.UE_served_by_UAV_color = env_params["UE_served_by_UAV_color"]
        self.UE_served_by_UAV_nvm_color = env_params["UE_served_by_UAV_nvm_color"]
        self.UE_served_by_FS_color = env_params["UE_served_by_FS_color"]
        self.UE_served_by_FS_nvm_color = env_params["UE_served_by_FS_nvm_color"]
        self.UE_served_by_Cloud_color = env_params["UE_served_by_Cloud_color"]

#         defining action space
        self.state_space = np.array([gym.spaces.box.Box(low = self.state_space_lb, high = self.state_space_ub) for i in range(self.UAV_count)])
#         defining observation space
        self.action_space = np.array([spaces.Discrete(9) for i in range(self.UAV_count)])
        
        if self.copy_env_from_file:
            dataset_folder_path = os.path.join(current_dir, "dataset")
            with open(os.path.join(dataset_folder_path, "system_data.json"), "r") as system_data_obj:
                copied_system_data = json.load(system_data_obj)
            with open(os.path.join(dataset_folder_path, "global_user_data.json"), "r") as global_user_data_obj:
                copied_global_user_data = json.load(global_user_data_obj)

            if self.UE_count > copied_system_data["system_component_counts"]["UE_count"]:
                self.increase_num_ues = True
                self.previous_ue_count = copied_system_data["system_component_counts"]["UE_count"]
        data_dict = utils.read_data_dict_from_file()
        #self.initial_UAV_state = initial_UAV_state 
        if not self.copy_env_from_file: 
            self.initial_UAV_state = self.select_random_state()
            self.system_data = {}
            self.global_user_data = {}
            self.system_data = system_comp_counts_filler.system_comp_counts_filler(self, self.system_data)
            self.system_data = mli_selection.select_mlis_for_system(data_dict, self.system_data)
            self.system_data = server_info_filler.server_info_filler(self, self.system_data, data_dict)
            self.place_UEs(position="centered", center = None, num_centers=self.num_UE_centers)
            self.current_state = copy.deepcopy(self.initial_UAV_state)
            self.system_data = fill_component_pos.populate_system_data_with_server_pos(self.system_data, \
                            server_beg_idx = 0, server_end_idx = (self.UAV_count), server_pos_ls = self.current_state)
            self.place_FSs()
        else:
            self.initial_UAV_state = np.array(copy_env_from_files.helper_return_current_server_set_state( \
                                copied_system_data, server_beg_idx=0, server_end_idx=self.UAV_count))
            # print(f"{self.initial_UAV_state=}")
            self.current_state = copy.deepcopy(self.initial_UAV_state)
            self.system_data = copied_system_data
            self.global_user_data = copied_global_user_data
            self.UE_positions = copy_env_from_files.helper_return_ue_pos_ls(copied_global_user_data)
            self.FS_positions = np.array(copy_env_from_files.helper_return_current_server_set_state( \
                            copied_system_data, server_beg_idx=self.UAV_count, server_end_idx=(self.UAV_count+self.FS_count)))
            self.UE_center, self.UE_radius = copy_env_from_files.helper_return_ue_center_radius(copied_system_data)
            if self.reassign_server_counts_mode:
                self.initial_UAV_state = self.select_random_state()
                self.current_state = copy.deepcopy(self.initial_UAV_state)
                self.system_data = system_comp_counts_filler.system_comp_counts_filler(self, self.system_data, new_component_counts=self.new_server_counts)
                self.system_data = server_info_filler.server_info_filler(self, self.system_data, data_dict, scale_uav_memory_according_to_num_uavs=True, variable_ue_count=True)
                self.system_data = fill_component_pos.populate_system_data_with_server_pos(self.system_data, \
                            server_beg_idx = 0, server_end_idx = (self.UAV_count), server_pos_ls = self.current_state)
                if self.increase_num_ues and (self.UE_count > self.previous_ue_count):
                    self.global_user_data = select_mlis_at_ue_placement.select_mli_for_ues(\
                        self.system_data, self.global_user_data, ue_beg_idx=self.previous_ue_count, \
                        ue_end_idx_exclusive=self.UE_count, select_mli_randomly=True)
                self.place_UEs(position="centered", center = None, num_centers=self.num_UE_centers, select_mlis=False)
                self.place_FSs()
            elif self.fs_bandwidth_plot_mode:
                # change bandwidth only if previously existing dataset is being used
                self.system_data = mli_selection.update_mli_fs_transmission_latencies_in_system_data( \
                                self.system_data, self.fs_bandwidth_divisor)
                init_user_data_filler.initialize_user_data(self.global_user_data, self.system_data)
        
        self.action_conversion()


    def action_conversion(self):
        self.n_actions = self.action_space[0].n
        # HARDCODED FOR n=9
        diag = (1 / math.sqrt(2))
        self.index_to_action_mapper = [(0, 0), (1, 0), (diag, diag), (0, 1), (-diag, diag), (-1, 0), (-diag, -diag), (0, -1), (diag, -diag)]  # angle changes in clockwise fashion
        return


    def reset(self):
        # RANDOMLY SELECTS UAV STARTING POSITION
        # self.current_state = self.select_random_state()

        # RESETS UAV STARTING POSITION TO self.initial_UAV_state
        self.current_state = copy.deepcopy(self.initial_UAV_state)
        # FOLLOWING 3 LINES ARE USED IF THE STATE SPACE INVOLVES THE UE COVERED BINARY VECTOR
        # horizontal_dist_UE_UAV = np.linalg.norm(self.UE_positions - self.current_state[:2], axis = 1)
        # rho_array = (horizontal_dist_UE_UAV <= (self.C_max_t)) * 1  # binary association vector
        # self.current_state = np.concatenate((self.current_state[:2], rho_array))
        self.timesteps_in_episode = 0
        self.done = False
        return self.current_state
    
    
    def get_count_of_UEs_covered(self):
        '''
        Returns count of UEs covered under UAV's current configuration
        '''
        # print(f"{self.current_state=}")
        ground_UAV_state = self.current_state[:, :2]
        horizontal_dist_UE_UAV = np.array([np.linalg.norm(self.UE_positions - ground_UAV_state[i], axis = 1) for i in range(self.UAV_count)])
        rho_matrix = (horizontal_dist_UE_UAV <= (self.C_max_t_array[:, None])) * 1  # binary association vector [(n_uav x n_ue) dimensions]
        Mn_t = ((rho_matrix.sum(axis = 0) >= 1) * 1)
        M_t = Mn_t.sum()  # total no. of UEs served by all the agent collectively
        utils.reward_in_reset_state(M_t)
        return M_t


    def render(self, **kwargs):
        '''
        Function to visualize UAV position
        (this function has different configurations for different use cases)
        '''

        if 'UAV_positions_list' in kwargs:
            self.render_UAV_movement_through_episode(UAV_positions_list = kwargs['UAV_positions_list'])

        if 'fig' not in kwargs:
            fig_size = 20
            fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
            ax = np.array([ax])
            kwargs['fig'] = fig
            kwargs['ax'] = ax
            kwargs['i'] = 0
        
        # if 'reward' not in kwargs:
        #     kwargs['reward'] = 'Not known'
        self.render_position_plot(**kwargs)


    def step(self, action):
        '''
        Takes action moving environment from current_state to next_state
        Arguments: `action` to be taken
        Returns: new_state, reward, done, info(=None)
        '''
        # action = action.numpy().flatten()
        # print("action inside step: ", action)
        for i in range(self.UAV_count):
            err_msg = f"{action[i]!r} ({type(action[i])}) for UAV {i} invalid"
            assert self.action_space[i].contains(action[i]), err_msg
        assert self.current_state is not None, "Call reset before using step method."

        self.timesteps_in_episode += 1
        prev_UAV_state = self.current_state
        _, inside_bounds = self.move(action)

        # make self.local_system_data and self.local_user_data by filtering out uncovered ues
        # calculate parent servers of ues and update in self.local_user_data
        # calculate parent servers of uavs and update in self.local_system_data
        self.local_system_data, self.local_user_data, self.local_to_global_ue_index_mapper = \
            filter_uncovered_ues.filter_uncovered_ues(self, self.system_data, self.global_user_data)
        # update power and energy fields of all local ues
        # print(f"{self.local_system_data=}")
        self.local_user_data = update_power_energy_of_local_ues.update_power_energy_of_local_ues(\
            self.local_system_data, self.local_user_data)
        # apply spa algo and computing matching and profit
        # print("intermediate step")
        improved_dts_matching = improved_dts_profit = dts_matching = dts_profit = spa_profit = spa_matching = None

        spa_revenue = None
        spa_energy_cost = None
        if self.fraction_testing_mode:
            improved_dts_profit, improved_dts_matching = temp_improved_dts.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=10)
            dts_profit, dts_matching = dts_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=10)
            # main profit is considered of our spa algo
            spa_result_ls = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=10)
            spa_profit, spa_matching, spa_revenue, spa_energy_cost = spa_result_ls
            profit, matching = spa_profit, spa_matching
            self.last_step_spa_profit, self.last_step_spa_matching = profit, matching
            self.last_step_dts_profit, self.last_step_dts_matching = dts_profit, dts_matching
            self.last_step_improved_dts_profit, self.last_step_improved_dts_matching = improved_dts_profit, improved_dts_matching
            self.last_step_profit, self.last_step_matching = self.last_step_spa_profit, self.last_step_spa_matching
            # include all other algos here
        elif self.which_algo_mode == "spa":
            spa_result_ls = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=10)
            spa_profit, spa_matching, spa_revenue, spa_energy_cost = spa_result_ls
            profit, matching = spa_profit, spa_matching
            self.last_step_spa_profit, self.last_step_spa_matching = profit, matching
            self.last_step_profit, self.last_step_matching = self.last_step_spa_profit, self.last_step_spa_matching
        elif self.which_algo_mode == "dts":
            dts_profit, dts_matching = dts_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=10)
            profit, matching = dts_profit, dts_matching
            self.last_step_dts_profit, self.last_step_dts_matching = dts_profit, dts_matching
            self.last_step_profit, self.last_step_matching = self.last_step_dts_profit, self.last_step_dts_matching
        elif self.which_algo_mode == "improved_dts":
            improved_dts_profit, improved_dts_matching = temp_improved_dts.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=10)
            profit, matching = improved_dts_profit, improved_dts_matching
            self.last_step_improved_dts_profit, self.last_step_improved_dts_matching = improved_dts_profit, improved_dts_matching
            self.last_step_profit, self.last_step_matching = self.last_step_improved_dts_profit, self.last_step_improved_dts_matching
        # TO-DO: calculate reward using the profit

        #check if overlapping constraint is met
        overlapping_penalty = 0
        collision_penalty = 0
        for uav1_idx in range(self.UAV_count):
            for uav2_idx in range(uav1_idx+1, self.UAV_count):
                euc_dist = np.linalg.norm(self.current_state[uav1_idx]\
                                          - self.current_state[uav2_idx])
                if euc_dist < (self.C_max_t_array[uav1_idx] + self.C_max_t_array[uav2_idx]):
                    overlapping_penalty += (2 * self.overlappping_penalty)
                
                if euc_dist < self.D_min:
                    collision_penalty = (2 * self.collision_penalty)

        reward = -10 * (collision_penalty + overlapping_penalty)

        # print(type(self.current_state), type(prev_UAV_state))
        if inside_bounds != 0:
            # MENTIONED OUTPUT IS OUT OF BOUNDS

            # self.is_done(end = True)
            reward += -1000
            self.is_done()
            if self.done:
                # print("episode end")
                if self.fraction_testing_mode:
                    profit1, matching1 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=1)
                    profit5, matching5 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=5)
                    profit15, matching15 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=15)
                    profit20, matching20 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=20)
                    profit25, matching25 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=25)
                    profit30, matching30 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=30)
                    # print(f"{profit1=}, {profit5=}, {profit=}, {profit15=}, {profit20=}, {profit25=}, {profit30=}")
                    utils.write_profit_and_matching_to_file([profit1, profit5, profit, profit15, profit20, profit25, profit30], \
                            [matching1, matching5, matching, matching15, matching20, matching25, matching30], normalizing_profit=dts_profit, \
                            normalizing_matching=dts_matching, num_ues=self.UE_count, fraction_testing_mode = self.fraction_testing_mode)
                else:
                    utils.write_profit_and_matching_to_file(profit={'spa_profit': spa_profit, 'dts_profit': dts_profit, 'improved_dts_profit': improved_dts_profit, \
                            'revenue': spa_revenue, 'energy_cost': spa_energy_cost}, matching={'spa_matching': spa_matching, 'dts_matching': dts_matching, 'improved_dts_matching': improved_dts_matching}, \
                            num_ues=self.UE_count, fraction_testing_mode = self.fraction_testing_mode, filename_prefix = self.which_algo_mode)
                
                # store self.local_system_data and self.local_user_data in json file
                if self.fraction_testing_mode:
                    filename_prefix = ""
                else:
                    filename_prefix = self.which_algo_mode
                utils.write_local_system_user_data_to_file(self.local_system_data, self.local_user_data, filename_prefix=filename_prefix)
                # print(f"{profit=}")

            return self.current_state, reward, self.done, None

        ground_UAV_state = self.current_state[:, :2]
        horizontal_dist_UE_UAV = np.array([np.linalg.norm(self.UE_positions - ground_UAV_state[i], axis = 1) for i in range(self.UAV_count)])
        rho_matrix = (horizontal_dist_UE_UAV <= (self.C_max_t_array[:, None])) * 1  # binary association vector [(n_uav x n_ue) dimensions]
        Mn_t = ((rho_matrix.sum(axis = 0) >= 1) * 1) 
        M_t = Mn_t.sum()  # total no. of UEs served by all the agent collectively

        reward += (M_t * 10)       
        self.is_done(M_t)

        if self.done:
            if self.fraction_testing_mode:
                profit1, matching1 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=1)
                profit5, matching5 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=5)
                profit15, matching15 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=15)
                profit20, matching20 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=20)
                profit25, matching25 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=25)
                profit30, matching30 = spa_algorithm_caller.main(system_data=self.local_system_data, user_data=self.local_user_data, fraction_bins=30)
                utils.write_profit_and_matching_to_file([profit1, profit5, profit, profit15, profit20, profit25, profit30], \
                        [matching1, matching5, matching, matching15, matching20, matching25, matching30], dts_profit=dts_profit, \
                        dts_matching=dts_matching, num_ues=self.UE_count, fraction_testing_mode = self.fraction_testing_mode)
            else:
                utils.write_profit_and_matching_to_file(profit={'spa_profit': spa_profit, 'dts_profit': dts_profit, 'improved_dts_profit': improved_dts_profit, \
                            'revenue': spa_revenue, 'energy_cost': spa_energy_cost}, matching={'spa_matching': spa_matching, 'dts_matching': dts_matching, 'improved_dts_matching': improved_dts_matching}, \
                            num_ues=self.UE_count, fraction_testing_mode = self.fraction_testing_mode, filename_prefix = self.which_algo_mode)
                # utils.write_profit_and_matching_to_file(profit={'spa_profit': spa_profit, 'dts_profit': dts_profit}, matching={'spa_matching': spa_matching, 'dts_matching': dts_matching}, \
                #         num_ues=self.UE_count, fraction_testing_mode = self.fraction_testing_mode)
            
            # store self.local_system_data and self.local_user_data in json file
            if self.fraction_testing_mode:
                filename_prefix = ""
            else:
                filename_prefix = self.which_algo_mode
            utils.write_local_system_user_data_to_file(self.local_system_data, self.local_user_data, filename_prefix=filename_prefix)
            # store self.local_system_data and self.local_user_data in json file
            
        return self.current_state, reward, self.done, None
    

    def move(self, action):
        '''
        Helper function to step() function.
        Clips the passed action to fit within action space bounds.
        Calculates new state after performing the passed action, and updates UAV position accordingly. 
        '''
#         evaluates new state reached upon performing the move and saves it in self.current_state
        # ACTION = [dx, dy]
        x_next_array = np.array([(self.current_state[i][0] + (self.index_to_action_mapper[action[i]][0] * self.action_step_size)) for i in range(self.UAV_count)], dtype=np.float32)
        y_next_array = np.array([(self.current_state[i][1] + (self.index_to_action_mapper[action[i]][1] * self.action_step_size)) for i in range(self.UAV_count)], dtype=np.float32)
        
    # updating horizontal_direction_angle if the new move is out of boundary
        positive_reward = True
        if ((x_next_array < 0).any()) or ((y_next_array < 0).any()) or ((x_next_array > self.boundary_x).any()) or ((y_next_array > self.boundary_y).any()):
            # UAV REMAINS IN ITS CURRENT POSITION
            positive_reward = False

        x_next_array = np.clip(x_next_array, a_min=np.array([0 for i in range(self.UAV_count)]), a_max=np.array([self.boundary_x for i in range(self.UAV_count)]))
        y_next_array = np.clip(y_next_array, a_min=np.array([0 for i in range(self.UAV_count)]), a_max=np.array([self.boundary_y for i in range(self.UAV_count)]))
        self.current_state[:, :2] = np.array([[x_next_array[i], y_next_array[i]] for i in range(self.UAV_count)])
        return action, positive_reward


    def is_done(self, M_t=0, end=False):
        '''
        Helper function to check if episode needs to be terminated
        '''
        if end == True:
            self.done = True
        if M_t >= int((self.UE_centered_ratio-0.1) * self.UE_count):
            self.done = True
        elif(self.timesteps_in_episode >= self.max_episode_steps):
            self.done = True
        # print(f"{self.done=}")
        return

    
    def select_random_state(self):
        '''
        Selects (and returns) random initial state (within bounds) for the UAV
        '''
        UAV_pos_list = []
        for _ in range(self.UAV_count):
            new_x = np.random.uniform(0.0, self.boundary_x)
            new_y = np.random.uniform(0.0, self.boundary_y)
            UAV_pos_list.append(np.array([new_x, new_y]))

        return np.array(UAV_pos_list)


    def place_UEs_centered(self, center = None, centered_UE_count = None, radius = None):
        '''
        Helper function to place_UEs
        Places `centered_UE_count` UEs within a circle with center `center` and radius `C_max_t`.
        Places remaining UEs (UE_count - centered_UE_count) randomly inside the area barring the above circular region
        Arguments: 
            desired_z_coord: Z coordinate with which to calculate radius C_max_t for centering UEs
            center: Provides center of circular region in which UEs will be scattered
            centered_UE_count: Number of UEs to be placed within the circular region created using previous arguments
        '''

        if centered_UE_count is None:
          centered_UE_count = self.UE_count
        # focuses the UEs inside the circular region
        if radius is None:
            radius = (5 * self.C_max_t) / 7
        if center is None:
            # (x, y) co-ordinates
            x = np.random.uniform(0, self.boundary_x)
            y = np.random.uniform(0, self.boundary_y)
            center = np.array([x, y])

        r = radius * np.sqrt(np.random.uniform(size = centered_UE_count))
        theta = np.random.uniform(size = centered_UE_count) * 2 * math.pi

        if self.UE_center is None:
            self.UE_center = []
            self.UE_radius = []
        self.UE_center.append(center)
        self.UE_radius.append(radius)

        if self.UE_positions is None: 
            self.UE_positions = np.zeros((centered_UE_count, 2))
            self.UE_positions[:, 0] = np.clip(center[0] + r * np.cos(theta), 0., self.boundary_x) 
            self.UE_positions[:, 1] = np.clip(center[1] + r * np.sin(theta), 0., self.boundary_y)
        else:
            current_iteration_UE_positions = np.zeros((centered_UE_count, 2))
            current_iteration_UE_positions[:, 0] = np.clip(center[0] + r * np.cos(theta), 0., self.boundary_x) 
            current_iteration_UE_positions[:, 1] = np.clip(center[1] + r * np.sin(theta), 0., self.boundary_y)
            # print("self.UE_positions: ", self.UE_positions)
            # print("current_iteration_UE_positions: ", current_iteration_UE_positions) 
            self.UE_positions = np.concatenate((self.UE_positions, current_iteration_UE_positions))

        return


    def place_UEs_randomly(self, random_UE_count = None, **kwargs):
        '''
        Helper function to place_UEs
        places `random_UE_count` UEs randomly onto the rectangular region
        if kwargs has the key `exclude_center`, then the circular region spanned by `exclude_center` \
        and `exclude_radius` is excluded

        '''

        if random_UE_count is None:
          random_UE_count = self.UE_count

        # places UE_count UEs on grid randomly
        if type(self.UE_positions) == np.ndarray:
            self.UE_positions = self.UE_positions.tolist()

        randomly_placed_count = 0
        while(randomly_placed_count < random_UE_count):
            x = np.random.uniform(0, self.boundary_x)
            y = np.random.uniform(0, self.boundary_y)
            coords = np.array([x, y])
            if 'exclude_center' in kwargs:
                if (np.linalg.norm(coords - kwargs['exclude_center']) <= kwargs['exclude_radius']):
                    continue
            self.UE_positions.append(coords)
            randomly_placed_count += 1

        self.UE_positions = np.array(self.UE_positions)


    def get_UE_cluster_center(self, current_cluster_radius, exclude_center_list = None, exclude_radius_list = None):
        
        buffer = 20.
        max_allowed_iterations = 10000
        idx = 0
        while True:
            if idx >= max_allowed_iterations:
                break
            idx += 1
            new_x = np.random.uniform(self.C_max_t, self.boundary_x - self.C_max_t)
            new_y = np.random.uniform(self.C_max_t, self.boundary_y - self.C_max_t)
            if exclude_center_list is not None:
                # print("exclude_center_list: ", exclude_center_list)
                # print("chosen coordinate: ", np.array([new_x, new_y]))
                chosen_center_to_excluded_centers_dist_list = np.linalg.norm(exclude_center_list - np.array([new_x, new_y]), axis = 1)
                if ((chosen_center_to_excluded_centers_dist_list <= (exclude_radius_list + current_cluster_radius + buffer)) * 1).any():
                    continue
                break

        return np.array([new_x, new_y])


    def place_UEs(self, position="random", center = None, num_centers = 3, select_mlis=True):
        '''
        Function to place UEs onto the rectangular region
        Arguments:
            position: has 2 modes, "random" and "centered", representing the two configuration for scattering UEs
            desired_z_coord: if position="centered", this argument gives height with which to calculate radius of circular region
            center: if position="centered", this argument provides center of circular region
        '''

        self.UE_center = None  # initializing for non-centered generation algorithms
        self.UE_positions = None

        if(position=="centered"):
            centered_count = int(self.UE_centered_ratio * self.UE_count)
            per_center_count = int(centered_count / num_centers)
            exclude_center_list = None
            exclude_radius_list = None
            curr_ue_idx = 0
            for i in range(num_centers):
                current_UE_cluster_center = self.get_UE_cluster_center(self.C_max_t, exclude_center_list = exclude_center_list, exclude_radius_list = exclude_radius_list)
                # print(i, "th cluster center: ", current_UE_cluster_center)
                self.place_UEs_centered(center = current_UE_cluster_center, centered_UE_count = per_center_count)
                if select_mlis:
                    self.global_user_data = select_mlis_at_ue_placement.select_mli_for_ues(self.system_data, self.global_user_data, ue_beg_idx=curr_ue_idx, ue_end_idx_exclusive=(curr_ue_idx+per_center_count), select_mli_randomly=self.select_mli_randomly)
                self.global_user_data = fill_component_pos.populate_user_data_with_ue_positions(user_data = self.global_user_data, ue_pos_ls = self.UE_positions, beg_ue_idx = curr_ue_idx, end_ue_idx = (per_center_count + curr_ue_idx))
                curr_ue_idx += per_center_count
                if exclude_center_list is None:
                    exclude_center_list = np.array([current_UE_cluster_center])
                    exclude_radius_list = np.array([self.UE_radius[-1]])
                else:
                    exclude_center_list = np.concatenate((exclude_center_list, np.array([current_UE_cluster_center])))
                    # print("exclude_radius_list: ", exclude_radius_list)
                    # print("current_radius: ", np.array(self.UE_radius[-1]))
                    exclude_radius_list = np.concatenate((exclude_radius_list, [np.array(self.UE_radius[-1])]))
            self.system_data = fill_component_pos.store_ue_center_and_radius(system_data = self.system_data, ue_center_ls = self.UE_center, ue_radius_ls = self.UE_radius)
            
            # Placing remaining UAVs randomly
            if((self.UE_count - (per_center_count * num_centers)) > 0):
                self.place_UEs_randomly(self.UE_count - (per_center_count * num_centers))
                if select_mlis:
                    self.global_user_data = select_mlis_at_ue_placement.select_mli_for_ues(self.system_data, self.global_user_data, ue_beg_idx=curr_ue_idx, select_mli_randomly=self.select_mli_randomly)
                self.global_user_data = fill_component_pos.populate_user_data_with_ue_positions(user_data = self.global_user_data, ue_pos_ls = self.UE_positions, beg_ue_idx = curr_ue_idx)
        
        elif(position=="random"):
            self.place_UEs_randomly()
            if select_mlis:
                self.global_user_data = select_mlis_at_ue_placement.select_mli_for_ues(self.system_data, self.global_user_data, select_mli_randomly=self.select_mli_randomly)
            self.global_user_data = fill_component_pos.populate_user_data_with_ue_positions(user_data = self.global_user_data, ue_pos_ls = self.UE_positions, beg_ue_idx = curr_ue_idx)

        # saves in self.UE_positions
        self.UE_positions = np.array(self.UE_positions)
        utils.gsa(position)
        # print(f"{self.global_user_data}")
        self.global_user_data = init_user_data_filler.initialize_user_data(self.global_user_data, self.system_data)
        return


    def place_FSs(self):
        self.FS_positions = []
        for fs_idx in range(self.FS_count):
            fs_x = random.randint(0, self.boundary_x)
            fs_y = random.randint(0, self.boundary_y)
            self.FS_positions.append([fs_x, fs_y])
        self.FS_positions = np.array(self.FS_positions)
        fs_beg_idx = self.UAV_count
        self.system_data = fill_component_pos.populate_system_data_with_server_pos(self.system_data, \
                        server_beg_idx = fs_beg_idx, server_end_idx = (fs_beg_idx + self.FS_count), server_pos_ls = self.FS_positions)
        return


    def render_UAV_movement_through_episode(self, **kwargs):
        '''
        Helper function to visualize UAV movement through an episode
        Arguments: A list specifying UAV positions throughout the episode
        '''

        # if 'reward' not in kwargs:
        #     kwargs['reward'] = 'Not known'
        
        buffer = 5.
        fig_size = 20
        fig, ax = plt.subplots(1, figsize=(fig_size,fig_size))
        ax.set_xlim(-buffer, self.boundary_x+buffer)
        ax.set_ylim(-buffer, self.boundary_y+buffer)
        ax.grid()
        UE_x = self.UE_positions[:, 0]
        UE_y = self.UE_positions[:, 1]
        ue_plot, = ax.plot(UE_x, UE_y, color='black', marker='o', markersize=6, linestyle = '')
        
        FS_x = self.FS_positions[:, 0]
        FS_y = self.FS_positions[:, 1]
        fs_plot, = ax.plot(FS_x, FS_y, color='red', marker='D', markersize=13, linestyle = '')
        
        for i in range(self.UAV_count):
            # uav_path, = ax.plot(kwargs['UAV_positions_list'][:, i, 0] , kwargs['UAV_positions_list'][:, i, 1], color=self.UAV_path_color, markersize=6, linestyle = '-') #, label = "UAV Path Color")
            x, y = kwargs['UAV_positions_list'][:, i, 0], kwargs['UAV_positions_list'][:, i, 1]

            uav_plot, = ax.plot(kwargs['UAV_positions_list'][-1, i, 0], kwargs['UAV_positions_list'][-1, i, 1], color='blue', marker='x', markeredgewidth=3, markersize=13, linestyle = '') #, label = "UAV ending position")
    
            UAV_coverage_area = plt.Circle((kwargs['UAV_positions_list'][-1, i, 0], kwargs['UAV_positions_list'][-1, i, 1]), self.C_max_t, color = self.UAV_coverage_circle_color)
            ax.add_artist(UAV_coverage_area)
            ax.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], color=self.UAV_path_color, scale_units='xy', angles='xy', scale=0.9, linewidths=[0.5 for _ in range(len(x)-1)])
      
        profit=None
        matching={}
        filename_prefix=""
        if not self.fraction_testing_mode:
            filename_prefix=self.which_algo_mode
        profit_dict, matching_dict = utils.read_profit_and_matching_from_file(filename_prefix=filename_prefix)
        # print(f"inside render_movement..., {profit_dict=}")
        # print(f"inside render_UAV_movement function, {profit=}, {matching=}")
        if not self.fraction_testing_mode:
            profit = profit_dict[self.which_algo_mode + "_profit"]
        if not self.fraction_testing_mode:
            matching = matching_dict[self.which_algo_mode + "_matching"]
        
        # print(f"{matching=}")
        ue_served_by_uav_plot, ue_served_by_fs_plot, ue_served_by_cloud_plot = None, None, None
        for local_ue_idx in matching:
            global_ue_idx = self.local_to_global_ue_index_mapper[int(local_ue_idx)]
            server = matching[local_ue_idx][2]
            if server < self.UAV_count:
                # print("in 1")
                ue_served_by_uav_plot, = ax.plot(self.UE_positions[global_ue_idx, 0], \
                    self.UE_positions[global_ue_idx, 1], color=self.UE_served_by_UAV_color, marker='o', \
                    markersize=6, linestyle = '')
            elif server < (self.FS_count + self.UAV_count):
                # print("in 2")
                ue_served_by_fs_plot, = ax.plot(self.UE_positions[global_ue_idx, 0], \
                    self.UE_positions[global_ue_idx, 1], color=self.UE_served_by_FS_color, marker='o', \
                    markersize=6, linestyle = '')
            else:
                # print("in 3")
                ue_served_by_cloud_plot, = ax.plot(self.UE_positions[global_ue_idx, 0], \
                    self.UE_positions[global_ue_idx, 1], color=self.UE_served_by_Cloud_color, marker='o', \
                    markersize=6, linestyle = '')

    #   # For testing: To plot the circle and visualize UE points inside it
    #     if self.UE_center is not None:
    #         for i in range(len(self.UE_center)):
    #             Test_UE_allotment_circle = plt.Circle((self.UE_center[i][0] , self.UE_center[i][1]), self.UE_radius[i], color = self.EC_concentrated_region_color)
    #             ax.add_artist(Test_UE_allotment_circle)
    #   # End

        ax.set_aspect(1)
        # naming the x axis
        ax.set_xlabel('X pos (m)', fontdict={'fontsize': 22})
        # naming the y axis
        ax.set_ylabel('Y pos (m)', fontdict={'fontsize': 22})
        # ax.set_title('UAV movement through episode', fontsize=22)
        # giving a title to my graph
        # plt.title('Visually Appealing!')

        # show a legend on the plot
        plot_object_ls = [ue_plot, uav_plot, fs_plot]
        plot_titles_ls = ["unserved UEs", "UAV", "FS"]
        if ue_served_by_uav_plot is not None:
            plot_object_ls.append(ue_served_by_uav_plot)
            plot_titles_ls.append("UEs served by UAVs")
        if ue_served_by_fs_plot is not None:
            plot_object_ls.append(ue_served_by_fs_plot)
            plot_titles_ls.append("UEs served by FSs")
        if ue_served_by_cloud_plot is not None:
            plot_object_ls.append(ue_served_by_cloud_plot)
            plot_titles_ls.append("UEs served by the cloud")
        plot_object_ls.append(UAV_coverage_area)
        plot_titles_ls.append("UAV Coverage Area")
        ax.legend(plot_object_ls, plot_titles_ls, fontsize="21", ncol=2)
        
        ax.tick_params(labelsize='22')
        if "title" in kwargs:
            ax.set_title(kwargs["title"], fontdict={'fontsize': 24})
        elif "reward" in kwargs:
            ax.set_title("reward: "+str(kwargs["reward"]), fontdict={'fontsize': 24})
        else:
            ax.set_title("Movement of UAV through an episode (post training)", fontdict={'fontsize': 24})
        
        filename_prefix = ""
        if not self.fraction_testing_mode:
            filename_prefix = self.which_algo_mode
        if "render_plot_path" in kwargs:
            plt.savefig(os.path.join(kwargs["render_plot_path"], filename_prefix + "uav_movement_plot.png"))
            plt.close(fig)
        else:
            plt.show()
        # plt.show()
        return


    def render_stacked_bar_plot(self, **kwargs):
        
        if self.fraction_testing_mode:
            return
        
        server_colors = [
                    self.UE_served_by_UAV_color, self.UE_served_by_UAV_nvm_color, \
                    self.UE_served_by_FS_color, self.UE_served_by_FS_nvm_color, \
                    self.UE_served_by_Cloud_color \
                ]
        server_labels = [
                    "served by trad. memory on UAV", "served by NVM memory on UAV", \
                    "served by trad. memory on UAV", "served by NVM memory on FS", \
                    "served by Cloud" \
                ]
        
        fig_size = 11
        fig, ax = plt.subplots(1, figsize=(fig_size,fig_size))

        filename_prefix=""
        if not self.fraction_testing_mode:
            filename_prefix=self.which_algo_mode
        profit_dict, matching_dict = utils.read_profit_and_matching_from_file(filename_prefix=filename_prefix)
        profit = profit_dict[self.which_algo_mode + "_profit"]
        matching = matching_dict[self.which_algo_mode + "_matching"]
        # y_stack_ls = [([0]*self.UE_count)]*5
        y_stack_ls = [[0 for i in range(self.UE_count)] for j in range(5)]
        for local_ue_idx in matching:
            global_ue_idx = self.local_to_global_ue_index_mapper[int(local_ue_idx)]
            server = matching[local_ue_idx][2]
            fraction = matching[local_ue_idx][3]
            if server < self.UAV_count:
                # print(f"in uav, {server=}")
                y_stack_ls[0][global_ue_idx] = (fraction / 10)
                y_stack_ls[1][global_ue_idx] = 1 - (fraction / 10)
            elif server < (self.UAV_count + self.FS_count):
                # print(f"in fs, {server=}")
                y_stack_ls[2][global_ue_idx] = (fraction / 10)
                y_stack_ls[3][global_ue_idx] = 1 - (fraction / 10)
            else:
                # print(f"in cloud, {server=}")
                y_stack_ls[4][global_ue_idx] = (fraction / 10)

            # print(y_stack_ls[0][global_ue_idx], y_stack_ls[1][global_ue_idx], \
            #       y_stack_ls[2][global_ue_idx], y_stack_ls[3][global_ue_idx], \
            #       y_stack_ls[4][global_ue_idx])
        
        # y_stack_bottom_ls = [([0]*self.UE_count)]*5
        y_stack_bottom_ls = [[0 for i in range(self.UE_count)] for j in range(5)]

        for ue_idx in range(self.UE_count):
            till_max_y = 0
            for i in range(5):
                # val=y_stack_ls[i][ue_idx]
                y_stack_bottom_ls[i][ue_idx] = max(y_stack_ls[i][ue_idx], till_max_y)
                till_max_y = y_stack_bottom_ls[i][ue_idx]

        x_range = [i for i in range(1, self.UE_count+1)]
        for i in range(5):
            if i!=0:
                ax.bar(x_range, y_stack_ls[i], bottom=y_stack_bottom_ls[i-1], color=server_colors[i])
            else:
                ax.bar(x_range, y_stack_ls[i], color=server_colors[i])

        ax.tick_params(labelsize='15')
        # naming the x axis
        ax.set_xlabel('UE Index', fontdict={'fontsize': 16})
        # naming the y axis
        ax.set_ylabel('Fraction of MLI Memory', fontdict={'fontsize': 16})
        legend_elements = []
        for i in range(5):
            legend_elements.append(mpatches.Patch(facecolor=server_colors[i], label=server_labels[i]))
        ax.legend(handles=legend_elements, title='Memory configuration of MLI hosted by UE x', title_fontsize=18, bbox_to_anchor=(0.15, 1.02), ncols=2)
        # ax.legend(handles=legend_elements, fontsize="15", ncol=1)

        if "render_plot_path" in kwargs:
            plt.savefig(os.path.join(kwargs["render_plot_path"], filename_prefix + "ue_service_plot.png"))
            plt.close(fig)
        else:
            plt.show()
        pass


    def render_policy_plot(self, **kwargs):

        fig, ax = plt.subplots(1, figsize=(10,10))
        position_action_list = kwargs['position_action_list']
        for position_action in position_action_list:
            position = position_action[0]
            action = position_action[1]
            new_state = self.move(action, get_new_state=True)
            ax.arrow(position[0], position[1], new_state[0] - position[0], new_state[1] - position[1], head_width=0.09, head_length=0.1)
        plt.show()
        return


    def render_bar_plot(self, ax, plot_number, idx):
        '''
        Plotting bar graph of each UE's fraction of tasks assigned to UAV and to EC, by the last action input given to step() function
        '''
        x_indices = np.arange(self.UE_count)
        width = 0.5

        gamma_array = (np.array(self.last_action[4:])).reshape((self.UE_count, self.EC_count))
        gamma_zeros = (1 - gamma_array.sum(axis = 1))
        
        ax[plot_number, idx].bar(x_indices, gamma_zeros, width=width)
        for i in range(self.EC_count):
            ax[plot_number, idx].bar(x_indices, gamma_array[:, i], width=width)
        
    
    def render_position_plot(self, **kwargs):
        '''
        Helper function to render(), plots the current position plot on given axes. 
        Plotting position plot of UAV's current position and coverage
        '''
        ax = kwargs['ax']
        fig = kwargs['fig']
        plot_number = kwargs['i']
        idx = 0
        # print(f"{ax.shape=}")
        buffer = 5.
        ax[plot_number].set_xlim(-buffer, self.boundary_x+buffer)
        ax[plot_number].set_ylim(-buffer, self.boundary_y+buffer)
        ax[plot_number].grid()
        UE_x = self.UE_positions[:, 0]
        UE_y = self.UE_positions[:, 1]
        ue_plot, = ax[plot_number].plot(UE_x, UE_y, color='black', marker='o', markersize=6, linestyle = '')
        
        FS_x = self.FS_positions[:, 0]
        FS_y = self.FS_positions[:, 1]
        fs_plot, = ax[plot_number].plot(FS_x, FS_y, color='red', marker='D', markersize=13, linestyle = '')

        
        for i in range(self.UAV_count):
            uav_plot, = ax[plot_number].plot(self.current_state[i][0], self.current_state[i][1], color='blue', marker='x', markeredgewidth=3, markersize=13, linestyle = '')
            UAV_coverage_area = plt.Circle((self.current_state[i][0] , self.current_state[i][1] ), self.C_max_t, color = self.UAV_coverage_circle_color)
            ax[plot_number].add_artist(UAV_coverage_area)
      
        # For testing: To plot the circle and visualize UE points inside it
        if ("testing_mode" in kwargs) and (kwargs["testing_mode"] == True) \
            and (self.UE_center is not None):
            for i in range(len(self.UE_center)):
                # print("self.UE_radius inside render_position_plot: ", self.UE_radius)
                Test_UE_allotment_circle = plt.Circle((self.UE_center[i][0] , self.UE_center[i][1]), self.UE_radius[i], color = self.EC_concentrated_region_color)
                ax[plot_number].add_artist(Test_UE_allotment_circle)
        # End

        if 'learnt_policy_visualization' in kwargs:
            position_action_list = kwargs['position_action_list']
            for position_action in position_action_list:
                position = position_action[0]
                action = position_action[1]
                action = np.clip(action, self.action_space_lb, self.action_space_ub)
                action = (((self.action_space_coversion_ub - self.action_space_coversion_lb) * (action + 1)) / 2) + self.action_space_coversion_lb
                print("scaled action: ", action)
                # print("action after scaling: ", action)
                new_state = self.move(action, get_new_state=True, provided_center=position)
                print("previous state: ", position, ", new state: ", new_state)
                # print("new_state: ", new_state)
                ax[plot_number].plot(position[0], position[1], color='blue', marker='o', markersize=6, linestyle = '')
                # ax[plot_number, idx].plot(new_state[0], new_state[1], color='orange', marker='o', markersize=6, linestyle = '')
                ax[plot_number].arrow(position[0], position[1], new_state[0] - position[0], new_state[1] - position[1], head_width=0.2, head_length=0.1)

        ax[plot_number].set_aspect(1)
        ax[plot_number].tick_params(labelsize='22')
        # naming the x axis
        ax[plot_number].set_xlabel('X pos (m)', fontdict={'fontsize': 22})
        # naming the y axis
        ax[plot_number].set_ylabel('Y pos (m)', fontdict={'fontsize': 22})
        if "title" in kwargs:
            ax[plot_number].set_title(kwargs["title"], fontdict={'fontsize': 24})
        # elif "reward" in kwargs:
        #     ax[plot_number].set_title("reward: "+str(kwargs["reward"]), fontdict={'fontsize': 24})
        else:
            ax[plot_number].set_title('Fixed UE Positions in the MEC System', fontdict={'fontsize': 24})
        # ax[plot_number].set_title('reward: ' + str(reward))
        # giving a title to my graph
        # plt.title('Visually Appealing!')

        # show a legend on the plot
        # blue_patch = mpatches.Patch(color='blue', label='Current UAV coverage positions')
        ax[plot_number].legend([ue_plot, uav_plot, fs_plot, UAV_coverage_area], ["UE", "UAV", "FS", "Initial UAV Coverage Area"], fontsize="19", ncol=3)
        
        filename_prefix = ""
        if not self.fraction_testing_mode:
            filename_prefix = self.which_algo_mode
        if "render_plot_path" in kwargs:
            plt.savefig(os.path.join(kwargs["render_plot_path"], filename_prefix + "ue_position_plot.png"))
            plt.close(fig)
        else:
            # plt.show()
            pass
        # plt.show()