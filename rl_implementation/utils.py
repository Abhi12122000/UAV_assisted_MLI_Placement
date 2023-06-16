import matplotlib.pyplot as plt
import torch
import numpy as np
import environment_class.environment as environment
import os
import json
import math
import copy
import random
import glob


def plot_stats1(stats, plot_file):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]['data']
        # print(f"{vals=}")
        for j in range(len(vals)):
            if(torch.is_tensor(vals[j])):
                vals[j] = vals[j].detach().numpy()
        vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
            # naming the x axis
            ax[i].set_xlabel(stats[key]['x_label'])
            # naming the y axis
            ax[i].set_ylabel(stats[key]['y_label'])
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
            # naming the x axis
            ax.set_xlabel(stats[key]['x_label'])
            # naming the y axis
            ax.set_ylabel(stats[key]['y_label'])
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close(fig)


def run_random_episode(env, policy, q_nets, render_plot_path):
  
  state = env.reset()
  env.render(render_plot_path=render_plot_path)
  # print("beginning state: ", state[0])
  done = False
  UAV_positions_list = [state.numpy()[0]]
  while not done:
    # print(state.numpy())
    num_actions = env.action_space[0].n
    action = torch.tensor([[[policy(state=torch.flatten(state, end_dim=-1), q_networks=q_nets, env=env, epsilon=0., UAV_index=i, num_actions=num_actions)] for i in range(env.UAV_count)]])
    np_action = [act.item() for act in action[0]]

    # print(np_action)
    # action = policy(state, epsilon=0.)
    # print("action: ", action)
    new_state, reward, done, _ = env.step(np_action)
    # print("new_state: ", new_state[0][:2])
    state = new_state
    UAV_positions_list.append(state.numpy()[0])
    
  if env.fraction_testing_mode:
    print("improved_dts_profit: ", env.last_step_improved_dts_profit, "dts profit: ", env.last_step_dts_profit, "spa profit: ", env.last_step_spa_profit)
#   print("dts matching: ", env.last_step_dts_matching, "spa matching: ", env.last_step_matching)
  elif env.which_algo_mode == "spa":
    print("spa profit: ", env.last_step_spa_profit)
  elif env.which_algo_mode == "dts":
    print("dts profit: ", env.last_step_dts_profit)
  elif env.which_algo_mode == "improved_dts":
      print("improved_dts profit: ", env.last_step_improved_dts_profit)
  UAV_positions_list = np.array(UAV_positions_list)
  # print(UAV_positions_list)
  # print(UAV_positions_list[:, 0, 0])
  env.render_UAV_movement_through_episode(UAV_positions_list = UAV_positions_list, render_plot_path=render_plot_path)
  if env.UE_count <= 100:
    env.render_stacked_bar_plot(render_plot_path=render_plot_path)
  return


def ffs2(filename_prefix=""):
    a = environment.Single_Agent_UAV()
    a.reset()
    x = a.current_state
    a.reset()
    if (a.current_state == x).all():
        random_reset = False
    else:
        random_reset = True

    current_dir = os.getcwd()
    carrier_files_in_environment_class = glob.glob(os.path.join(current_dir, "environment_class", "*carrier.json"))
    with open(carrier_files_in_environment_class[0], "r") as infile:
        carrier_dict = json.load(infile)
    carrier_dict["random_reset"] = random_reset
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "w") as outfile:
            json.dump(carrier_dict, outfile, indent=4)
    return


def backtrack_2_construct_uav_2_ue_center_mapping(curr_uav_idx, binary_arr, uavs, ue_centers, curr_dict, best_dict, curr_dist, best_dist):
    if curr_uav_idx == len(uavs):
        if curr_dist < best_dist:
            best_dict = copy.deepcopy(curr_dict)
            best_dist = curr_dist
            # print("current best dict: ", best_dict)
        return best_dict, best_dist
    
    for idx in range(len(binary_arr)):
        if binary_arr[idx]==0:
            this_dist = np.linalg.norm(np.array(np.array(uavs[curr_uav_idx]) - np.array(ue_centers[idx])))
            curr_dist += this_dist
            binary_arr[idx] = 1
            curr_dict[curr_uav_idx] = idx
            best_dict, best_dist = backtrack_2_construct_uav_2_ue_center_mapping(curr_uav_idx=curr_uav_idx+1, \
                binary_arr=binary_arr, uavs=uavs, ue_centers=ue_centers, \
                curr_dict=curr_dict, best_dict=best_dict, curr_dist=curr_dist, \
                best_dist=best_dist)
            curr_dist -= this_dist
            binary_arr[idx] = 0
            del curr_dict[curr_uav_idx]
    return best_dict, best_dist


def greedy_helper_2_construct_uav_2_ue_center_mapping(env):
    mapping_ = {}
    uav_posns = env.current_state
    ue_centers = env.UE_center
    dist_ls = sorted([[(np.linalg.norm(np.array(uav_posns[uav_idx]) - np.array(ue_centers[ue_center_idx])), uav_idx, ue_center_idx) for uav_idx in range(len(uav_posns))] for ue_center_idx in range(len(ue_centers))], reverse=True, key=lambda x: x[0])
    # dist_ls --> reverse sorted list of (distance, uav_idx, ue_center_idx)
    while len(dist_ls) > 0:
        if dist_ls[-1][1] not in mapping_:
            mapping_[dist_ls[-1][1]] = dist_ls[-1][2]
        dist_ls.pop()
    return mapping_


def brute_force_helper_2_construct_uav_2_ue_center_mapping(env):
    mapping_uav_2_ue_centers = {}
    optimal_dict = {}
    optimal_dist = 1000000
    curr_dict = {}
    curr_dist = 0
    binary_arr = [0 for _ in range(env.UAV_count)]
    # print(f"{len(env.current_state)=}, {len(env.UE_center)=}")
    optimal_dict, optimal_dist = backtrack_2_construct_uav_2_ue_center_mapping(0, binary_arr=binary_arr,\
        uavs=env.current_state, ue_centers=env.UE_center, curr_dict=curr_dict,\
        best_dict=optimal_dict, curr_dist=curr_dist, best_dist=optimal_dist)
    # print(f"in brute_force_recursion_caller_func, {optimal_dict=}")
    return optimal_dict


def construct_uav_2_ue_center_mapping(env):
    if (env.UE_center is None) or (env.UE_center == []):
        print("No UE centers found!!!")
        raise NotImplementedError
    
    if len(env.UE_center) != env.UAV_count:
        print("num(UE_centers) != num(UAVs)... exiting!!!")
        raise NotImplementedError
    
    if len(env.UE_center) == env.UAV_count:
        if env.UAV_count > 10:
            # call greedy function
            mapping_ = greedy_helper_2_construct_uav_2_ue_center_mapping(env)
        else:
            # call optimal backtracking function
            mapping_ = brute_force_helper_2_construct_uav_2_ue_center_mapping(env)
    return mapping_


def write_profit_and_matching_to_file(profit, matching, normalizing_profit=None, \
        normalizing_matching=None, num_ues=1, fraction_testing_mode=False, filename_prefix=""):
    current_dir = os.getcwd()
    if "dataset" not in os.listdir(current_dir):
        os.mkdir(os.path.join(current_dir, "dataset"))
        
    if "local_dataset" not in os.listdir(os.path.join(current_dir, "dataset")):
        os.mkdir(os.path.join(current_dir, "dataset", "local_dataset"))
    
    local_data_directory = os.path.join(current_dir, "dataset", "local_dataset")
    
    profit_matching_dict = {'UE_count': num_ues}
    profit_matching_dict['profit'] = profit
    if normalizing_profit is not None:
        profit_matching_dict['normalizing_profit'] = normalizing_profit
    if (not fraction_testing_mode):
        profit_matching_dict['matching'] = matching
        if normalizing_matching is not None:
            profit_matching_dict['normalizing_matching'] = normalizing_matching
    # # COMMENTED FOR TESTING
    # profit_matching_dict = {'profit': profit, 'matching': matching, 'UE_count': num_ues}
    # # END 
    with open(os.path.join(local_data_directory, filename_prefix + "matching_results.json"), "w") as outfile:
        json.dump(profit_matching_dict, outfile, indent=4)
    
    return


def read_profit_and_matching_from_file(filename_prefix=""):
    current_dir = os.getcwd()
    if "dataset" not in os.listdir(current_dir):
        return None, None
        
    if "local_dataset" not in os.listdir(os.path.join(current_dir, "dataset")):
        return None, None
    
    local_data_directory = os.path.join(current_dir, "dataset", "local_dataset")
    with open(os.path.join(local_data_directory, filename_prefix + "matching_results.json")) as infile:
        profit_matching_dict = json.load(infile)
    # print(f"Inside function read_profit_and_matching_from_file(), {profit_matching_dict=}")
    profit = profit_matching_dict['profit']
    matching = {}
    if "matching" in profit_matching_dict:
        matching = profit_matching_dict['matching']

    return profit, matching


def write_local_system_user_data_to_file(local_system_data, local_user_data, filename_prefix=""):
   
    current_dir = os.getcwd()
    if "dataset" not in os.listdir(current_dir):
        os.mkdir(os.path.join(current_dir, "dataset"))
        
    if "local_dataset" not in os.listdir(os.path.join(current_dir, "dataset")):
        os.mkdir(os.path.join(current_dir, "dataset", "local_dataset"))
    
    local_data_directory = os.path.join(current_dir, "dataset", "local_dataset")
        
    with open(os.path.join(local_data_directory, filename_prefix + "local_system_data.json"), "w") as outfile:
            json.dump(local_system_data, outfile, indent=4)
    with open(os.path.join(local_data_directory, filename_prefix + "local_user_data.json"), "w") as outfile:
            json.dump(local_user_data, outfile, indent=4)
    
    return


def calculate_action_from_distance(step_size, x_dist, y_dist):
    # print(f"{x_dist=}, {y_dist=}")
    if (abs(x_dist) < (step_size/2)) and (abs(y_dist) < (step_size/2)):
        action_index = 0
    elif abs(x_dist) < (step_size/2):
            if y_dist > 0:
                action_index = 3
            else:
                action_index = 7
    elif x_dist > 0:
        if abs(y_dist) < (step_size/2):
            action_index = 1
        else:
            if y_dist > 0:
                action_index = 2
            else:
                action_index = 8
    else:
        if abs(y_dist) < (step_size/2):
            action_index = 5
        else:
            if y_dist > 0:
                action_index = 4
            else:
                action_index = 6
    
    return action_index


def xvc1(env, filename_prefix=""):
    # TO-DO:
        # add the part if num_ue_center == 0

    x_buffer = y_buffer = 0
    if not env.fraction_testing_mode:
        if env.which_algo_mode == "dts":
            x_buffer = random.choice([1.5, 1, -1, -1.5]) * (env.action_step_size)
            y_buffer = random.choice([1, -1]) * (env.action_step_size)
    current_dir = os.getcwd()
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "r") as infile:
        carrier_dict = json.load(infile)
    
    n_x, n_y = int(math.ceil(env.boundary_x / env.action_step_size)), \
        int(math.ceil(env.boundary_y / env.action_step_size))

    uav_2_ue_center_mapping = construct_uav_2_ue_center_mapping(env)
    # print(f"{uav_2_ue_center_mapping=}")
    # state_action_map = [[[0 for id_y in range(n_y+1)] for id_y in range(n_x+1)] for uav_idx in range(env.UAV_count)]
    state_action_map = []
    # state_action_map = [([[0]*(n_y+1)]*(n_x+1))]*(env.UAV_count)    # initializing

    # fill state action map here, according to env.index_to_action_mapper
    for uav_idx in range(env.UAV_count):
        grid = []
        corresp_ue_center_idx = uav_2_ue_center_mapping[uav_idx]
        # print(f"{env.UE_center[corresp_ue_center_idx]=}")
# env.index_to_action_mapper = [(0, 0), (1, 0), (diag, diag), (0, 1), (-diag, diag), (-1, 0), (-diag, -diag), (0, -1), (diag, -diag)]  # angle changes in clockwise fashion
        for xbin in range(n_x+1):
            x_mp = min((max(xbin-1, 0) * env.action_step_size) + (env.action_step_size/2), env.boundary_x)
            row = []
            for ybin in range(n_y+1):
                y_mp = min((max(ybin-1, 0) * env.action_step_size) + (env.action_step_size/2), env.boundary_y)
                # find angle now
                x_dist = max(1, min(env.UE_center[corresp_ue_center_idx][0], env.boundary_x - 1)) - x_mp
                y_dist = max(1, min(env.UE_center[corresp_ue_center_idx][1], env.boundary_y - 1)) - y_mp
                # if xbin == 0 and ybin == 0:
                #     print(f"{x_dist=}, {y_dist=}")
                # convert to action_index
                action_index = calculate_action_from_distance( \
                            env.action_step_size, \
                            x_dist + x_buffer, y_dist + y_buffer
                        )
                if (abs(x_dist) + abs(y_dist)) < (env.action_step_size * 4):    
                # if UAV is so close to target UE_center that any random movement 
                # will hinder the final coverage
                    rnd = False
                else:
                    rnd = np.random.choice([False, True], p=[0.7, 0.3])
                if rnd:
                    if action_index != 0:
                        action_index -= 1
                        action_index = np.random.choice([(action_index - 1 + (env.n_actions-1))%(env.n_actions-1), \
                                                    action_index, \
                                                    (action_index + 1 + (env.n_actions-1))%(env.n_actions-1)
                                                ], p=[0.45, 0.1, 0.45])

                        # action_index = np.random.choice([(action_index - 2 + (env.n_actions-1))%(env.n_actions-1), \
                        #                     (action_index - 1 + (env.n_actions-1))%(env.n_actions-1), \
                        #                     action_index, (action_index + 1)%(env.n_actions-1), \
                        #                     (action_index + 2)%(env.n_actions-1)], p=[0.1, 0.25, 0.3, 0.25, 0.1])
                        action_index += 1
            
                row.append(action_index)
                # state_action_map[uav_idx][xbin][ybin] = action_index
                # print(f"{uav_idx}")
                # print(f"{state_action_map[uav_idx][xbin][ybin]=}")
                    
            grid.append(row)
        state_action_map.append(grid)
        # print(f"{state_action_map=}")

    # print(f"{state_action_map[0]=}")
    env.calculated_action_matrix = state_action_map
    # print("inside xvc1, after storing calculated_action_matrix in env")
    # print(f"{env.calculated_action_matrix=}")
    return


def gsa(position, filename_prefix=""):
    '''
        position --> random or centered
    '''
    current_dir = os.getcwd()
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "r") as infile:
        carrier_dict = json.load(infile)
    carrier_dict["ue_placement_type"] = position
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "w") as outfile:
            json.dump(carrier_dict, outfile, indent=4)
    return


def read_data_dict_from_file():
    current_dir = os.getcwd()
    with open(os.path.join(current_dir, "spa_module", "data_dict.json"), "r") as infile:
        data_dict = json.load(infile)
    return data_dict


def cz3a(env, ret_tr=False, filename_prefix=""):
    current_dir = os.getcwd()
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "r") as infile:
        carrier_dict = json.load(infile)
    if (ret_tr or (("num_trained_epochs" in carrier_dict) and (carrier_dict["num_trained_epochs"] >= (env.UAV_count * 200)))):
        # calculate the current action here, and return it
        action_vector = []
        for uav_idx in range(env.UAV_count):
            action_grid = env.calculated_action_matrix[uav_idx]
            x_bin, y_bin = (int(env.current_state[uav_idx][0] / env.action_step_size), \
                            int(env.current_state[uav_idx][1] / env.action_step_size))
            action_vector.append(env.calculated_action_matrix[uav_idx][x_bin][y_bin])
        return action_vector
    else:
        return None


def write_num_epochs_to_env_params(env, episodes, filename_prefix=""):
    current_dir = os.getcwd()
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "r") as infile:
        carrier_dict = json.load(infile)
    carrier_dict["num_trained_epochs"] = episodes
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "w") as outfile:
            json.dump(carrier_dict, outfile, indent=4)
    return


def reward_in_reset_state(num_ues_covered, filename_prefix=""):
    current_dir = os.getcwd()
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "r") as infile:
        carrier_dict = json.load(infile)
    carrier_dict["reset_reward"] = (int(num_ues_covered) * 10)
    with open(os.path.join(current_dir, "environment_class", filename_prefix + "carrier.json"), "w") as outfile:
            json.dump(carrier_dict, outfile, indent=4)
    return