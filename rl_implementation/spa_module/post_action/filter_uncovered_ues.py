import numpy as np
import copy

def helper_update_local_user_data(global_user_data, local_user_data, local_to_global_ue_index_mapper, num_local_ues):
    
    for local_ue_idx in range(num_local_ues):
        global_ue_idx = local_to_global_ue_index_mapper[local_ue_idx]
        local_user_data[str(local_ue_idx)] = copy.deepcopy(global_user_data[str(global_ue_idx)])
    
    return local_user_data


def helper_update_parent_server_of_ue(local_user_data, local_ue_idx_to_parent_uav_mapper, num_local_ues):
    
    for local_ue_idx in range(num_local_ues):
        local_user_data[str(local_ue_idx)]["parent_server"] = local_ue_idx_to_parent_uav_mapper[local_ue_idx]

    return local_user_data


def helper_update_parent_server_of_uavs(env, local_system_data, uav_count, fs_count):

    ground_UAV_state = env.current_state[:, :2]
    for uav in range(uav_count):
        horizontal_dist_FS_UAV = np.linalg.norm(env.FS_positions - ground_UAV_state[uav], axis = 1)
        parent_fs = int(np.argmin(horizontal_dist_FS_UAV) + uav_count)
        # print(parent_fs)
        local_system_data["server_info"][uav]["parent_server"] = parent_fs

    return local_system_data


def filter_uncovered_ues(env, system_data, global_user_data):
    ue_count = system_data["system_component_counts"]["UE_count"]
    uav_count = system_data["system_component_counts"]["UAV_count"]
    fs_count = system_data["system_component_counts"]["FS_count"]

    local_system_data = copy.deepcopy(system_data)
    local_user_data = {}

    ground_UAV_state = env.current_state[:, :2]
    horizontal_dist_UE_UAV = np.array([np.linalg.norm(env.UE_positions \
                                                    - ground_UAV_state[i], axis = 1) for i in range(env.UAV_count)])
    rho_matrix = (horizontal_dist_UE_UAV <= (env.C_max_t_array[:, None])) * 1  # binary association vector [(n_uav x n_ue) dimensions]
    Mn_t = ((rho_matrix.sum(axis = 0) >= 1) * 1)
    M_t = Mn_t.sum()  # total no. of UEs served by all the agent collectively
    
    local_system_data["system_component_counts"]["UE_count"] = int(M_t)
    
    global_to_local_ue_index_mapper = {}
    local_to_global_ue_index_mapper = {}
    local_ue_idx_to_parent_uav_mapper = {}
    
    local_ue_idx = 0
    for global_ue_idx in range(ue_count):
        for uav_idx in range(uav_count):
            if rho_matrix[uav_idx][global_ue_idx] != 0:
                local_ue_idx_to_parent_uav_mapper[local_ue_idx] = uav_idx
                global_to_local_ue_index_mapper[global_ue_idx] = local_ue_idx
                local_to_global_ue_index_mapper[local_ue_idx] = global_ue_idx
                local_ue_idx += 1
                break
    
    local_user_data = helper_update_local_user_data(global_user_data, local_user_data, \
                    local_to_global_ue_index_mapper, M_t)
    local_user_data = helper_update_parent_server_of_ue(local_user_data, \
                    local_ue_idx_to_parent_uav_mapper, M_t)
    local_system_data = helper_update_parent_server_of_uavs(env, local_system_data, \
                    uav_count, fs_count)
    
    return local_system_data, local_user_data, local_to_global_ue_index_mapper