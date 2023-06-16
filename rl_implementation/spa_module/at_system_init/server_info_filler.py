import random

def server_info_filler(env, curr_system_data, data_dict, scale_uav_memory_according_to_num_uavs=False, \
                variable_ue_count=False):
    #creating server info 
    curr_system_data["server_info"] = []

    uav_count = curr_system_data["system_component_counts"]["UAV_count"]
    fs_count = curr_system_data["system_component_counts"]["FS_count"]
    cloud = (curr_system_data["system_component_counts"]["total_server_count"]-1)
    ue_count = env.UE_count
    num_ue_centers = env.num_UE_centers
    ue_centered_ratio = env.UE_centered_ratio
    num_ues_per_server = int((ue_count * ue_centered_ratio) // num_ue_centers)
    if variable_ue_count:
        # print("fixing memory range")
        ue_var_count = 300
    else:
        ue_var_count = ue_count
    for server in range(curr_system_data["system_component_counts"]["total_server_count"]):
        receiving_power_bp = data_dict["power_coefficient_blueprint"]["receiving_power"]
        transmission_power_bp = data_dict["power_coefficient_blueprint"]["transmission_power"]
        computation_power_bp = data_dict["power_coefficient_blueprint"]["computation_power"]
        current_server_dict = {
            "server_idx": server,
            "receiving_power": max(1, \
                random.randint(int(receiving_power_bp/2), int(3*receiving_power_bp/2))),
            "transmission_power": max(1, \
                random.randint(int(transmission_power_bp/2), int(3*transmission_power_bp/2))),
            "computation_power": max(1, \
                random.randint(int(computation_power_bp/2), int(3*computation_power_bp/2))),
            "parent_server": None,
            "children_servers": None
        }
        if (server >= uav_count) and (server < cloud):
            #Fog server, set parent to Cloud
            current_server_dict["parent_server"] = cloud

        server_type = "cloud"   
        if server < uav_count:
            server_type = "uav"
        elif server < uav_count + fs_count:
            server_type = "fs"
        
        if server_type == "uav":
            if scale_uav_memory_according_to_num_uavs:
                uav_memory_divisor = 4
                fs_memory_divisor = 4
            else:
                uav_memory_divisor = 1
                fs_memory_divisor = 1
            trad_memory_limit = data_dict["server_memory_resource_limit_info"][server_type]["trad_memory_resource"]
            nvm_memory_limit = data_dict["server_memory_resource_limit_info"][server_type]["nvm_memory_resource"]
            if scale_uav_memory_according_to_num_uavs:
                trad_memory_limit_scaled = (trad_memory_limit * ue_var_count) // 5
                nvm_memory_limit_scaled = (nvm_memory_limit * ue_var_count) // 5
            else:
                trad_memory_limit_scaled = (trad_memory_limit * num_ues_per_server) // 5
                nvm_memory_limit_scaled = (nvm_memory_limit * num_ues_per_server) // 5
            current_server_dict["trad_memory_resource"] = (random.randint(int(trad_memory_limit_scaled/20), int(trad_memory_limit_scaled/15)) / uav_memory_divisor)
            current_server_dict["nvm_memory_resource"] = (random.randint(int(nvm_memory_limit_scaled/11), int(nvm_memory_limit_scaled/5)) / uav_memory_divisor)
        if server_type == "fs":
            trad_memory_limit = data_dict["server_memory_resource_limit_info"][server_type]["trad_memory_resource"]
            nvm_memory_limit = data_dict["server_memory_resource_limit_info"][server_type]["nvm_memory_resource"]
            if scale_uav_memory_according_to_num_uavs:
                trad_memory_limit_scaled = (trad_memory_limit * ue_var_count) // 5
                nvm_memory_limit_scaled = (nvm_memory_limit * ue_var_count) // 5
            else:
                trad_memory_limit_scaled = (trad_memory_limit * num_ues_per_server) // 5
                nvm_memory_limit_scaled = (nvm_memory_limit * num_ues_per_server) // 5
            if scale_uav_memory_according_to_num_uavs:
                current_server_dict["trad_memory_resource"] = random.randint(int(trad_memory_limit_scaled/14), int(trad_memory_limit_scaled/8)) / fs_memory_divisor
                current_server_dict["nvm_memory_resource"] = random.randint(int(nvm_memory_limit_scaled/9), int(nvm_memory_limit_scaled/5)) / fs_memory_divisor
            else:
                current_server_dict["trad_memory_resource"] = random.randint(int(trad_memory_limit_scaled/10), int(trad_memory_limit_scaled/5)) / fs_memory_divisor
                current_server_dict["nvm_memory_resource"] = random.randint(int(nvm_memory_limit_scaled/7), int(nvm_memory_limit_scaled/3)) / fs_memory_divisor
        curr_system_data["server_info"].append(current_server_dict)
    curr_system_data["cost_conversion_parameters"] = data_dict["cost_conversion_parameters"]
    
    return curr_system_data