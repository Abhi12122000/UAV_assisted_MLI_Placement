import random

def select_mlis_for_system(data_dict, curr_system_data):

    #generating counts
    num_mlis = random.randint(data_dict["system_component_count_limits"]["num_mlis"]["mini"], data_dict["system_component_count_limits"]["num_mlis"]["maxi"])
    # print(f"{ue_count=}, {total_server_count=}, {uav_count=}, {fs_count=}, {num_mlis=}")
    curr_system_data["system_component_counts"]["num_mlis"] = num_mlis

    #generating mli info
    curr_system_data["mli_info"] = {
        "mli_details": [],
        "mli_transmission_latencies": []
    }

    curr_system_data["server_mli_computation_coeff"] = {
        "uav": {
            "trad_memory": [],
            "nvm_memory": []
        },
        "fs": {
            "trad_memory": [],
            "nvm_memory": []
        },
        "cloud": {
            "computation_time": []
        }
    }

    #setting mli details of system
    for mli_idx in range(num_mlis):
        possible_mlis = data_dict["mli_info"]["mli_details"]
        chosen_mli_idx = random.randint(0, len(possible_mlis)-1)
        #setting mli details
        chosen_mli_blueprint = possible_mlis[chosen_mli_idx]
        chosen_mli = {}
        for key in chosen_mli_blueprint:
            chosen_mli[key] = max(1, random.randint(int((3*chosen_mli_blueprint[key])/4), int((5*chosen_mli_blueprint[key])/4)))
        curr_system_data["mli_info"]["mli_details"].append(chosen_mli)

        #setting mli_transmission_latencies
        curr_system_data["mli_info"]["mli_transmission_latencies"].append(data_dict["mli_info"]["mli_transmission_latencies"][chosen_mli_idx])
        
        for server_type in data_dict["server_mli_computation_coeff"]:
            for memory_type in data_dict["server_mli_computation_coeff"][server_type]:
                computation_coeff_bp_value = data_dict["server_mli_computation_coeff"][server_type][memory_type][chosen_mli_idx]
                curr_system_data["server_mli_computation_coeff"][server_type][memory_type].append(
                    max(1, random.randint(int(computation_coeff_bp_value/2), int((3*computation_coeff_bp_value)/2)))
                )
    
    return curr_system_data


def update_mli_fs_transmission_latencies_in_system_data(system_data, divisor):
    
    # print(f"{divisor=}")
    num_mlis = system_data["system_component_counts"]["num_mlis"]
    for mli_idx in range(num_mlis):
        system_data["mli_info"]["mli_transmission_latencies"][mli_idx]["uav_fs_transmission_latency"] *= divisor
        # since divisor for bandwidth means multiplier for latency
    
    return system_data