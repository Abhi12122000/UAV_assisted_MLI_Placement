def system_comp_counts_filler(env, curr_system_data, new_component_counts=None):
    if new_component_counts is not None:
        for key in new_component_counts:
            curr_system_data["system_component_counts"][key] = new_component_counts[key]
    else:
        curr_system_data["system_component_counts"] = {
            "UE_count": env.UE_count,
            "total_server_count": (env.UAV_count + env.FS_count + 1),
            "UAV_count": env.UAV_count,
            "FS_count": env.FS_count
        }
    curr_system_data["system_component_counts"]["total_server_count"] = \
        (curr_system_data["system_component_counts"]["UAV_count"] + curr_system_data["system_component_counts"]["FS_count"] + 1)
            
    return curr_system_data
