import random

def update_power_energy_of_local_ues(local_system_data, local_user_data):
    
    ue_count = local_system_data["system_component_counts"]["UE_count"]
    uav_count = local_system_data["system_component_counts"]["UAV_count"]
    fs_count = local_system_data["system_component_counts"]["FS_count"]
    num_mlis = local_system_data["system_component_counts"]["num_mlis"]

    cloud = uav_count + fs_count
    for ue in range(ue_count):
        
        #calculating power
        local_user_data[str(ue)]["uav_receiving_power"] = local_system_data["server_info"][local_user_data[str(ue)]["parent_server"]]["receiving_power"]
        local_user_data[str(ue)]["uav_computation_power"] = local_system_data["server_info"][local_user_data[str(ue)]["parent_server"]]["computation_power"]
        local_user_data[str(ue)]["uav_transmission_power"] = local_system_data["server_info"][local_user_data[str(ue)]["parent_server"]]["transmission_power"]
        
        # print(f"parent fs index of {c} = {p}")
        local_user_data[str(ue)]["fs_receiving_power"] = local_system_data["server_info"][local_system_data["server_info"][local_user_data[str(ue)]["parent_server"]]["parent_server"]]["receiving_power"]
        local_user_data[str(ue)]["fs_computation_power"] = local_system_data["server_info"][local_system_data["server_info"][local_user_data[str(ue)]["parent_server"]]["parent_server"]]["computation_power"]
        local_user_data[str(ue)]["fs_transmission_power"] = local_system_data["server_info"][local_system_data["server_info"][local_user_data[str(ue)]["parent_server"]]["parent_server"]]["transmission_power"]
        
        local_user_data[str(ue)]["cloud_receiving_power"] = local_system_data["server_info"][cloud]["receiving_power"]
        local_user_data[str(ue)]["cloud_computation_power"] = local_system_data["server_info"][cloud]["computation_power"]
        
        
        #calculating energies
        local_user_data[str(ue)]["ue_uav_receiving_energy"] = (local_user_data[str(ue)]["ue_uav_transmission_latency"] * local_user_data[str(ue)]["uav_receiving_power"])
        local_user_data[str(ue)]["uav_computation_energy"] = None
        local_user_data[str(ue)]["uav_fs_transmission_energy"] = (local_user_data[str(ue)]["uav_fs_transmission_latency"] * local_user_data[str(ue)]["uav_transmission_power"])
        local_user_data[str(ue)]["uav_fs_receiving_energy"] = (local_user_data[str(ue)]["uav_fs_transmission_latency"] * local_user_data[str(ue)]["fs_receiving_power"])
        local_user_data[str(ue)]["fs_computation_energy"] = None
        local_user_data[str(ue)]["fs_cloud_transmission_energy"] = (local_user_data[str(ue)]["fs_cloud_transmission_latency"] * local_user_data[str(ue)]["fs_transmission_power"])
        local_user_data[str(ue)]["fs_cloud_receiving_energy"] = (local_user_data[str(ue)]["fs_cloud_transmission_latency"] * local_user_data[str(ue)]["cloud_receiving_power"])
        local_user_data[str(ue)]["cloud_computation_energy"] = (local_user_data[str(ue)]["cloud_computation_latency"] * local_user_data[str(ue)]["cloud_computation_power"])

    return local_user_data