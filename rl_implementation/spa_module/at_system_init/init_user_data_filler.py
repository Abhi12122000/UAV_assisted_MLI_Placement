import random

def initialize_user_data(global_user_data, system_data, beg_idx=0, end_idx=None):
    ue_count = system_data["system_component_counts"]["UE_count"]
    num_mlis = system_data["system_component_counts"]["num_mlis"]
    if end_idx is None:
        end_idx = ue_count
    for ue in range(beg_idx, end_idx):
        curr_user_data = global_user_data[str(ue)]
        # print(f"{curr_user_data=}")
        if "parent_server" not in curr_user_data:
            curr_user_data["parent_server"] = None
        
        #calculating computation latency coefficients
        curr_user_data["uav_trad_memory_linear_coeff"] = system_data["server_mli_computation_coeff"]["uav"]["trad_memory"][curr_user_data["mli_idx"]]
        curr_user_data["uav_nvm_memory_linear_coeff"] = system_data["server_mli_computation_coeff"]["uav"]["nvm_memory"][curr_user_data["mli_idx"]]
        curr_user_data["fs_trad_memory_linear_coeff"] = system_data["server_mli_computation_coeff"]["fs"]["trad_memory"][curr_user_data["mli_idx"]]
        curr_user_data["fs_nvm_memory_linear_coeff"] = system_data["server_mli_computation_coeff"]["fs"]["nvm_memory"][curr_user_data["mli_idx"]]
        curr_user_data["cloud_computation_latency"] = system_data["server_mli_computation_coeff"]["cloud"]["computation_time"][curr_user_data["mli_idx"]]
        
        #calculating latencies
        curr_user_data["ue_uav_transmission_latency"] = system_data["mli_info"]["mli_transmission_latencies"][curr_user_data["mli_idx"]]["ue_uav_transmission_latency"]
        curr_user_data["uav_fs_transmission_latency"] = system_data["mli_info"]["mli_transmission_latencies"][curr_user_data["mli_idx"]]["uav_fs_transmission_latency"]
        curr_user_data["fs_cloud_transmission_latency"] = system_data["mli_info"]["mli_transmission_latencies"][curr_user_data["mli_idx"]]["fs_cloud_transmission_latency"]
        
    return global_user_data