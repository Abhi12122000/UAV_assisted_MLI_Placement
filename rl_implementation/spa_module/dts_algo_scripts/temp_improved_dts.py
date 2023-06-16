def calculate_profit_for_current_placement(system_data, user_data, ue, server, fraction, fraction_bins=10):
    '''
    Returns None if latency requirement of mli of `ue` cannot be satisfied using memory config. `fraction` of `server`.
    Else, returns the profit of the placement.
    '''
    
    num_servers = system_data["system_component_counts"]["total_server_count"]
    num_uav = system_data["system_component_counts"]["UAV_count"]
    num_fs = system_data["system_component_counts"]["FS_count"]
    num_ues = system_data["system_component_counts"]["UE_count"]
    
    mli_idx = user_data[str(ue)]["mli_idx"]
    resource_required = system_data["mli_info"]["mli_details"][mli_idx]["R"]
    delay_requirement = system_data["mli_info"]["mli_details"][mli_idx]["p"]
    
    if server < num_uav:  # uav server
        total_latency = (user_data[str(ue)]["ue_uav_transmission_latency"]) \
                    + (((user_data[str(ue)]["uav_trad_memory_linear_coeff"] * (fraction/fraction_bins)) \
                    + (user_data[str(ue)]["uav_nvm_memory_linear_coeff"] * (1-(fraction/fraction_bins)))))
        
        total_energy = (user_data[str(ue)]["ue_uav_transmission_latency"] * user_data[str(ue)]["uav_receiving_power"]) \
                    + (((user_data[str(ue)]["uav_trad_memory_linear_coeff"] * (fraction/fraction_bins)) \
                    + (user_data[str(ue)]["uav_nvm_memory_linear_coeff"] * (1-(fraction/fraction_bins)))) * user_data[str(ue)]["uav_computation_power"])
    
    elif server < (num_servers - 1):  # fog server
        total_latency = (user_data[str(ue)]["ue_uav_transmission_latency"]) \
                    + (user_data[str(ue)]["uav_fs_transmission_latency"]) \
                    + (((user_data[str(ue)]["fs_trad_memory_linear_coeff"] * (fraction/fraction_bins)) \
                    + (user_data[str(ue)]["fs_nvm_memory_linear_coeff"] * (1-(fraction/fraction_bins)))))
        
        total_energy = (user_data[str(ue)]["ue_uav_transmission_latency"] * user_data[str(ue)]["uav_receiving_power"]) \
                    + (user_data[str(ue)]["uav_fs_transmission_latency"] * (user_data[str(ue)]["uav_transmission_power"] + user_data[str(ue)]["fs_receiving_power"])) \
                    + (((user_data[str(ue)]["fs_trad_memory_linear_coeff"] * (fraction/fraction_bins)) \
                    + (user_data[str(ue)]["fs_nvm_memory_linear_coeff"] * (1-(fraction/fraction_bins)))) * user_data[str(ue)]["fs_computation_power"])
        
    else:  # cloud server
        total_latency = user_data[str(ue)]["ue_uav_transmission_latency"]\
                    + user_data[str(ue)]["uav_fs_transmission_latency"]\
                    + user_data[str(ue)]["fs_cloud_transmission_latency"] \
                    + user_data[str(ue)]["cloud_computation_latency"]
        
        total_energy = (user_data[str(ue)]["ue_uav_transmission_latency"] * user_data[str(ue)]["uav_receiving_power"]) \
                    + (user_data[str(ue)]["uav_fs_transmission_latency"] * (user_data[str(ue)]["uav_transmission_power"] + user_data[str(ue)]["fs_receiving_power"])) \
                    + (user_data[str(ue)]["fs_cloud_transmission_latency"] * (user_data[str(ue)]["fs_transmission_power"] + user_data[str(ue)]["cloud_receiving_power"])) \
                    + (user_data[str(ue)]["cloud_computation_latency"] * user_data[str(ue)]["cloud_computation_power"])
    
    # print(f"at {server=}, {fraction=}, {total_latency=}")

    # latency constraint?
    if total_latency > delay_requirement:
        return None
    
    phi = float(system_data["cost_conversion_parameters"]["phi"])
    # print(f"{phi=}")
    energy_consumption_cost = (total_energy * phi)
    mli_details = system_data["mli_info"]["mli_details"][mli_idx]
    R_by_p = (resource_required / delay_requirement)
    revenue = (R_by_p * float(system_data["cost_conversion_parameters"]["a"]))
    profit = max(0, (revenue - energy_consumption_cost))
    
    return profit


def improved_dts_algo(system_data, user_data):
    '''
    Implements the exact DTS algo to generate the matching 
    Arguments:
        system_data
        user_data
    Returns:
        matching[ue] --> (profit, mli_idx, server_idx, fraction * 10)
        mli_matching[server] --> [(mli)]
    '''

    ue_count = system_data["system_component_counts"]["UE_count"]
    uav_count = system_data["system_component_counts"]["UAV_count"]
    fs_count = system_data["system_component_counts"]["FS_count"]
    num_mlis = system_data["system_component_counts"]["num_mlis"]

    server_ue_ls = {}   # (key, value) --> (parent_server, ue_idx)
    for ue_idx in user_data:
        parent_server = user_data[ue_idx]["parent_server"]
        if parent_server not in server_ue_ls:
            server_ue_ls[parent_server] = []
        server_ue_ls[parent_server].append(int(ue_idx))
    
    fs_uav_ls = {}
    for ue_idx in user_data:
        ps = user_data[ue_idx]["parent_server"]
        pps = system_data["server_info"][ps]["parent_server"]
        if pps not in fs_uav_ls:
            fs_uav_ls[pps] = []
        if ps not in fs_uav_ls[pps]:
            fs_uav_ls[pps].append(ps)

    matching = {}
    # matching[ue] --> (profit, mli_idx, server_idx, fraction * 10)
    mli_matching = {}
    # mli_matching[server] --> [mli]

    cloud_server = (uav_count + fs_count)
    mli_matching[cloud_server] = []

    for fs_server in range(uav_count, fs_count+uav_count):
        if fs_server not in fs_uav_ls:
            continue
        mli_matching[fs_server] = []
        non_edge_better_set = []
        to_remove_ues = []
        for server in fs_uav_ls[fs_server]:  # Improved DTS algo considers 3-tier architecture, uavs, fog servers and cloud server
            if server not in server_ue_ls:
                continue
            mli_matching[server] = []
            nvm_better_set = []    # ((beta-alpha)/R_i, ue_idx)
            cloud_better_set = []    # ((beta-alpha)/R_i, ue_idx)
            for ue_idx_int in server_ue_ls[server]:
                mli_idx = user_data[str(ue_idx_int)]["mli_idx"]
                total_cloud_lat_coeff = (user_data[str(ue_idx_int)]["cloud_computation_latency"] + \
                                    user_data[str(ue_idx_int)]["uav_fs_transmission_latency"] + \
                                    user_data[str(ue_idx_int)]["ue_uav_transmission_latency"] + \
                                    user_data[str(ue_idx_int)]["fs_cloud_transmission_latency"])
                edge_nvm_lat_coeff = (user_data[str(ue_idx_int)]["uav_nvm_memory_linear_coeff"] + \
                                    user_data[str(ue_idx_int)]["ue_uav_transmission_latency"])
                edge_trad_lat_coeff = (user_data[str(ue_idx_int)]["uav_trad_memory_linear_coeff"] + \
                                    user_data[str(ue_idx_int)]["ue_uav_transmission_latency"])
                if edge_nvm_lat_coeff < total_cloud_lat_coeff:
                    nvm_better_set.append(((edge_nvm_lat_coeff - edge_trad_lat_coeff) / system_data["mli_info"]["mli_details"][mli_idx]["R"], ue_idx_int))
                else:
                    cloud_better_set.append(((total_cloud_lat_coeff - edge_trad_lat_coeff) / system_data["mli_info"]["mli_details"][mli_idx]["R"], ue_idx_int))

            nvm_better_set = sorted(nvm_better_set, key=lambda x: (-x[0]))
            # print("nvm: ", nvm_better_set)
            nvm_better_set = [ele[1] for ele in nvm_better_set]
            cloud_better_set = sorted(cloud_better_set, key=lambda x: (-x[0]))
            # print("cloud: ", cloud_better_set)
            cloud_better_set = [ele[1] for ele in cloud_better_set]
            remaining_trad_memory = system_data["server_info"][server]["trad_memory_resource"]
            remaining_nvm_memory = system_data["server_info"][server]["nvm_memory_resource"]

            to_remove_ues.clear()
            for ue_idx in nvm_better_set:
                mli_idx = user_data[str(ue_idx)]["mli_idx"]
                profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, server, fraction=10, fraction_bins=10)
                if profit == None:    # latency requirement cannot be met at this server
                    continue
                if mli_idx in mli_matching[server]:
                    matching[ue_idx] = (profit, mli_idx, server, 10)
                    to_remove_ues.append(ue_idx)
                    continue
                mli_R_value = system_data["mli_info"]["mli_details"][mli_idx]["R"]
                if mli_R_value > remaining_trad_memory:
                    continue

                to_remove_ues.append(ue_idx)
                remaining_trad_memory -= mli_R_value
                matching[ue_idx] = (profit, mli_idx, server, 10)
                if server not in mli_matching:
                    mli_matching[server] = []
                mli_matching[server].append(mli_idx)

            for ue in to_remove_ues:
                nvm_better_set.remove(ue)
            to_remove_ues.clear()

            if len(nvm_better_set) > 0:
                # some mlis from nvm_better_set and all mlis from cloud_better_set are remaining, and have to be placed 
                for ue_idx in nvm_better_set:
                    mli_idx = user_data[str(ue_idx)]["mli_idx"]
                    profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, server, fraction=0, fraction_bins=10)
                    if profit == None:    # latency requirement cannot be met at this server
                        continue
                    if mli_idx in mli_matching[server]:
                        matching[ue_idx] = (profit, mli_idx, server, 0)
                        to_remove_ues.append(ue_idx)
                        continue
                    mli_R_value = system_data["mli_info"]["mli_details"][mli_idx]["R"]
                    if mli_R_value > remaining_nvm_memory:
                        break

                    to_remove_ues.append(ue_idx)
                    remaining_nvm_memory -= mli_R_value
                    matching[ue_idx] = (profit, mli_idx, server, 0)
                    if server not in mli_matching:
                        mli_matching[server] = []
                    mli_matching[server].append(mli_idx)

                for ue in to_remove_ues:
                    nvm_better_set.remove(ue)

                # place remaining nvm_better_set and cloud_better_set mlis on cloud
                nvm_better_set.extend(cloud_better_set)
                non_edge_better_set.extend(nvm_better_set)
                continue   
            else:
                # all ues in nvm_better_set have been placed using traditional memory at edge
                # placing mlis of cloud_better_set on trad memory first, and the remaining on cloud
                to_remove_ues.clear()
                for ue_idx in cloud_better_set:
                    mli_idx = user_data[str(ue_idx)]["mli_idx"]
                    profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, server, fraction=10, fraction_bins=10)
                    if profit == None:    # latency requirement cannot be met at this server
                        continue
                    if mli_idx in mli_matching[server]:
                        matching[ue_idx] = (profit, mli_idx, server, 10)
                        to_remove_ues.append(ue_idx)
                        continue
                    mli_R_value = system_data["mli_info"]["mli_details"][mli_idx]["R"]
                    if mli_R_value > remaining_trad_memory:
                        break

                    to_remove_ues.append(ue_idx)
                    remaining_trad_memory -= mli_R_value
                    matching[ue_idx] = (profit, mli_idx, server, 10)
                    if server not in mli_matching:
                        mli_matching[server] = []
                    mli_matching[server].append(mli_idx)
                
                for ue in to_remove_ues:
                    cloud_better_set.remove(ue)
                to_remove_ues.clear()

                non_edge_better_set.extend(cloud_better_set)
                continue
                
        
        # Assumption: Trad. memory at fog is better than both NVM at fog, and the cloud server
        if len(non_edge_better_set) == 0:
            continue
        mli_matching[fs_server] = []
        fog_nvm_better_set = []    # ((beta-alpha)/R_i, ue_idx)
        cloud_better_set = []    # ((beta-alpha)/R_i, ue_idx)
        for ue_idx_int in non_edge_better_set:
            mli_idx = user_data[str(ue_idx_int)]["mli_idx"]
            total_cloud_lat_coeff = (user_data[str(ue_idx_int)]["cloud_computation_latency"] + \
                                user_data[str(ue_idx_int)]["uav_fs_transmission_latency"] + \
                                user_data[str(ue_idx_int)]["ue_uav_transmission_latency"] + \
                                user_data[str(ue_idx_int)]["fs_cloud_transmission_latency"])
            fog_nvm_lat_coeff = (user_data[str(ue_idx_int)]["fs_nvm_memory_linear_coeff"] + \
                                user_data[str(ue_idx_int)]["uav_fs_transmission_latency"] + \
                                user_data[str(ue_idx_int)]["ue_uav_transmission_latency"])
            fog_trad_lat_coeff = (user_data[str(ue_idx_int)]["fs_trad_memory_linear_coeff"] + \
                                user_data[str(ue_idx_int)]["uav_fs_transmission_latency"] + \
                                user_data[str(ue_idx_int)]["ue_uav_transmission_latency"])
            if fog_nvm_lat_coeff < total_cloud_lat_coeff:
                fog_nvm_better_set.append(((fog_nvm_lat_coeff - fog_trad_lat_coeff) / system_data["mli_info"]["mli_details"][mli_idx]["R"], ue_idx_int))
            else:
                cloud_better_set.append(((total_cloud_lat_coeff - fog_trad_lat_coeff) / system_data["mli_info"]["mli_details"][mli_idx]["R"], ue_idx_int))

        fog_nvm_better_set = sorted(fog_nvm_better_set, key=lambda x: (-x[0]))
        # print("nvm: ", fog_nvm_better_set)
        fog_nvm_better_set = [ele[1] for ele in fog_nvm_better_set]
        cloud_better_set = sorted(cloud_better_set, key=lambda x: (-x[0]))
        # print("cloud: ", cloud_better_set)
        cloud_better_set = [ele[1] for ele in cloud_better_set]
        remaining_trad_memory = system_data["server_info"][fs_server]["trad_memory_resource"]
        remaining_nvm_memory = system_data["server_info"][fs_server]["nvm_memory_resource"]

        to_remove_ues.clear()
        for ue_idx in fog_nvm_better_set:
            mli_idx = user_data[str(ue_idx)]["mli_idx"]
            profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, fs_server, fraction=10, fraction_bins=10)
            if profit == None:    # latency requirement cannot be met at this server
                continue
            if mli_idx in mli_matching[fs_server]:
                matching[ue_idx] = (profit, mli_idx, fs_server, 10)
                to_remove_ues.append(ue_idx)
                continue
            mli_R_value = system_data["mli_info"]["mli_details"][mli_idx]["R"]
            if mli_R_value > remaining_trad_memory:
                continue

            to_remove_ues.append(ue_idx)
            remaining_trad_memory -= mli_R_value
            matching[ue_idx] = (profit, mli_idx, fs_server, 10)
            if fs_server not in mli_matching:
                mli_matching[fs_server] = []
            mli_matching[fs_server].append(mli_idx)

        for ue in to_remove_ues:
            fog_nvm_better_set.remove(ue)
        to_remove_ues.clear()

        if len(fog_nvm_better_set) > 0:
            # some mlis from fog_nvm_better_set and all mlis from cloud_better_set are remaining, and have to be placed 
            for ue_idx in fog_nvm_better_set:
                mli_idx = user_data[str(ue_idx)]["mli_idx"]
                profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, fs_server, fraction=0, fraction_bins=10)
                if profit == None:    # latency requirement cannot be met at this server
                    continue
                if mli_idx in mli_matching[fs_server]:
                    matching[ue_idx] = (profit, mli_idx, fs_server, 0)
                    to_remove_ues.append(ue_idx)
                    continue
                mli_R_value = system_data["mli_info"]["mli_details"][mli_idx]["R"]
                if mli_R_value > remaining_nvm_memory:
                    break

                to_remove_ues.append(ue_idx)
                remaining_nvm_memory -= mli_R_value
                matching[ue_idx] = (profit, mli_idx, fs_server, 0)
                if fs_server not in mli_matching:
                    mli_matching[fs_server] = []
                mli_matching[fs_server].append(mli_idx)

            for ue in to_remove_ues:
                fog_nvm_better_set.remove(ue)

            # place remaining fog_nvm_better_set and cloud_better_set mlis on cloud
            for ue_idx in fog_nvm_better_set:
                mli_idx = user_data[str(ue_idx)]["mli_idx"]
                profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, cloud_server, fraction=10, fraction_bins=10)
                if profit == None:    # latency requirement cannot be met at this server
                    continue
                if mli_idx in mli_matching[cloud_server]:
                    matching[ue_idx] = (profit, mli_idx, cloud_server, 10)
                    to_remove_ues.append(ue_idx)
                    continue

                to_remove_ues.append(ue_idx)
                matching[ue_idx] = (profit, mli_idx, cloud_server, 10)
                if cloud_server not in mli_matching:
                    mli_matching[cloud_server] = []
                mli_matching[cloud_server].append(mli_idx)

        else:
            # all ues in fog_nvm_better_set have been placed using traditional memory at edge
            # placing mlis of cloud_better_set on trad memory first, and the remaining on cloud
            to_remove_ues.clear()
            for ue_idx in cloud_better_set:
                mli_idx = user_data[str(ue_idx)]["mli_idx"]
                profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, fs_server, fraction=10, fraction_bins=10)
                if profit == None:    # latency requirement cannot be met at this server
                    continue
                if mli_idx in mli_matching[fs_server]:
                    matching[ue_idx] = (profit, mli_idx, fs_server, 10)
                    to_remove_ues.append(ue_idx)
                    continue
                mli_R_value = system_data["mli_info"]["mli_details"][mli_idx]["R"]
                if mli_R_value > remaining_trad_memory:
                    break

                to_remove_ues.append(ue_idx)
                remaining_trad_memory -= mli_R_value
                matching[ue_idx] = (profit, mli_idx, fs_server, 10)
                if fs_server not in mli_matching:
                    mli_matching[fs_server] = []
                mli_matching[fs_server].append(mli_idx)
            
            for ue in to_remove_ues:
                cloud_better_set.remove(ue)
            to_remove_ues.clear()

            # place remaining mlis on cloud
            for ue_idx in cloud_better_set:
                mli_idx = user_data[str(ue_idx)]["mli_idx"]
                profit = calculate_profit_for_current_placement(system_data, user_data, ue_idx, cloud_server, fraction=10, fraction_bins=10)
                if profit == None:    # latency requirement cannot be met at this server
                    continue
                matching[ue_idx] = (profit, mli_idx, cloud_server, 10)
                if cloud_server not in mli_matching:
                    mli_matching[cloud_server] = []
                if mli_idx not in mli_matching[cloud_server]:
                    mli_matching[cloud_server].append(mli_idx)
        
    return matching, mli_matching


def calculate_profit_from_matching(matching_final):
    '''
    Calculates the total_system_profit from the given matching
    Arguments:
        matching_final[ue] --> (profit, mli_idx, server_idx, fraction * 10)

    Returns:
        total_system_profit
    '''
    total_system_profit = 0 
    for (ue, ue_matched_data) in matching_final.items():
        # print("printing data: ", ue_matched_data)
        total_system_profit += ue_matched_data[0]
    
    return total_system_profit


def main(system_data, user_data, fraction_bins=10):
    matching, mli_matching = improved_dts_algo(system_data, user_data)
    total_system_profit = calculate_profit_from_matching(matching)
    # print(f"{matching=}, {total_system_profit=}")
    return total_system_profit, matching
