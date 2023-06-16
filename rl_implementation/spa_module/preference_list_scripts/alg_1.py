# returns the 

def generate_student_preferences(system_data, user_data, fraction_bins=10):
    # print("inside generate_student_preferences")
    num_servers = system_data["system_component_counts"]["total_server_count"]
    num_uav = system_data["system_component_counts"]["UAV_count"]
    num_fs = system_data["system_component_counts"]["FS_count"]
    num_ues = system_data["system_component_counts"]["UE_count"]

    student_preferences = []
    system_preferences_data = [] # (ue, total_energy, mli_idx, server_idx, fraction * fraction_bins)
    for ue in range(num_ues):
        # calculate latencies (time delays) of ue on different servers and fractions
        ue_preferences = []  # (latency, server_idx, fraction * fraction_bins)
        uav = user_data[str(ue)]["parent_server"]
        fs = system_data["server_info"][uav]["parent_server"]
        cloud = num_servers - 1
        # print(f"{uav=}, {fs=}, {cloud=}")

        mli_idx = user_data[str(ue)]["mli_idx"]
        resource_required = system_data["mli_info"]["mli_details"][mli_idx]["R"]
        delay_requirement = system_data["mli_info"]["mli_details"][mli_idx]["p"]
        for server in [uav, fs]:  # arranging different fractions of uav and fog servers
            for x_i in range(0, fraction_bins+1):
                # resource constraint?
                required_trad_memory = (x_i/fraction_bins) * resource_required
                required_nvm_memory = resource_required - required_trad_memory
                # print(f"memory resource of {server=} is:-", system_data["server_info"][server]["trad_memory_resource"], system_data["server_info"][server]["nvm_memory_resource"])
                # print(f"{required_trad_memory=}, {required_nvm_memory=}")
                # print(system_data["server_info"][server])
                if (required_trad_memory > system_data["server_info"][server]["trad_memory_resource"]) or (required_nvm_memory > system_data["server_info"][server]["nvm_memory_resource"]):
                    continue
                
                # latency constraint?
                if server < num_uav:  # uav server
                    total_latency = (user_data[str(ue)]["ue_uav_transmission_latency"]) \
                                + (((user_data[str(ue)]["uav_trad_memory_linear_coeff"] * (x_i/fraction_bins)) \
                                + (user_data[str(ue)]["uav_nvm_memory_linear_coeff"] * (1-(x_i/fraction_bins)))))
                    
                    total_energy = (user_data[str(ue)]["ue_uav_transmission_latency"] * user_data[str(ue)]["uav_receiving_power"]) \
                                + (((user_data[str(ue)]["uav_trad_memory_linear_coeff"] * (x_i/fraction_bins)) \
                                + (user_data[str(ue)]["uav_nvm_memory_linear_coeff"] * (1-(x_i/fraction_bins)))) * user_data[str(ue)]["uav_computation_power"])
                    

                else:  # fog server
                    total_latency = user_data[str(ue)]["ue_uav_transmission_latency"] \
                                + user_data[str(ue)]["uav_fs_transmission_latency"] \
                                + (user_data[str(ue)]["fs_trad_memory_linear_coeff"] * (x_i/fraction_bins)) \
                                + (user_data[str(ue)]["fs_nvm_memory_linear_coeff"] * (1-(x_i/fraction_bins)))
                    
                    total_energy = (user_data[str(ue)]["ue_uav_transmission_latency"] * user_data[str(ue)]["uav_receiving_power"])\
                                + (user_data[str(ue)]["uav_fs_transmission_latency"] * (user_data[str(ue)]["uav_transmission_power"] + user_data[str(ue)]["fs_receiving_power"]))\
                                + (((user_data[str(ue)]["fs_trad_memory_linear_coeff"] * (x_i/fraction_bins)) \
                                + (user_data[str(ue)]["fs_nvm_memory_linear_coeff"] * (1-(x_i/fraction_bins)))) * user_data[str(ue)]["fs_computation_power"])
                
                # print(f"at {server=}, {x_i=}, {total_latency=}")
                if total_latency > delay_requirement:
                    continue
                ue_preferences.append((total_latency, server, x_i))
                system_preferences_data.append((ue, total_energy, mli_idx, server, x_i))
        
        # arranging cloud server
        # latency constraint?
        total_latency = user_data[str(ue)]["ue_uav_transmission_latency"]\
                    + user_data[str(ue)]["uav_fs_transmission_latency"]\
                    + user_data[str(ue)]["fs_cloud_transmission_latency"] \
                    + user_data[str(ue)]["cloud_computation_latency"]
        
        total_energy = (user_data[str(ue)]["ue_uav_transmission_latency"] * user_data[str(ue)]["uav_receiving_power"]) \
                    + (user_data[str(ue)]["uav_fs_transmission_latency"] * (user_data[str(ue)]["uav_transmission_power"] + user_data[str(ue)]["fs_receiving_power"])) \
                    + (user_data[str(ue)]["fs_cloud_transmission_latency"] * (user_data[str(ue)]["fs_transmission_power"] + user_data[str(ue)]["cloud_receiving_power"])) \
                    + (user_data[str(ue)]["cloud_computation_latency"] * user_data[str(ue)]["cloud_computation_power"])

        # print(f"at server={cloud}, x_i=fraction_bins, {total_latency=}")

        if total_latency <= delay_requirement:
            ue_preferences.append((total_latency, cloud, fraction_bins))
            system_preferences_data.append((ue, total_energy, mli_idx, cloud, fraction_bins))

        if len(ue_preferences) != 0:
            ue_preferences = sorted(ue_preferences)
        student_preferences.append(ue_preferences)
        
    system_preferences_dat = generate_mec_preferences(system_data=system_data, user_data=user_data, system_preferences_data=system_preferences_data)
    # system_preferences_dat --> (ue, profit, mli_idx, server_idx, fraction * fraction_bins, revenue, energy_cost)
    
    return student_preferences, system_preferences_dat



def generate_mec_preferences(system_data, user_data, system_preferences_data):

    phi = float(system_data["cost_conversion_parameters"]["phi"])
    # print(f"{phi=}")
    #generating profit values
    system_preferences_dat = []
    # system_preferences_dat --> (ue, profit, mli_idx, server_idx, fraction * fraction_bins, revenue, energy_cost)
    for data in system_preferences_data:
        total_energy = data[1]
        energy_consumption_cost = (total_energy * phi)
        mli_idx = data[2]
        mli_details = system_data["mli_info"]["mli_details"][mli_idx]
        R_by_p = (mli_details["R"] / mli_details["p"])
        revenue = (R_by_p * float(system_data["cost_conversion_parameters"]["a"]))
        dat = list(data)
        dat.extend([revenue, energy_consumption_cost])
        dat[1] = max(0, (revenue - energy_consumption_cost))
        system_preferences_dat.append(tuple(dat))

    return system_preferences_dat