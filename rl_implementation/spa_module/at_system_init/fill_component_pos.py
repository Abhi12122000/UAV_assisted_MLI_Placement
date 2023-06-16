def populate_user_data_with_ue_positions(user_data, ue_pos_ls, beg_ue_idx = 0, end_ue_idx = None):
    
    if end_ue_idx == None:
        end_ue_idx = ue_pos_ls.shape[0]
    
    for ue_idx_int in range(beg_ue_idx, end_ue_idx):
        if str(ue_idx_int) not in user_data:
            user_data[str(ue_idx_int)] = {}
        user_data[str(ue_idx_int)]["x"] = float(ue_pos_ls[ue_idx_int][0])
        user_data[str(ue_idx_int)]["y"] = float(ue_pos_ls[ue_idx_int][1])
    
    return user_data


def populate_system_data_with_server_pos(system_data, server_beg_idx, server_end_idx, server_pos_ls):
    
    for offset in range(server_end_idx-server_beg_idx):
        system_data["server_info"][offset+server_beg_idx]['x'] = float(server_pos_ls[offset][0])
        system_data["server_info"][offset+server_beg_idx]['y'] = float(server_pos_ls[offset][1])

    return system_data


def store_ue_center_and_radius(system_data, ue_center_ls, ue_radius_ls):
    system_data["UE_centered_details"] = {}
    system_data["UE_centered_details"]["center"] = [[float(c[0]), float(c[1])] for c in ue_center_ls]    # making it a list because numpy values cannot be serialized in JSON
    system_data["UE_centered_details"]["radius"] = [float(r) for r in ue_radius_ls]    # making it a list because numpy values cannot be serialized in JSON

    return system_data