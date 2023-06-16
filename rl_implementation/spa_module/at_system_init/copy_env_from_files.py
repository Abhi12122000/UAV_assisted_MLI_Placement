import numpy as np

def helper_return_ue_pos_ls(user_data):
    ue_pos_ls = []
    for ue in user_data:
        if (ue not in user_data) or ('x' not in user_data[ue]):
            continue
        ue_pos_ls.append([user_data[ue]['x'], user_data[ue]['y']]) 

    return np.array(ue_pos_ls)


def helper_return_current_server_set_state(system_data, server_beg_idx, server_end_idx):
    server_pos_ls = []
    # print(f"{system_data=}")
    for server_idx in range(server_beg_idx, server_end_idx):
        if (server_idx >= len(system_data["server_info"])) or ('x' not in system_data["server_info"][server_idx]):
            continue
        x, y = system_data["server_info"][server_idx]['x'], system_data["server_info"][server_idx]['y']
        server_pos_ls.append([x, y])

    return server_pos_ls


def helper_return_ue_center_radius(system_data):
    return np.array(system_data["UE_centered_details"]["center"]), system_data["UE_centered_details"]["radius"]