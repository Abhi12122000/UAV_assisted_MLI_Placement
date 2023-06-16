import random
import numpy as np

def select_mli_for_ues(system_data, global_user_data, ue_beg_idx = 0, ue_end_idx_exclusive = None, select_mli_randomly=False):
    ue_count = system_data["system_component_counts"]["UE_count"]
    uav_count = system_data["system_component_counts"]["UAV_count"]
    fs_count = system_data["system_component_counts"]["FS_count"]
    num_mlis = system_data["system_component_counts"]["num_mlis"]

    if ue_end_idx_exclusive == None:
        ue_end_idx_exclusive = ue_count

    if not select_mli_randomly:
        # prob=[0.4, 0.4, 0.2]
        # num_mlis_for_hub = int(np.random.choice([(i+1) for i in range(min(3, num_mlis))], p=prob[:(min(3, num_mlis))]))
        
        num_mlis_for_hub = int(np.random.choice([(i+1) for i in range(num_mlis//3, num_mlis)]))
        mli_hubs = list(map(int, list(np.random.choice( \
                np.arange(num_mlis), \
                size=num_mlis_for_hub, \
                replace=False
                    ))))    #unique mli hubs here
        if not isinstance(mli_hubs, list):
            mli_hubs = [mli_hubs]
        other_mlis = [mli_idx for mli_idx in range(num_mlis) if (mli_idx not in mli_hubs)]
        # other_mlis = list(map(int, list(np.random.choice( \
        #         np.arange(num_mlis), \
        #         size=num_mlis_for_hub, \
        #         replace=False
        #             ))))
        # if not isinstance(other_mlis, list):
        #     other_mlis = [other_mlis]
        # print(f"{mli_hubs=}, {other_mlis=}")

    for ue in range(ue_beg_idx, ue_end_idx_exclusive):
        if str(ue) not in global_user_data:
            global_user_data[str(ue)] = {}
        if select_mli_randomly:
            curr_mli_choice = random.randint(0, num_mlis-1)
        else:
            from_mli_hub = np.random.choice([True, False], p=[0.7, 0.3])
            if from_mli_hub:
                curr_mli_choice = random.choice(mli_hubs)
            else:
                if other_mlis == []:
                    curr_mli_choice = random.choice(mli_hubs)
                else:
                    curr_mli_choice = random.choice(other_mlis)
            # print(f"inside select_mlis_at_ue_placement function, for centered mli selection, {curr_mli_choice=}")
        
        global_user_data[str(ue)]["mli_idx"] = curr_mli_choice
        
    return global_user_data