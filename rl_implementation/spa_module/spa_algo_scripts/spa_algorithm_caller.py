# computes profit from the 2 SPA algorithms
import numpy as np
import json
from spa_module.preference_list_scripts import alg_1
import os

##HELPER FUNCTION
def recalculate_ue_set(ue_preferences, ue_set):
    # popping the last preferred RB from UE set
    to_remove_ues = []

    # popping UEs whose preference lists have become empty
    for ue in ue_set:
        if len(ue_preferences[ue]) == 0:
            to_remove_ues.append(ue)
        else:
            del ue_preferences[ue][0]
            if len(ue_preferences[ue]) == 0:
                to_remove_ues.append(ue)

    for remove_ue in to_remove_ues:
        if remove_ue not in ue_set:
            continue
        else:
            ue_set.remove(remove_ue)

    return ue_set


##HELPER FUNCTION
def calculate_mlis_total_memory_required(system_data, user_data, RB_dict, server_idx, fraction_bins=10):
    trad_comp = 0
    nvm_comp = 0
    # mli_info_placed_on_server --> [(ue_set)]
    for rb in RB_dict:
        if rb[1] == server_idx:
            resource_requirement = system_data["mli_info"]["mli_details"][rb[0]]["R"]
            trad_comp += (resource_requirement * (rb[2]/fraction_bins))
            nvm_comp += (resource_requirement * (1 - (rb[2]/fraction_bins)))

    return (trad_comp, nvm_comp)


##HELPER FUNCTION
def compute_server_wise_mec_preferences(system_data, RB_dict, mec_preferences_data, fraction_bins=10):
    # mec_preferences_data --> (ue, profit, mli_idx, server_idx, fraction * fraction_bins, revenue, energy_cost)
    # RB_dict --> (key, value) <==> ((q,s,z,f), [(ue, profit)])

    server_wise_preference_ls = [[] for _ in range(system_data["system_component_counts"]["total_server_count"])]
    # server_wise_preference_ls[server_idx] --> (profit, mli_idx, fraction * fraction_bins)
    
    # compute server_wise_preference_ls
    for rb in RB_dict:
        # print(f"{rb=}")
        total_profit = 0
        ue_list = []
        for ls in RB_dict[rb]:
            total_profit += ls[1]
            ue_list.append(ls[0])
        server_wise_preference_ls[rb[1]].append((total_profit, rb[0], rb[2]))
    
    server_wise_preference_ls = [sorted(server_wise_preference_ls[i], reverse=True, key=lambda x: x[0]) for i in range(len(server_wise_preference_ls))]
    return server_wise_preference_ls


def spa_algorithm(system_data, user_data, ue_preferences, mec_preferences_data, fraction_bins=10):
    '''
    Runs the SPA Algorithm and returns the profit
    Arguments:
        ue_preferences[ue] --> [(latency, server_idx, fraction * fraction_bins)]
        mec_preferences_data --> (ue, profit, mli_idx, server_idx, fraction * fraction_bins, revenue, energy_cost)
    Returns:
        RB_dict
        matching
    '''

    RB_to_profit_mapper = {}  
    # RB_to_profit_mapper[(ue, mli, server, fraction * fraction_bins)] --> profit
    
    # filling RB_to_profit_mapper
    for data in mec_preferences_data:
        RB_to_profit_mapper[(data[0], data[2], data[3], data[4])] = data[1]


    ue_set = []
    matching = {}
    # matching[ue] --> (profit, mli_idx, server_idx, fraction * fraction_bins)
    RB_dict = {}    # to be used in mec preference calculation 
    # initialize a dictionary with (key:value) pair as ((q,s,z,f), [(ue, profit)])

    # calculates initial user set U 
    for ue, ue_ls in enumerate(ue_preferences):
        if len(ue_ls) > 0:
            ue_set.append(ue)

    
    while len(ue_set) > 0:

        # ues proposing servers
        for ue in ue_set:
            user_tuple = (ue, user_data[str(ue)]["mli_idx"], ue_preferences[ue][0][1], ue_preferences[ue][0][2])
            current_profit = RB_to_profit_mapper[user_tuple]
            matching[ue] = (current_profit, user_data[str(ue)]["mli_idx"], ue_preferences[ue][0][1], ue_preferences[ue][0][2])
            rb = (user_data[str(ue)]["mli_idx"], ue_preferences[ue][0][1], ue_preferences[ue][0][2])
            if rb not in RB_dict:
                RB_dict[rb]=[]
            RB_dict[rb].append((ue, current_profit))
            del ue_preferences[ue][0]

        # servers accepting/rejecting proposals

        server_wise_preference_ls = compute_server_wise_mec_preferences(system_data, RB_dict, mec_preferences_data, fraction_bins)
        # server_wise_preference_ls[server_idx] --> [(profit, mli_idx, fraction * fraction_bins)]
        removed_ues = []
        for server in range(system_data["system_component_counts"]["total_server_count"] - 1):    # excluding cloud server
            # print(f"{server=}")
            # print(f"{server=}, {server_wise_preference_ls[server]=}")
            trad_comp, nvm_comp = calculate_mlis_total_memory_required(system_data, user_data, RB_dict, server, fraction_bins)
            while (trad_comp > system_data["server_info"][server]["trad_memory_resource"]) or (nvm_comp > system_data["server_info"][server]["nvm_memory_resource"]):
                # release the least preferred RB, remove from matching, add those UEs in removed_ues, clear RB_dict[rb]
                while len(server_wise_preference_ls[server]) > 0:
                    least_preferred_rb = server_wise_preference_ls[server][-1]
                    rb_released = (least_preferred_rb[1], server, least_preferred_rb[2])
                    server_wise_preference_ls[server].pop()
                    if rb_released in RB_dict:
                        break
                for ue, profit in RB_dict[rb_released]:
                    del matching[ue]
                    removed_ues.append(ue)
                # print(f"{rb_released=}")
                del RB_dict[rb_released]
                # subtract this RB's resource requirements from trad_comp and nvm_comp
                trad_comp -= (system_data["mli_info"]["mli_details"][rb_released[0]]["R"] * (rb_released[2]/fraction_bins))
                nvm_comp -= (system_data["mli_info"]["mli_details"][rb_released[0]]["R"] * (1 - (rb_released[2]/fraction_bins)))
            # end of while

        #cleanup code, updates the ue_set after current removals made by mec system
        ue_set = removed_ues
        ue_set = recalculate_ue_set(ue_preferences, ue_set)
    # end of while

    return RB_dict, matching, RB_to_profit_mapper


def place_remaining_ues(system_data, user_data, RB_dict, matching, RB_to_profit_mapper, fraction_bins=10):
    '''
    Calculates the final matching from the given matching
    Arguments:
        RB_dict --> initialize a dictionary with (key:value) pair as ((q,s,z,f), [(ue, profit)])     (generated from the 1st algo)
        matching[ue] --> (profit, mli_idx, server_idx, fraction * fraction_bins)     (generated from the 1st algo)
        RB_to_profit_mapper[(ue, mli, server, fraction * fraction_bins)] --> profit

    Returns:
        matching_final --> final matching representing the MLI Placement
        RB_dict_final --> final RB_dict 
    '''
    matching_final = {}
    RB_dict_final = {}

    num_ues = system_data["system_component_counts"]["UE_count"]
    for ue in range(num_ues):
        if ue in matching:
            matching_final[ue] = matching[ue]
            profit, mli_idx, server, fraction = matching_final[ue]
            if (mli_idx, server, fraction) not in RB_dict_final:
                RB_dict_final[(mli_idx, server, fraction)]=[]
            RB_dict_final[(mli_idx, server, fraction)].append((ue, profit))

        else:
            curr_profit = 0
            curr_best_RB = None
            uav = user_data[str(ue)]["parent_server"]
            fs = system_data["server_info"][uav]["parent_server"]
            cloud = system_data["system_component_counts"]["total_server_count"]-1
            mli_idx = user_data[str(ue)]["mli_idx"]
            for server in [uav, fs]:
                for fraction in range(0, fraction_bins+1):
                    if ((mli_idx, server, fraction) in RB_dict) and (RB_to_profit_mapper[(ue, mli_idx, server, fraction)] > curr_profit):
                        curr_profit = RB_to_profit_mapper[(ue, mli_idx, server, fraction)]
                        curr_best_RB = (mli_idx, server, fraction)

            if ((ue, mli_idx, cloud, fraction_bins) in RB_to_profit_mapper) \
                and (RB_to_profit_mapper[(ue, mli_idx, cloud, fraction_bins)] > curr_profit):
                curr_profit = RB_to_profit_mapper[(ue, mli_idx, cloud, fraction_bins)]
                curr_best_RB = (mli_idx, cloud, fraction_bins)

            if curr_profit>0:
                # include this UE in matching, and in RB_dict
                matching_final[ue] = (curr_profit, curr_best_RB[0], curr_best_RB[1], curr_best_RB[2])
                current_RB = (curr_best_RB[0], curr_best_RB[1], curr_best_RB[2])
                if current_RB not in RB_dict_final:
                    RB_dict_final[current_RB] = []
                RB_dict_final[current_RB].append((ue, curr_profit))
            
    return matching_final, RB_dict_final


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
    

def compute_revenue_energy_cost(matching_final, mec_preferences_data):
    '''
    Calculates total revenue and total energy cost of the final matching
    Arguments:
        matching_final[ue] --> (profit, mli_idx, server_idx, fraction * 10)
        mec_preferences_data --> (ue, profit, mli_idx, server_idx, fraction * fraction_bins, revenue, energy_cost)
    
    Returns:
        total_revenue
        total_energy_cost
    '''

    total_revenue = 0
    total_energy_cost = 0
    for data in mec_preferences_data:
        ue, mli_idx, server_idx, fraction = data[0], data[2], data[3], data[4]
        if ue in matching_final:
            matching_mli_idx, matching_server_idx, matching_fraction = matching_final[ue][1], matching_final[ue][2], matching_final[ue][3]
            if (matching_mli_idx == mli_idx) and (matching_server_idx == server_idx) and (matching_fraction == fraction):
                total_revenue += data[5]
                total_energy_cost += data[6]
    
    return total_revenue, total_energy_cost


def main(system_data, user_data, fraction_bins=10):
    # student_preferences format: (latency, server, fraction * fraction_bins)
    # mec_preferences_data --> (ue, profit, mli_idx, server_idx, fraction * fraction_bins, revenue, energy_cost)

    ue_preferences, mec_preferences_data = alg_1.generate_student_preferences(system_data=system_data, user_data=user_data, fraction_bins=fraction_bins)
    # print(f"{ue_preferences=}")
    # print(f"{mec_preferences_data=}")
    # mec_preferences = alg_1.generate_mec_preferences(system_data=system_data, user_data=user_data)
    # spa_algorithm(system_data=system_data, user_data=user_data, student_preferences=student_preferences, mec_preferences=mec_preferences)
    # print(f"{ue_preferences=}")
    # print(f"{mec_preferences_data=}")
    # print(f"{fraction_bins=}")
    RB_dict, matching, RB_to_profit_mapper = spa_algorithm(system_data=system_data, user_data=user_data, ue_preferences=ue_preferences, mec_preferences_data=mec_preferences_data, fraction_bins=fraction_bins)
    matching_final, RB_dict_final = place_remaining_ues(system_data=system_data, user_data=user_data, RB_dict=RB_dict, matching=matching, RB_to_profit_mapper=RB_to_profit_mapper, fraction_bins=fraction_bins)
    total_system_profit = calculate_profit_from_matching(matching_final=matching_final)
    # print(f"{total_system_profit=}")
    # print(f"{matching_final=}")
    total_revenue, total_energy_cost = compute_revenue_energy_cost(matching_final, mec_preferences_data)
    
    return total_system_profit, matching_final, total_revenue, total_energy_cost
