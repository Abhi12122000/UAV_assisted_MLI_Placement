import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import os
import json
import random

def apply_gurobi(data_folder=None, algo_type="spa"):
    
    if data_folder is None:
        data_folder = os.path.join(os.getcwd(), "dataset", "local_dataset")
    system_file = algo_type + "local_system_data.json"
    user_file = algo_type + "local_user_data.json"
    with open(os.path.join(data_folder, system_file),"r") as system_file_obj:
        system_data = json.load(system_file_obj)
    with open(os.path.join(data_folder, user_file),"r") as user_file_obj:
        user_data = json.load(user_file_obj)

    ue_count = system_data["system_component_counts"]["UE_count"]
    uav_count = system_data["system_component_counts"]["UAV_count"]
    fs_count = system_data["system_component_counts"]["FS_count"] 
    num_mlis = system_data["system_component_counts"]["num_mlis"]
    try:
        # Create a new model
        m = gp.Model("quadratic")

        # Create variables
        ohm = m.addMVar(shape=ue_count, vtype=GRB.BINARY, name="ohm")
        alpha = m.addMVar(shape=ue_count, vtype=GRB.BINARY, name="alpha")
        beta = m.addMVar(shape=ue_count, vtype=GRB.BINARY, name="beta")
        oa = m.addMVar(shape=ue_count, vtype=GRB.BINARY, name="o_into_a")
        ob = m.addMVar(shape=ue_count, vtype=GRB.BINARY, name="o_into_b")
        oab = m.addMVar(shape=ue_count, vtype=GRB.BINARY, name="o_into_a_into_b")
        xi = m.addMVar(shape=ue_count, lb = np.zeros(ue_count, dtype=np.float64), \
                    ub = np.ones(ue_count, dtype=np.float64), name="xi")
        yi = m.addMVar(shape=ue_count, lb = np.zeros(ue_count, dtype=np.float64), \
                    ub = np.ones(ue_count, dtype=np.float64), name="yi")
        ue_uav_trans_lat = np.array([user_data[str(i)]["ue_uav_transmission_latency"] for i in range(ue_count)])
        uav_comp_trad_lat = np.array([user_data[str(i)]["uav_trad_memory_linear_coeff"] for i in range(ue_count)])
        uav_comp_nvm_lat = np.array([user_data[str(i)]["uav_nvm_memory_linear_coeff"] for i in range(ue_count)])
        uav_fs_trans_lat = np.array([user_data[str(i)]["uav_fs_transmission_latency"] for i in range(ue_count)])
        fs_comp_trad_lat = np.array([user_data[str(i)]["fs_trad_memory_linear_coeff"] for i in range(ue_count)])
        fs_comp_nvm_lat = np.array([user_data[str(i)]["fs_nvm_memory_linear_coeff"] for i in range(ue_count)])
        fs_cloud_trans_lat = np.array([user_data[str(i)]["fs_cloud_transmission_latency"] for i in range(ue_count)])
        cloud_comp_lat = np.array([user_data[str(i)]["cloud_computation_latency"] for i in range(ue_count)])
        
        uav_receiving_power = np.array([user_data[str(i)]["uav_receiving_power"] for i in range(ue_count)])
        uav_computation_power = np.array([user_data[str(i)]["uav_computation_power"] for i in range(ue_count)])
        uav_transmission_power = np.array([user_data[str(i)]["uav_transmission_power"] for i in range(ue_count)])
        fs_receiving_power = np.array([user_data[str(i)]["fs_receiving_power"] for i in range(ue_count)])
        fs_computation_power = np.array([user_data[str(i)]["fs_computation_power"] for i in range(ue_count)])
        fs_transmission_power = np.array([user_data[str(i)]["fs_transmission_power"] for i in range(ue_count)])
        cloud_receiving_power = np.array([user_data[str(i)]["cloud_receiving_power"] for i in range(ue_count)])
        cloud_computation_power = np.array([user_data[str(i)]["cloud_computation_power"] for i in range(ue_count)])
        R_array = np.array([system_data["mli_info"]["mli_details"][user_data[str(i)]["mli_idx"]]["R"] for i in range(ue_count)])
        p_array = np.array([system_data["mli_info"]["mli_details"][user_data[str(i)]["mli_idx"]]["p"] for i in range(ue_count)])
        a = system_data["cost_conversion_parameters"]["a"]
        phi = system_data["cost_conversion_parameters"]["phi"] 
        revenue_coeff_array = a * (R_array / p_array)
        revenue = ohm * revenue_coeff_array
        service_latency = (ohm * ue_uav_trans_lat) \
                + (oa * ((uav_comp_trad_lat * xi) + (uav_comp_nvm_lat * (1 - xi)))) \
                + (ohm * (1 - alpha) * uav_fs_trans_lat) \
                + ((ob - oab) * ((fs_comp_trad_lat * yi) + (fs_comp_nvm_lat * (1 - yi)))) \
                + ((ohm - oa - ob + oab) * (fs_cloud_trans_lat + cloud_comp_lat))
        service_energy = (ohm * ue_uav_trans_lat * uav_receiving_power)\
                + (oa * ((uav_comp_trad_lat * xi) + (uav_comp_nvm_lat * (1 - xi))) * uav_computation_power) \
                + (ohm * (1 - alpha) * uav_fs_trans_lat * (uav_transmission_power + fs_receiving_power)) \
                + ((ob - oab) * ((fs_comp_trad_lat * yi) + (fs_comp_nvm_lat * (1 - yi))) * fs_computation_power) \
                + ((ohm - oa - ob + oab) * (fs_cloud_trans_lat * (fs_transmission_power + cloud_receiving_power) + cloud_comp_lat * cloud_computation_power))
        
        energy_cost = phi * service_energy
        #Objective
        m.setObjective(sum(revenue - energy_cost), GRB.MAXIMIZE)

        # OLD CODE:
        # # Add constraints
        # # m.addConstr(xi <= 1.0, name="c1")
        # # m.addConstr(xi >= 0.0, name="c2")
        # # m.addConstr(yi <= 1.0, name="c3")
        # # m.addConstr(yi >= 0.0, name="c4")
        # END OF OLD CODE
        m.addConstr(service_latency <= p_array, name="c5")
        curr = 6

        # # TEMP CODE
        # m.addConstr(ohm - alpha == 0., name="c"+str(curr))
        # curr += 1
        # # END OF TEMP CODE

        # alpha + beta less than equal to 1.
        # # How to give this in matrix form (i.e., without looping)?
        # for ue in range(ue_count):
        #     m.addConstr(alpha[ue] + beta[ue] <= 1.0, name="c"+str(curr))
        #     curr += 1

        m.addConstr(oa - (ohm * alpha) == 0.0, name="c"+str(curr))
        m.addConstr(ob - (ohm * beta) == 0.0, name="c"+str(curr+1))
        m.addConstr(oab - (oa * beta) == 0.0, name="c"+str(curr+2))
        # m.addConstr(oa - xi >= 0.0, name="c"+str(curr))
        # m.addConstr(ob - oab - yi >= 0.0, name="c"+str(curr))

        curr += 3
        # OLD CODE:
        # # for ue in range(ue_count):
        # #     m.addConstr(oa[ue] - (ohm[ue] * alpha[ue]) == 0.0, name="c"+str(curr))
        # #     m.addConstr(ob[ue] - (ohm[ue] * beta[ue]) == 0.0, name="c"+str(curr+1))
        # #     m.addConstr(oab[ue] - (oa[ue] * beta[ue]) == 0.0, name="c"+str(curr+2))
        # #     curr += 3
        # END OF OLD CODE

        uav_server_trad_mem = np.array([system_data["server_info"][i]["trad_memory_resource"] for i in range(uav_count)])
        uav_server_nvm_mem = np.array([system_data["server_info"][i]["nvm_memory_resource"] for i in range(uav_count)])
        fs_server_trad_mem = np.array([system_data["server_info"][i]["trad_memory_resource"] for i in range(uav_count, uav_count + fs_count)])
        fs_server_nvm_mem = np.array([system_data["server_info"][i]["nvm_memory_resource"] for i in range(uav_count, uav_count + fs_count)])
        
        M = 100000 # smallest possible given bounds on x and y
        eps = 0.01
        uav_children = [[] for uav in range(uav_count)]
        for ue in range(ue_count):
            ps = user_data[str(ue)]["parent_server"]
            uav_children[ps].append(ue)
        barr = []
        for uav in range(uav_count):
            # print(f"for {uav=}")
            children = uav_children[uav]
            mli_dict = {}
            for ue in children:
                curr_mli = user_data[str(ue)]["mli_idx"]
                if curr_mli not in mli_dict:
                    mli_dict[curr_mli] = []
                mli_dict[curr_mli].append(ue)
            b = m.addMVar(shape=len(mli_dict), vtype=GRB.BINARY, name="b"+str(uav))
            barr.append(b)
            this_constr_trad_mem_ls = []
            this_constr_nvm_mem_ls = []
            extra_constraints = []
            for i, mli in enumerate(mli_dict):
                mli_ue_ls = mli_dict[mli]
                for idx in range(1, len(mli_ue_ls)):
                    m.addConstr(oa[mli_ue_ls[idx]] == oa[mli_ue_ls[idx-1]])
                    m.addConstr(alpha[mli_ue_ls[idx]] == alpha[mli_ue_ls[idx-1]])
                    m.addConstr(ohm[mli_ue_ls[idx]] == ohm[mli_ue_ls[idx-1]])
                    m.addConstr(xi[mli_ue_ls[idx]] == xi[mli_ue_ls[idx-1]])
                    
                # print(f"children having {mli=} are:- {mli_ue_ls=}")
                m.addConstr(sum(oa[mli_ue_ls]) >= 0 + eps - M * (1 - b[i]), name="bigM_constr1" + str(uav) + "_" + str(i))
                m.addConstr(sum(oa[mli_ue_ls]) <= 0 + M * b[i], name="bigM_constr2" + str(uav) + "_" + str(i)+"a")
                cubic_var = m.addMVar(shape=1, vtype=GRB.BINARY)
                m.addConstr(cubic_var == (b[i] * oa[mli_ue_ls[0]]))
                this_constr_trad_mem_ls.append(cubic_var * R_array[mli_ue_ls[0]] * xi[mli_ue_ls[0]])
                this_constr_nvm_mem_ls.append(cubic_var * R_array[mli_ue_ls[0]] * (1 - xi[mli_ue_ls[0]]))
            # Model if x > y, then b = 1, otherwise b = 0
            m.addConstr(sum(this_constr_trad_mem_ls) <= uav_server_trad_mem[uav], name="c"+str(curr))
            m.addConstr(sum(this_constr_nvm_mem_ls) <= uav_server_nvm_mem[uav], name="c"+str(curr+1))
            curr += 2
        
        
        fs_children = [[] for fs in range(fs_count)]
        for ue in range(ue_count):
            ps = user_data[str(ue)]["parent_server"]
            pps = system_data["server_info"][ps]["parent_server"]
            fs_children[pps-uav_count].append(ue)
        carr = []
        for fs in range(fs_count):
            # print(f"for {fs=}")
            children = fs_children[fs]
            mli_dict = {}
            for ue in children:
                curr_mli = user_data[str(ue)]["mli_idx"]
                if curr_mli not in mli_dict:
                    mli_dict[curr_mli] = []
                mli_dict[curr_mli].append(ue)
            b = m.addMVar(shape=len(mli_dict), vtype=GRB.BINARY, name="b"+str(fs+uav_count))
            carr.append(b)
            this_constr_trad_mem_ls = []
            this_constr_nvm_mem_ls = []
            for i, mli in enumerate(mli_dict):
                mli_ue_ls = mli_dict[mli]
                for idx in range(1, len(mli_ue_ls)):
                    # m.addConstr(oa[mli_ue_ls[idx]] == oa[mli_ue_ls[idx-1]])
                    m.addConstr(beta[mli_ue_ls[idx]] == beta[mli_ue_ls[idx-1]])
                    # m.addConstr(ohm[mli_ue_ls[idx]] == ohm[mli_ue_ls[idx-1]])
                    m.addConstr(yi[mli_ue_ls[idx]] == yi[mli_ue_ls[idx-1]])
                # print(f"children having {mli=} are:- {mli_ue_ls=}")
                m.addConstr(sum((ob[mli_ue_ls] - oab[mli_ue_ls])) >= 0.0 + eps - M * (1 - b[i]), name="bigM_constr1" + str(fs+uav_count) + "_" + str(i))
                m.addConstr(sum((ob[mli_ue_ls] - oab[mli_ue_ls])) <= 0.0 + M * b[i], name="bigM_constr2" + str(fs+uav_count) + "_" + str(i)+"a")
                cubic_var = m.addMVar(shape=1, vtype=GRB.BINARY)
                m.addConstr(cubic_var == (b[i] * beta[mli_ue_ls[0]]))
                this_constr_trad_mem_ls.append(cubic_var * R_array[mli_ue_ls[0]] * yi[mli_ue_ls[0]])
                this_constr_nvm_mem_ls.append(cubic_var * R_array[mli_ue_ls[0]] * (1 - yi[mli_ue_ls[0]]))
            # Model if x > y, then b = 1, otherwise b = 0
            m.addConstr(sum(this_constr_trad_mem_ls) <= fs_server_trad_mem[fs], name="c"+str(curr))
            m.addConstr(sum(this_constr_nvm_mem_ls) <= fs_server_nvm_mem[fs], name="c"+str(curr+1))
            curr += 2
        # # for fs in range(fs_count):
        # #     this_constr_trad_mem = sum((ob[fs_children[fs]] - oab[fs_children[fs]]) * yi[fs_children[fs]] * R_array[fs_children[fs]])
        # #     this_constr_nvm_mem = sum((ob[fs_children[fs]] - oab[fs_children[fs]]) * (1. - yi[fs_children[fs]]) * R_array[fs_children[fs]])
        # #     m.addConstr(this_constr_trad_mem <= fs_server_trad_mem[fs], name="c"+str(curr))
        # #     m.addConstr(this_constr_nvm_mem <= fs_server_nvm_mem[fs], name="c"+str(curr+1))
        # #     curr += 2

        # Optimize model
        m.optimize()

        # print(f"{ohm.X=}, {alpha.X=}, {oa.X=}, {xi.X=}")
        # print(f" for ue=32, {ohm.X[32]=}, {alpha.X[32]=}, {oa.X[32]=}, {xi.X[32]=}")
        # print(f"{barr[1].X=}")
        # print(f"{ohm.X=}, {alpha.X=}, {beta.X=}, {xi.X=}, {yi.X=}, {oa.X=}, {ob.X=}, {oab.X=}")
        print('Obj: %g' % m.ObjVal)
        #correctness checks
        # print(f"{ohm.X * alpha.X=}")
        # print(f"{oa.X=}")
        # assert (ohm.X * alpha.X == oa.X).all()
        # assert (ohm.X * beta.X == ob.X).all()
        # assert (oa.X * beta.X == oab.X).all()
        # for uav in range(uav_count):
        #     mli_supported_by_uav_dict = {}
        #     mli_profit_per_instance = {}
        #     for ue in range(ue_count):
        #         if user_data[str(ue)]["parent_server"] == uav:
        #             if oa.X[ue] == 0.:
        #                 continue
        #             this_mli = user_data[str(ue)]["mli_idx"]
        #             if this_mli not in mli_supported_by_uav_dict:
        #                 mli_supported_by_uav_dict[this_mli] = []
        #             if this_mli not in mli_profit_per_instance:
        #                 ener = (ohm.X[ue] * ue_uav_trans_lat[ue] * uav_receiving_power[ue])\
        #                     + (oa.X[ue] * ((uav_comp_trad_lat[ue] * xi.X[ue]) + (uav_comp_nvm_lat[ue] * (1 - xi.X[ue]))) * uav_computation_power[ue]) \
        #                     + (ohm.X[ue] * (1 - alpha.X[ue]) * uav_fs_trans_lat[ue] * (uav_transmission_power[ue] + fs_receiving_power[ue])) \
        #                     + ((ob.X[ue] - oab.X[ue]) * ((fs_comp_trad_lat[ue] * yi.X[ue]) + (fs_comp_nvm_lat[ue] * (1 - yi.X[ue]))) * fs_computation_power[ue]) \
        #                     + ((ohm.X[ue] - oa.X[ue] - ob.X[ue] + oab.X[ue]) * (fs_cloud_trans_lat[ue] * (fs_transmission_power[ue] + cloud_receiving_power[ue]) + cloud_comp_lat[ue] * cloud_computation_power[ue]))
        #                 rev = ohm.X[ue] * (a * (R_array[ue] / p_array[ue]))
        #                 mli_profit_per_instance[this_mli] = rev - (phi * ener)
        #             mli_supported_by_uav_dict[this_mli].append(ue)
        #     total_trad_mem_occupied = 0
        #     total_nvm_mem_occupied = 0
        #     for mli, ue_ls in mli_supported_by_uav_dict.items():
        #         total_trad_mem_occupied += (R_array[ue_ls[0]] * xi.X[ue_ls[0]])
        #         total_nvm_mem_occupied += (R_array[ue_ls[0]] * (1 - xi.X[ue_ls[0]]))
        #     print(f"for {uav=}, {mli_supported_by_uav_dict=}, {mli_profit_per_instance=}")
        #     print(f"{uav_server_trad_mem[uav]=}, {total_trad_mem_occupied=}")
        #     print(f"{uav_server_nvm_mem[uav]=}, {total_nvm_mem_occupied=}")


        # for fs in range(fs_count):
        #     mli_supported_by_fs_dict = {}
        #     mli_profit_per_instance = {}
        #     for ue in fs_children[fs]:
        #         if (ob.X[ue] - oab.X[ue] != 1.):
        #             continue
            
        #         this_mli = user_data[str(ue)]["mli_idx"]
        #         if this_mli not in mli_supported_by_fs_dict:
        #             mli_supported_by_fs_dict[this_mli] = []
        #         if this_mli not in mli_profit_per_instance:
        #             ener = (ohm.X[ue] * ue_uav_trans_lat[ue] * uav_receiving_power[ue])\
        #                 + (oa.X[ue] * ((uav_comp_trad_lat[ue] * xi.X[ue]) + (uav_comp_nvm_lat[ue] * (1 - xi.X[ue]))) * uav_computation_power[ue]) \
        #                 + (ohm.X[ue] * (1 - alpha.X[ue]) * uav_fs_trans_lat[ue] * (uav_transmission_power[ue] + fs_receiving_power[ue])) \
        #                 + ((ob.X[ue] - oab.X[ue]) * ((fs_comp_trad_lat[ue] * yi.X[ue]) + (fs_comp_nvm_lat[ue] * (1 - yi.X[ue]))) * fs_computation_power[ue]) \
        #                 + ((ohm.X[ue] - oa.X[ue] - ob.X[ue] + oab.X[ue]) * (fs_cloud_trans_lat[ue] * (fs_transmission_power[ue] + cloud_receiving_power[ue]) + cloud_comp_lat[ue] * cloud_computation_power[ue]))
        #             rev = ohm.X[ue] * (a * (R_array[ue] / p_array[ue]))
        #             mli_profit_per_instance[this_mli] = rev - (phi * ener)
        #         mli_supported_by_fs_dict[this_mli].append(ue)
        #     total_trad_mem_occupied = 0
        #     total_nvm_mem_occupied = 0
        #     for mli, ue_ls in mli_supported_by_fs_dict.items():
        #         total_trad_mem_occupied += (R_array[ue_ls[0]] * yi.X[ue_ls[0]])
        #         total_nvm_mem_occupied += (R_array[ue_ls[0]] * (1 - yi.X[ue_ls[0]]))
        #     print(f"for {fs=}, {mli_supported_by_fs_dict=}, {mli_profit_per_instance=}")
        #     print(f"{fs_server_trad_mem[fs]=}, {total_trad_mem_occupied=}")
        #     print(f"{fs_server_nvm_mem[fs]=}, {total_nvm_mem_occupied=}")


    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    env_params_folder = os.path.join(os.getcwd(), "environment_class")
    # extracting global_UE_count
    env_params_filename = "environment_parameters.json"
    with open(os.path.join(env_params_folder, env_params_filename),"r") as env_params_file_obj:
        env_params = json.load(env_params_file_obj)
    global_UE_count = env_params["UE_count"]

    optimal_matching_filename = "optimalmatching_results.json"
    optimal_matching = {
        "UE_count": global_UE_count,
        "profit": {
            "optimal_profit": float(m.ObjVal)
            }
    }
    with open(os.path.join(data_folder, optimal_matching_filename),"w") as outfile:
        json.dump(optimal_matching, outfile, indent=4)

    return


if __name__ == "__main__":
    current_dir = os.getcwd()
    data_folder = os.path.join(current_dir, "dataset", "local_dataset")
    apply_gurobi(data_folder=data_folder, algo_type="dts")