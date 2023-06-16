import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import environment_class.environment as env_file
from environment_class.environment_wrapper import PreprocessEnv
from model.q_network import create_q_networks
from model.exploration_policy import policy
from training_scripts import training
import utils
import json

def main(algo_type="spa", copy_env_from_file=True, fs_bandwidth_plot_divisor=None, \
    reassign_server_counts=None):
    current_dir = os.getcwd()

    if "plots" not in os.listdir(current_dir):
        os.mkdir(os.path.join(current_dir, "plots"))
    plots_root_path = os.path.join(current_dir, "plots")

    if "results_and_logs" not in os.listdir(current_dir):
        os.mkdir("results_and_logs")
    results_and_log_root_path = os.path.join(current_dir, "results_and_logs")

    sanity_check_log_file_path_relative = "sanity_check_logs.txt"
    sanity_check_log_file_path = os.path.join(results_and_log_root_path, \
                                        sanity_check_log_file_path_relative)
    # algo_type = "dts"
    # fs_bandwidth_plot_divisor=100
    # copy_env_from_file=True
    env = env_file.Single_Agent_UAV(fraction_testing_mode=False, copy_env_from_file=copy_env_from_file, \
            which_algo_mode=algo_type, fs_bandwidth_plot_divisor=fs_bandwidth_plot_divisor, reassign_server_counts=reassign_server_counts)

    env.reset()
    if (not copy_env_from_file) or (reassign_server_counts is not None):
        if "dataset" not in os.listdir(current_dir):
            os.mkdir(os.path.join(current_dir, "dataset"))
        with open(os.path.join(current_dir, "dataset", "system_data.json"), "w") as outfile:
            json.dump(env.system_data, outfile, indent=4)
        with open(os.path.join(current_dir, "dataset", "global_user_data.json"), "w") as outfile:
            json.dump(env.global_user_data, outfile, indent=4)
    num_ues_covered = env.get_count_of_UEs_covered()
    # # COMMENTED FOR TESTING
    # env.render(testing_mode = False, render_plot_path = plots_root_path)
    # # END OF COMMENTED BLOCK

    state_dims = (env.state_space[0].shape[0] * env.UAV_count)
    num_actions = env.action_space[0].n
    env = PreprocessEnv(env)
    num_training_episodes = (env.UAV_count * 200) + 5
    q_nets, target_q_nets = create_q_networks(state_dims, num_actions, env)

    # policy(state, q_networks, epsilon=0., UAV_index=0, num_actions=9)

    training_log_file_path_relative = algo_type + "training_logs.txt"
    training_log_file_path = os.path.join(results_and_log_root_path, \
                                    training_log_file_path_relative)
    # stats = training.deep_q_learning(training_log_file_path, q_nets, target_q_nets, policy, episodes, env, alpha = 0.001,
    #                     batch_size = 32, gamma = 0.99, epsilon = 0.2)
    stats = training.deep_q_learning( \
            log_file=training_log_file_path, \
            q_networks=q_nets, target_q_networks=target_q_nets, \
            policy=policy,  episodes=num_training_episodes, env=env, alpha = 0.001, \
            batch_size = 32, gamma = 0.99, epsilon = 0.2 \
        )

    training_loss_and_avg_returns_plot_file_path_relative = algo_type + "loss_and_returns.png"
    training_loss_and_avg_returns_plot_file_path = os.path.join(results_and_log_root_path, \
                                    training_loss_and_avg_returns_plot_file_path_relative)
    # COMMENTED FOR TESTING
    utils.plot_stats1(stats, plot_file=training_loss_and_avg_returns_plot_file_path)

    utils.run_random_episode(env=env, policy=policy, q_nets=q_nets, render_plot_path=plots_root_path)
    # END OF COMMENTED BLOCK

    # env.reset()
    # env.render()

if __name__ == "__main__":
    algo_type="spa"
    copy_env_from_file=True
    fs_bandwidth_plot_divisor=None
    main(algo_type=algo_type, copy_env_from_file=copy_env_from_file, fs_bandwidth_plot_divisor=fs_bandwidth_plot_divisor)