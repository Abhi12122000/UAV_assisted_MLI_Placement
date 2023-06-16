import torch
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from training_scripts import experience_buffer
from training_scripts.stat_module import list_
import numpy as np
import utils


def deep_q_learning(log_file, q_networks, target_q_networks, policy, episodes, env, alpha = 0.001,
                    batch_size = 32, gamma = 0.99, epsilon = 0.2):
    optims = [AdamW(q_networks[i].parameters(), lr= alpha) for i in range(env.UAV_count)]
    memory = experience_buffer.ReplayMemory()
    utils.write_num_epochs_to_env_params(env, episodes)
    # stats= {'MSE Loss': list_(is_loss=True), 'Average_Returns': list_()}
    stats= {'Last Reward of Episode (through the training process)': {
        'data': list_(env, policy, q_nets=q_networks),
        'x_label': 'training iteration no.',
        'y_label': 'reward in the episode\'s last timestep'
        }
    }
    log_file_obj = open(log_file, 'w')
    # print("reached here")
    for episode in tqdm(range(1, 1)):
    # for episode in tqdm(range(1, episodes+1)):
        # print(f"inside training {episode=}, {id(env.initial_UAV_state)=}, {env.initial_UAV_state=}")
        # env.render()
        # print("state after reset: ", state)
        # print("printing state type inside deep_q_learning() function: ", state)
        done = False
        ep_return = 0.
        log_file_obj.write("-----------------------------episode " + str(episode) + "-----------------------------\n")
        # print("-----------------------------episode ", episode, "-----------------------------")
        # print(f"inside training {episode=}")
        state = env.reset()
        while not done:
            # print(f"{env.timesteps_in_episode=}")
            np_state = torch.flatten(state, end_dim=-2).numpy()
            # print("test print here: ", torch.flatten(state, end_dim=-2)[0])
            # print("policy output of 0th network: ", policy(torch.flatten(state, end_dim=-2)[0], epsilon, 0))
            # action = torch.tensor([[[policy(torch.flatten(state, end_dim=-2)[i], epsilon, i)] for i in range(env.UAV_count)]])
            num_actions = env.action_space[0].n
            action = torch.tensor([[[policy(state, q_networks, env, epsilon, UAV_index=0, num_actions=9)] for i in range(env.UAV_count)]])
            np_action = [act.item() for act in action[0]]
            # print("current state: ", state)
            # print("current action being taken: ", action)
            # print("numpy version of the action: ", np_action)
            # print("extra printing: ", action[0][0])
            # print("action inside deep_q_learning(): ", action)
            next_state, reward, done, _ = env.step(np_action)
            # print("UEs covered: ", env.get_count_of_UEs_covered())
            # print("next state: ", next_state)
            memory.insert([state, action, reward, done, next_state])
            
            # COMMENTED FOR TESTING
            if memory.can_sample(batch_size):
                # print("reached here")
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                # action_b = torch.flatten(action_b, end_dim = -2)
                # print("UAV action tensor: ", action_b)
                # print("state sample 0th UAV: ", state_b[:, 0, :])
                # print("action sample 0th UAV: ", action_b[:, 0, :])
                # print("Entering gradient descent:-\n state: ", state_b[:, 0, :], "action: ", action_b[:, 0, :])
                loss_sum = 0
                for i in range(env.UAV_count):
                    # qsa_input_action = 
                    # qsa_b = q_networks[i](state_b[:, i, :]).gather(1, action_b[:, i, :])
                    qsa_b = q_networks[i](torch.flatten(state_b, start_dim=1, end_dim=-1)).gather(1, action_b[:, i, :])

                    # next_qsa_b = target_q_networks[i](next_state_b[:, i, :])
                    next_qsa_b = target_q_networks[i](torch.flatten(next_state_b, start_dim=1, end_dim=-1))
                    next_qsa_b = torch.max(next_qsa_b, dim =-1, keepdim=True)[0]
                    
                    # print("printing inverse of done values: ", ~done_b)
                    target_b = reward_b + (~done_b) * gamma * next_qsa_b
                    # print("printing target: ", target_b)
                    # print("reward[0]: ", reward_b[1], "done: ", done_b[1], "gamma: ", gamma, "next_qsa_b: ", next_qsa_b[1])
                    loss = F.mse_loss(qsa_b, target_b)
                    q_networks[i].zero_grad()
                    loss.backward()
                    optims[i].step()
                    loss_sum += loss
                # stats['MSE Loss'].append(loss_sum)
            # END OF TESTING

            state = next_state
            ep_return +=reward.item()
        log_file_obj.write("no. of UEs covered (Reward): " + str(reward.item()) + "\n")
        # print("no. of UEs covered (Reward): ", reward.item())
        # stats['Average_Returns'].append(float(ep_return) / env.timesteps_in_episode)
        stats['Last Reward of Episode (through the training process)']['data'].append(reward.item())

        if episode % 10 == 0:
            for i in range(env.UAV_count):
                target_q_networks[i].load_state_dict(q_networks[i].state_dict())
    
    stats['Last Reward of Episode (through the training process)']['data'] = stats['Last Reward of Episode (through the training process)']['data']._list()
    # stats['MSE Loss'] = stats['MSE Loss']._list()

    log_file_obj.close()
    return stats