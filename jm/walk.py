import torch
import numpy as np
from config import *

# image size is fixed as 32x32 [0,1,...,8] * 3 is the available nodes
def explore_walk_wo_wall(model):
    # construct position and action
    action_one_hot_value_numpy = np.zeros((model.batch_size, model.a_dim, model.total_dim - 1), np.float32)
    position = np.zeros((model.batch_size, model.s_dim, model.total_dim), np.int32)
    action_selection = np.zeros((model.batch_size, model.total_dim - 1), np.int32)
    goto_ran_len_list = np.zeros((model.batch_size,), np.int32)
        
    # global exploration variable (traverse whole image)
    order = [0, 3, 1, 2]  # right, down, left, up
    global_exp_action_seq = []
    for i in range(8):
        for j in range(i + 1):
            global_exp_action_seq.append(order[(2 * i) % 4])    
        for j in range(i + 1):
            global_exp_action_seq.append(order[(2 * i + 1) % 4])    
    for i in range(8):
        global_exp_action_seq.append(0)
    
    # observation phase (exploration) and prediction phase (random walk)
    for index_sample in range(model.batch_size):    
        
        # local exploration variable (go to random position)
        random_position = np.random.randint(1, 8, size=(2,))
        goto_rand_action_seq = []
        dist = np.array([random_position[0], 8 - random_position[1]], np.int)
        for i in range(dist[0]):
            goto_rand_action_seq.append(3)
        for i in range(dist[1]):
            goto_rand_action_seq.append(1)
        goto_ran_len_list[index_sample] = len(goto_rand_action_seq)
        
        # global exploration & local exploration
        for t in range(model.observe_dim):
            if t == 0:
                position[index_sample, :, t] = np.ones(2) * 4
            elif t > 0 and t < 81:  # global exploration
                action = global_exp_action_seq[t - 1]
                if action == 0:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                    action_selection[index_sample, t - 1] = action
                elif action == 1:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                    action_selection[index_sample, t - 1] = action
                elif action == 2:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                    action_selection[index_sample, t - 1] = action
                elif action == 3:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                    action_selection[index_sample, t - 1] = action
            elif t >= 81 and t < 81 + dist[0] + dist[1]:  # go to random position
                action = goto_rand_action_seq[t - 81]
                if action == 1:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                    action_selection[index_sample, t - 1] = action
                elif action == 3:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                    action_selection[index_sample, t - 1] = action
            else:  # local exploration
                if (position[index_sample, :, t - 1] == random_position).all():
                    action = np.random.choice([0, 1, 2, 3], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([0, 1])).all():
                    action = np.random.choice([1, 2, 3], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([1, 1])).all():
                    action = np.random.choice([1, 2], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([1, 0])).all():
                    action = np.random.choice([0, 1, 2], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([1, -1])).all():
                    action = np.random.choice([0, 2], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([0, -1])).all():
                    action = np.random.choice([0, 2, 3], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([-1, -1])).all():
                    action = np.random.choice([0, 3], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([-1, 0])).all():
                    action = np.random.choice([0, 1, 3], 1)[0]
                elif (position[index_sample, :, t - 1] == random_position + np.array([-1, 1])).all():                    
                    action = np.random.choice([1, 3], 1)[0]
                
                if action == 0:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                    action_selection[index_sample, t - 1] = action
                elif action == 1:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                    action_selection[index_sample, t - 1] = action
                elif action == 2:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                    action_selection[index_sample, t - 1] = action
                elif action == 3:
                    position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                    action_selection[index_sample, t - 1] = action
        
        # random walk wo wall
        new_continue_action_flag = True
        for t in range(model.observe_dim, model.total_dim):
            if new_continue_action_flag:
                new_continue_action_flag = False
                need_to_stop = False

                while 1:
                    action_random_selection = np.random.randint(0, 4, size=(1))
                    if not (action_random_selection == 0 and position[index_sample, 1, t - 1] == 8):
                        if not (action_random_selection == 1 and position[index_sample, 1, t - 1] == 0):
                            if not (action_random_selection == 2 and position[index_sample, 0, t - 1] == 0):
                                if not (action_random_selection == 3 and position[index_sample, 0, t - 1] == 8):
                                    break
                                
                action_duriation = np.random.poisson(2, 1)

            if action_duriation > 0:
                if not need_to_stop:
                    if action_random_selection == 0:
                        if position[index_sample, 1, t - 1] == 8:
                            need_to_stop = True
                            position[index_sample, :, t] = position[index_sample, :, t - 1]
                            action_selection[index_sample, t - 1] = 4
                        else:
                            position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                            action_selection[index_sample, t - 1] = action_random_selection
                    elif action_random_selection == 1:
                        if position[index_sample, 1, t - 1] == 0:
                            need_to_stop = True
                            position[index_sample, :, t] = position[index_sample, :, t - 1]
                            action_selection[index_sample, t - 1] = 4
                        else:
                            position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                            action_selection[index_sample, t - 1] = action_random_selection
                    elif action_random_selection == 2:
                        if position[index_sample, 0, t - 1] == 0:
                            need_to_stop = True
                            position[index_sample, :, t] = position[index_sample, :, t - 1]
                            action_selection[index_sample, t - 1] = 4
                        else:
                            position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                            action_selection[index_sample, t - 1] = action_random_selection
                    elif action_random_selection == 3:
                        if position[index_sample, 0, t - 1] == 8:
                            need_to_stop = True
                            position[index_sample, :, t] = position[index_sample, :, t - 1]
                            action_selection[index_sample, t - 1] = 4
                        else:
                            position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0])
                            action_selection[index_sample, t - 1] = action_random_selection
                else:
                    position[index_sample, :, t] = position[index_sample, :, t - 1]
                    action_selection[index_sample, t - 1] = 4
                action_duriation -= 1
            else:
                action_selection[index_sample, t - 1] = 4
                position[index_sample, :, t] = position[index_sample, :, t - 1]
            if action_duriation <= 0:
                new_continue_action_flag = True
    
    for index_sample in range(model.batch_size):
        action_one_hot_value_numpy[index_sample, action_selection[index_sample], np.array(range(model.total_dim - 1))] = 1

    action_one_hot_value = torch.from_numpy(action_one_hot_value_numpy).to(device=device)

    return action_one_hot_value, position, action_selection, goto_ran_len_list
        
        
            
            
            
            
            