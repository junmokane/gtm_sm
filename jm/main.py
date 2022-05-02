import os
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.torch_utils import initNetParams, ChunkSampler, show_images, device_agnostic_selection
from model import GTM_SM
from config import *
from show_results import show_experiment_information
from jm.walk import explore_walk_wo_wall
from jm.visualize import show_trajectory, show_experiment

# plt.rcParams['figure.figsize'] = (33.0, 10.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#load data
data_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
testing_dataset = dset.ImageFolder(root='./datasets/CelebA/testing',
                                           transform=data_transform)
loader_val = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=True)

path_to_load = f'saves/[2022-04-19 16:24:11]_no_matrix_loss/gtm_sm_state_dict_40.pth'
# path_to_load = f'saves/gtm_sm_state_dict.pth'
if torch.cuda.is_available():
    state_dict = torch.load(path_to_load)
else:
    state_dict = torch.load(path_to_load, map_location=lambda storage, loc: storage)
GTM_SM_model = GTM_SM(batch_size=args.batch_size,
                      observe_dim=180,
                      total_dim=308,)
GTM_SM_model.load_state_dict(state_dict)
GTM_SM_model.to(device=device)

GTM_SM_model.eval()
test_loss = 0
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(loader_val):
        # transforming data
        training_data = data.to(device=device)
        
        ''' # To see how exploration works, run following
        action_one_hot_value, position, action_selection, goto_ran_len_list = explore_walk_wo_wall(GTM_SM_model)
        show_trajectory(position, goto_ran_len_list, GTM_SM_model)
        '''
        
        # forward
        kld_loss, nll_loss, matrix_loss, st_observation_list, st_prediction_list, xt_prediction_list, position, goto_ran_len_list, memory_time_index = GTM_SM_model.forward_eval(training_data)
        if batch_idx == 0:
            show_experiment(GTM_SM_model, data, st_observation_list, 
                            st_prediction_list, xt_prediction_list, 
                            position, goto_ran_len_list, memory_time_index)
        test_loss += nll_loss

test_loss /= len(loader_val.dataset)
print('====> Test set loss: {:.4f}'.format(test_loss))        
        
        
        