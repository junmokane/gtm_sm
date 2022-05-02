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

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# specifications
path_to_load = f'saves/[2022-04-20 01:09:41]_no_matrix_loss_train_sigmoid_original/gtm_sm_state_dict_70.pth'
training_wo_wall = False
training_sigmoid = True
random_walk = False

# load data
data_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
testing_dataset = dset.ImageFolder(root='./datasets/CelebA/testing', transform=data_transform)
loader_val = DataLoader(testing_dataset, batch_size=args.batch_size, shuffle=True)

if torch.cuda.is_available():
    state_dict = torch.load(path_to_load)
else:
    state_dict = torch.load(path_to_load, map_location=lambda storage, loc: storage)

GTM_SM_model = GTM_SM(batch_size = args.batch_size, 
                      training_wo_wall=training_wo_wall, 
                      training_sigmoid=training_sigmoid)
GTM_SM_model.load_state_dict(state_dict)
GTM_SM_model.to(device=device)

exp = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]_jm_trained_w_{training_wo_wall}_s_{training_sigmoid}_r_{random_walk}'
fname = f"./results/{exp}/"
os.makedirs(fname, exist_ok=True)


def sample():
    GTM_SM_model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader_val):

            # transforming data
            training_data = data.to(device=device)
            
            # forward
            if random_walk:
                kld_loss, nll_loss, matrix_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model.forward(training_data)
            else:
                kld_loss, nll_loss, matrix_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model.forward_tmp(training_data)

            show_experiment_information(GTM_SM_model, data, st_observation_list, st_prediction_list, xt_prediction_list, position, fname)
            break  # run only once    
sample()