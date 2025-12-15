import sys

from xidle_metrics_class import *
from xidle_metrics_saliency import *
from xidle_data import *
from xidle_learn import *
from xidle_net import *
from xidle_gen import *

import torch
from torch import nn as nn
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import copy

matplotlib.use('Agg')
plt.rcParams['axes.facecolor'] = 'white'

import argparse

if __name__ == '__main__':
    print('torch.get_num_threads()=%d'%torch.get_num_threads())

    lr, patience_reduce_lr = 1e-5, 40
    optimizer_dict = {'optimizer':optim.Adam, 'param':{}, 'name':'Adam'}
    lr_factor = 0.1
    lr_min = 1.0e-8
    duration_max = 23.5*60*60
    patience_early_stop = patience_reduce_lr*2+3
    
    lg_sigma_image = None
    lg_sigma_class = 0.0
    n_classes = 4
    
    down = 5
    blur = 500

    classification_loss = nn.CrossEntropyLoss()
    saliency_pred_loss = nn.KLDivLoss(reduction='batchmean')

    Metrics = {'class':MetricsHandle_Class, 'saliency':MetricsHandle_Saliency}
    Model = XIDLE_Net_preset

    name = 'NET'
    folder_string = 'run'
    qH = QuickHelper(path=os.getcwd()+'/'+folder_string)
    print('New Folder name: %s'%qH.ID)
    print(folder_string)

    data_timer = QuickTimer()
    path = './XIDLE-Net/dataset'
    batch_size = 16
    epoch_max = 20
        
    dataAll = DataMaster(path=path, batch_size=batch_size)
    print('Data Preparing time: %fsec'%data_timer())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Net = Model(
        device=device,
        out_dict={'class':4, 'image':1},
        loss_dict={'class':lg_sigma_class, 'image':lg_sigma_image},
        img_size=224, num_classes = n_classes)
    
    model_dict = Net.state_dict()
    swin_path = './XIDLE-Net/Module/model_data/swin_base_patch4_window7_224_imagenet1k.pth'
    swin_ckpt = torch.load(swin_path, map_location=device)
    swin_dict = swin_ckpt['model'] if 'model' in swin_ckpt else swin_ckpt

    load_key_s, no_load_key_s, temp_dict_s = [], [], {}

    for k, v in swin_dict.items():
        full_key = "swin." + k

        if full_key in model_dict and model_dict[full_key].shape == v.shape:
            temp_dict_s[full_key] = v
            load_key_s.append(full_key)
        else:
            no_load_key_s.append(full_key)

    model_dict.update(temp_dict_s)

    print("\n[Swin] Successful Load Key:", str(load_key_s)[:500], "……\nLoaded:", len(load_key_s))
    print("\n[Swin] Fail To Load Key:", str(no_load_key_s)[:500], "……\nUnloaded:", len(no_load_key_s))

    Net.load_state_dict(model_dict)

    Net.to(device)

    netLearn = NetLearn(
        net=Net,
        dataAll=dataAll,
        criterion={'class':classification_loss, 'saliency':saliency_pred_loss},
        optimizer_dict=optimizer_dict,
        lr=lr,
        lr_min=lr_min,
        lr_factor=lr_factor,
        epoch_max=epoch_max,
        duration_max=duration_max,
        patience_reduce_lr=patience_reduce_lr,
        patience_early_stop=patience_early_stop,
        device=device,
        metrics=Metrics,
        name=name,
        path=qH())

    netLearn.load_params('./XIDLE-Net/Module/model_weight/')

    print(netLearn.test())
    qH.summary()

