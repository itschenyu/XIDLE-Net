#########################################################################################
######## Augumentation (Hflip+rotate) for 'gaze' and 'cxr' + Color Jitter for cxr #####
#########################################################################################

from xidle_gen import *

import torch, pickle, copy
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import torchvision.transforms as T
import torchvision.transforms.functional as TF

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os, random
import cv2



def random_scale_and_crop(cxr, gaze, target_shape, jitter=0.3, scale_range=(0.25, 2.0)):
    _, H, W = cxr.shape
    out_h, out_w = target_shape
    # aspect ratio
    ar = W/H * np.random.uniform(1-jitter,1+jitter) / np.random.uniform(1-jitter,1+jitter)
    scale = np.random.uniform(*scale_range)
    if ar < 1:
        nh = int(scale*out_h)
        nw = int(nh*ar)
    else:
        nw = int(scale*out_w)
        nh = int(nw/ar)
    # 
    cxr_resized = F.interpolate(cxr.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
    if gaze.ndim == 2:
        gaze = gaze.unsqueeze(0)
    gaze_resized = F.interpolate(gaze.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False).squeeze(0)
    # 
    dx = int(np.random.uniform(0, out_w-nw)) if out_w-nw>0 else 0
    dy = int(np.random.uniform(0, out_h-nh)) if out_h-nh>0 else 0
    # 
    cxr_new = torch.ones((3, out_h, out_w), dtype=cxr.dtype) * 0.5   
    gaze_new = torch.zeros((1, out_h, out_w), dtype=gaze.dtype)
    # paste
    cxr_new[:, dy:dy+nh, dx:dx+nw] = cxr_resized
    gaze_new[:, dy:dy+nh, dx:dx+nw] = gaze_resized
    return cxr_new, gaze_new.squeeze(0)

def random_hsv_jitter(cxr, hue=.1, sat=.7, val=.3):
    # cxr: Tensor(3,H,W), 0~1 float
    np_img = (cxr.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    img_hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    r = np.random.uniform(-1, 1, 3) * [hue*180, sat*255, val*255] + [0, 0, 0]
    # 
    img_hsv = img_hsv.astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] + r[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * (1 + r[1]/255), 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * (1 + r[2]/255), 0, 255)
    img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img_rgb = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.
    return img_rgb


class DataHandle(Dataset):
    
    def __init__(self, path, augment=False):
        """
        :param path: Train/Test/Valid
        :param augment: True/False
        """
        self.path = path
        self.augment = augment
        self.init()

        if self.augment:
            self.color_jitter = T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.ToTensor()
            ])
        else:
            self.color_jitter = None
        
    def init(self):
        self.file_list = ls_file(path=self.path)
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):
        """
        """
        local_path = os.path.join(self.path, self.file_list[i])
        data = torch.load(local_path)

        if self.augment:
            data['cxr'], data['gaze'] = self.random_augmentation(data['cxr'], data['gaze'])

        return data
    
    def random_augmentation(self, cxr, gaze):
        """
        """
        # ------------------ 1)  ------------------
        if random.random() < 0.5:
            # cxr: (3,H,W) W
            cxr = torch.flip(cxr, dims=[2])
            
            if gaze.ndim == 3:
                gaze = torch.flip(gaze, dims=[2])
            else:
                gaze = torch.flip(gaze, dims=[1])
        
        # ------------------ 2) ------------------
        angle = random.uniform(-15, 15)
        
        cxr_4d = cxr.unsqueeze(0)
        # cxr_4d = TF.rotate(cxr_4d, angle)
        cxr_4d = TF.rotate(cxr_4d, angle, interpolation=TF.InterpolationMode.BILINEAR)
        cxr = cxr_4d.squeeze(0)
        
        if gaze.ndim == 2:
            # (H,W) => (1,H,W)
            gaze = gaze.unsqueeze(0)
        gaze_4d = gaze.unsqueeze(0)
        
        gaze_4d = TF.rotate(gaze_4d, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        gaze_3d = gaze_4d.squeeze(0)
        
        # 3
        cxr, gaze = random_scale_and_crop(cxr, gaze, (self.target_h, self.target_w))

        # 4. ColorJitter
        if random.random() < 0.3 and self.color_jitter is not None:
            cxr = self.color_jitter(cxr)

        # 5. HSV
        if random.random() < 0.3:
            cxr = random_hsv_jitter(cxr)
        return cxr, gaze
    
    def plot(self, i):
        """
        """
        data = self[i]
        plt.figure(figsize=(8,10))
        for idx, key in enumerate(data):
            plt.subplot(2, 3, 1 + idx)
            shape = data[key].shape
            if len(shape) == 3:
                plt.imshow(data[key][0,:,:]) 
            elif len(shape) == 2:
                plt.imshow(data[key])
            plt.title(f'{data[key].dtype}\n{data[key].shape}\n'
                      f'{data[key].min():.3f}\n{data[key].max():.3f}')
        plt.show()


class DataMaster:
    
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        name_list = ['Train','Test','Valid']
        num_workers = torch.get_num_threads()-1 if torch.get_num_threads() <= 9 else 8
        
        self.handle = {
            'Train': DataHandle(os.path.join(self.path, 'Train'), augment=True),
            'Test':  DataHandle(os.path.join(self.path, 'Test'),  augment=False),
            'Valid': DataHandle(os.path.join(self.path, 'Valid'), augment=False)
        }
        
        self.dataLoader = {
            item: DataLoader(
                self.handle[item],
                batch_size=self.batch_size,
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=4
            ) for item in name_list
        }
        
    def __call__(self, key):
        """
        DataMaster->Dataloader
        """
        return self.dataLoader[key]