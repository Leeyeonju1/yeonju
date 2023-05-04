#!/usr/bin/env python
# coding: utf-8

# # Import package & data

# In[1]:


# Unzipped the dataset
get_ipython().system("tar -xvf  'Task01_BrainTumour.tar'")


# In[2]:


import os
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, RandRotate90d,
    LoadImaged,Resized,
    RandFlipd, RandRotated,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    AddChanneld, CropForegroundd,
    RandCropByPosNegLabeld, AsDiscrete,
    RandCropByLabelClassesd
   )
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss, DiceFocalLoss
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
import einops
import wandb
from utils import *
from seg_swinunetr import Swinunetr
from datetime import datetime
from pytz import timezone


# In[3]:


# Here you can change the hyperparameter
class CFG:
    max_epochs = 10
    in_channels = 4
    out_channels = 4
    train_batch_size = 1
    test_batch_size = 1
    optimizer_name = 'AdamW'
    loss_function = 'DiceCELoss'
    from_scale_intensity_max = 3000
    from_scale_intensity_min = 0
    to_scale_intensity_max = 1
    to_scale_intensity_min = 0
    feature_size = 24
    save_epoch = 20
    vol_size = 96
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


import os
import torch

def load_data(URL, train_ratio):
    '''
    URL: The directory data located
    train ratio: The ratio of train dataset
    '''
    torch.manual_seed(40)
    folder = [os.path.join(URL, o) for o in os.listdir(URL) if os.path.isdir(os.path.join(URL,o))]
    del folder[0]

    for url in folder:
        if url == 'Task01_BrainTumour/imagesTr':
            images = list(map(lambda x: url + "/" + x, sorted(list(filter(lambda x: not x.startswith('.'), os.listdir(url))), key=lambda x: int(x[-10:-7]))))
        
        elif url == 'Task01_BrainTumour/labelsTr':
            labels = list(map(lambda x: url + "/" + x, sorted(list(filter(lambda x: not x.startswith('.'), os.listdir(url))), key=lambda x: int(x[-10:-7]))))

    # Pair image and label
    full_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(images, labels)]

    # Spliting train / validation / test
    train_size = int(train_ratio * len(full_files))
    test_size = len(full_files) - train_size
    train_files, test_files = torch.utils.data.random_split(full_files, [train_size, test_size]) 

    return train_files, test_files

train_files, test_files = load_data("Task01_BrainTumour", 0.9)


# In[5]:


test_transform = [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),         
        ScaleIntensityRanged(
            keys=["image"], 
            a_min=CFG.from_scale_intensity_min, 
            a_max=CFG.from_scale_intensity_max, 
            b_min=CFG.to_scale_intensity_min, 
            b_max=CFG.to_scale_intensity_max, 
            clip=True
        ),
]

train_transform = test_transform.copy()

# Below is some data augmentation technique
train_transform.extend([
        RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96,96,96),
                num_samples=1,
                pos=1,
                neg=1
            ),
])

train_transforms, test_transforms = Compose(train_transform), Compose(test_transform)


# In[6]:


train_ds = Dataset(data=train_files, transform=train_transforms)  
train_loader = DataLoader(train_ds, batch_size=CFG.train_batch_size, shuffle=True, drop_last=True)

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=CFG.test_batch_size, shuffle=False, drop_last=False)  


# In[7]:


# Load the model
def load_model():
    model = Swinunetr(img_size=(96,96,96), in_channels=CFG.in_channels, out_channels=CFG.out_channels, feature_size=CFG.feature_size)
    model = model.to('cuda')
    weight=torch.load('../swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt')
    model.load_from(weights=weight)
    print("Using pretrained self-supervised Swin UNETR backbone weights !")
    return model


# In[8]:


import torch
import torch.nn as nn

def my_dice(val_output,val_label):

    smooth = 1e-7
    denom = torch.sum(val_output,axis=(1,2,3)) + torch.sum(val_label,axis=(1,2,3)) + smooth
    intersection = torch.sum(val_label*val_output,axis=(1,2,3))
    num = 2*intersection + smooth
    res = torch.mean((num/denom), axis = 0)

    return res

def calculate_seg_avg(results):
    
    overall_avg = sum([i['overall'] for i in results.values()])/len(results)
    back_avg = sum([i['bg'] for i in results.values()])/len(results)
    edema_avg = sum([i['edema'] for i in results.values()])/len(results)
    non_enhancing_avg = sum([i['non_enhancing'] for i in results.values()])/len(results)
    enhancing_avg = sum([i['enhancing'] for i in results.values()])/len(results)
    
    return overall_avg, back_avg, edema_avg, non_enhancing_avg, non_enhancing_avg

def check_dice_metrics(pred_label, true_label, num_class):
    
    true = nn.functional.one_hot(true_label.long(), num_classes=num_class)
    true = torch.squeeze(true, 1)
    pred = torch.argmax(pred_label, dim=1)
    pred = nn.functional.one_hot(pred.long(), num_classes=num_class)
    perf = my_dice(pred, true)
    bg, edema, non_enhancing_tumor, enhancing_tumor = perf
    overall = torch.mean(perf)
    
    return overall, bg, edema, non_enhancing_tumor, enhancing_tumor


# In[9]:

def train_metrics(model, train_loader):
    with torch.no_grad():
        l = torch.zeros(4).view(1, -1).cuda()
        for batch in train_loader:
            torch.cuda.empty_cache()
            train_inputs, train_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                logits = model(train_inputs)
            
            overall, bg, edema, non_enhancing_tumor, enhancing_tumor  = check_dice_metrics(logits, train_labels, 4)
            res = torch.Tensor([bg.item(), edema.item(), non_enhancing_tumor.item(), enhancing_tumor.item()]).view(1, -1).cuda()
            l = torch.cat([l, res], dim=0) 
    
        l = l[1:, ...]
        back, edema, non_enhancing, enhancing = torch.nanmean(l, dim=0)
        overall = torch.mean(torch.nanmean(l, dim=0))
    
    return overall, back, edema, non_enhancing, enhancing


# In[10]:


def train(train_loader, model, optimizer, loss_fn):
    model.train()
    steps = len(train_loader)
    epoch_iterator = tqdm(
      train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
      )
    train_results = {}
    for step, batch in enumerate(epoch_iterator):
        step += 1
        with torch.cuda.amp.autocast():
          train_inputs, train_labels = (batch["image"].cuda(), batch["label"].cuda())
          logits = model(train_inputs) 
          loss = loss_fn(logits, train_labels)
          
        # Backward the loss
        loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()
        optimizer.zero_grad()

        # Save dice metrics  
        overall, bg, edema, non_enhancing_tumor, enhancing_tumor  = check_dice_metrics(logits, train_labels, CFG.out_channels)
        del logits
        del train_labels
        train_results.update({step: {'overall': overall, 'bg': bg, 'edema': edema, 'non_enhancing': non_enhancing_tumor, 'enhancing': enhancing_tumor}})

        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)"
            % (step, steps, loss.item())
            )

    return calculate_seg_avg(train_results)



# In[19]:


post_label = AsDiscrete(to_onehot=4)
post_pred = AsDiscrete(argmax=True, to_onehot=4)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
def inference(model, test_loader):
    with torch.no_grad():
        l = torch.zeros(4).view(1, -1).cuda()
        for batch in test_loader:
            torch.cuda.empty_cache()
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, device="cpu", progress=True, overlap=0.8)
                torch.cuda.empty_cache()
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor).cuda() for val_pred_tensor in val_outputs_list]
            l = torch.cat([l, dice_metric(y_pred=val_output_convert, y=val_labels_convert)], dim=0)
            print(dice_metric(y_pred=val_output_convert, y=val_labels_convert))
           
        l = l[1:, ...]
        back, edema, non_enhancing, enhancing = torch.nanmean(l, dim=0)
        overall = torch.mean(torch.nanmean(l, dim=0))
        dice_metric.reset()
    
    return overall, back, edema, non_enhancing, enhancing

def test(test_loader, model):
    
    model.eval()
    
    return inference(model, test_loader)

def save_model(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# In[13]:


URL = 'Directory you want to saved model'


# ## You only have to change the weight here!

# In[14]:


# In[15]:


def main():
    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    #model, optimizer, start_epoch = resume_training()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=torch.Tensor([w1, w2, w3, w4].cuda()) 
    
    for epoch in range(1, CFG.max_epochs + 1):
        # Call train & test function
        train_result = train(train_loader, model, optimizer, loss_fn)
        # test every 5 epoch
        if epoch % 5 == 0:
            print("############### Epoch {} Train performance ###############".format(epoch))
            overall, bg, edema, non_enhancing_tumor, enhancing_tumor = train_metrics(model, train_loader)
            print(train_metrics(model, train_loader))
            print("############### Epoch {} Test performance ###############".format(epoch))
            test_overall, test_bg, test_edema, test_non_enhancing, test_enhancing = test(test_loader, model)
        
        if (epoch % 5 == 0) | (epoch == CFG.max_epochs):
            checkpoint = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }
            save_model(checkpoint, filename=os.path.join(URL, "epoch{}_model.pth".format(epoch)))
      
        


# In[16]:


main()


# ------------
