import pickle
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from config import ALL_EXPERIMENTS_CONFIG, is_point_pred, save_config_params
from src.model import KeypointsGauss, ClassificationModel
from src.dataset import KeypointsDataset, transform
import matplotlib.pyplot as plt
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument('--expt_name', type=str, default='')
parser.add_argument('--expt_class', type=str, default='UNDER_OVER')
parser.add_argument('--checkpoint_path', type=str, default='')

flags = parser.parse_args()

# get time in PST
experiment_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
expt_name = flags.expt_name
expt_class = flags.expt_class
checkpoint_path = flags.checkpoint_path

if expt_class not in ALL_EXPERIMENTS_CONFIG:
    raise ValueError(f"expt_class must be one of {list(ALL_EXPERIMENTS_CONFIG.keys())}")

config = ALL_EXPERIMENTS_CONFIG[expt_class]()

if expt_name == '':
    expt_name = f"{experiment_time}_{expt_class}"
else:
    expt_name = f"{experiment_time}_{expt_class}_{expt_name}"

def forward(sample_batched, model):
    img, gt_gauss = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_gauss = model.forward(img).double()
    if expt_class == 'UNDER_OVER_NONE':
        gt_class = np.zeros(pred_gauss.shape)
        idxs_1 = gt_gauss.byte().cpu().detach().numpy()
        idxs_0 = np.arange(idxs_1.shape[0], dtype=int)
        idxs = np.vstack((idxs_0, idxs_1)).T
        for idx in idxs:
            gt_class[idx[0]][idx[1]] = 1.0
        gt_class = torch.from_numpy(gt_class).cuda()
        loss = nn.BCELoss()(pred_gauss.squeeze(), gt_class.squeeze())
    else:
        loss = nn.BCELoss()(pred_gauss.squeeze(), gt_gauss.squeeze())
    return loss

def fit(train_data, test_data, model, epochs, optimizer, checkpoint_path = ''):
    train_epochs = []
    test_epochs = []
    test_losses = []
    train_losses = []
    last_checkpoint_epoch = -1
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0.0
        num_iters = len(train_data) / config.batch_size
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='') #,f"\t epoch est. time. left {((time.time() - start_time) * (num_iters) / (i_batch + 1)) * (epochs - epoch)}", end='')
            print('\r', end='')
        print('train loss:', train_loss / (i_batch + 1))
        train_epochs.append(epoch)
        train_losses.append(train_loss / (i_batch + 1))

        if epoch % config.eval_checkpoint_freq == (config.eval_checkpoint_freq - 1):
            test_loss = 0.0
            for i_batch, sample_batched in enumerate(test_data):
                loss = forward(sample_batched, model)
                test_loss += loss.item()
            test_loss_per_batch = test_loss / (i_batch + 1)
            print('test loss:', test_loss_per_batch)
            test_epochs.append(epoch)
            test_losses.append(test_loss_per_batch)

            np.save(os.path.join(checkpoint_path, "test_losses.npy"), test_losses)
            np.save(os.path.join(checkpoint_path, "train_losses.npy"), train_losses)

            if len(test_losses) <= 1 or test_loss_per_batch < np.min(test_losses[:-1]) or epoch - last_checkpoint_epoch >= config.min_checkpoint_freq:
                torch.save(keypoints.state_dict(), os.path.join(checkpoint_path, f'model_{epoch}_{test_loss_per_batch:.5f}.pth'))
                last_checkpoint_epoch = epoch

# dataset
workers=0
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, expt_name)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def append_to_folders(folder_name, append='train'):
    return [os.path.join(folder, append) for folder in folder_name]

train_dataset = KeypointsDataset(append_to_folders(config.dataset_dir, 'train'),
                                transform, 
                                augment=True, 
                                config=config)
test_dataset = KeypointsDataset(append_to_folders(config.dataset_dir, 'test'),
                                transform, 
                                augment=False,
                                config=config)

train_data = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=workers)
test_data = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=workers)


use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
if not is_point_pred(config.expt_type):
    keypoints = ClassificationModel(num_classes=config.classes, img_height=config.img_height, img_width=config.img_width, resnet_type=config.resnet_type).cuda()
else:
    keypoints = KeypointsGauss(num_keypoints=1, img_height=config.img_height, img_width=config.img_width, resnet_type=config.resnet_type, pretrained=config.pretrained).cuda()

# load from checkpoint
if checkpoint_path != '':
    print('loading checkpoint from', checkpoint_path)
    keypoints.load_state_dict(torch.load(checkpoint_path))

# optimizer
optimizer = optim.Adam(keypoints.parameters(), lr=config.learning_rate, weight_decay=1.0e-4)

# save the config to a file
save_config_params(save_dir, config)
fit(train_data, test_data, keypoints, epochs=config.epochs, optimizer=optimizer, checkpoint_path=save_dir)
