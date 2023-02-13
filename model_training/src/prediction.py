import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import numpy as np

class Prediction:
    def __init__(self, model, num_keypoints, img_height, img_width, use_cuda=True, parallelize=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        if parallelize:
            self.model = nn.DataParallel(model)
            # self.model = nn.parallel.DistributedDataParallel(model)
            self.model.to(self.device)

        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.use_cuda = use_cuda
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width) 
        if len(imgs.shape) == 4:
            imgs = imgs.view(-1, imgs.shape[1], imgs.shape[2], imgs.shape[3])    
        elif len(imgs.shape) == 3:
            imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])
            
        heatmap = self.model.forward(Variable(imgs))
        return heatmap

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def expectation(self, d):
        width, height = d.T.shape
        d = d.T.ravel()
        d_norm = self.softmax(d)
        x_indices = np.array([i % width for i in range(width*height)])
        y_indices = np.array([i // width for i in range(width*height)])
        exp_val = [int(np.dot(d_norm, x_indices)), int(np.dot(d_norm, y_indices))]
        return exp_val
    
    def plot(self, input1, heatmap, image_id=0, cls=None, classes=None, write_image=True, heatmap_id=0):
        print("Running inferences on image: %d"%image_id)
        input1 = np.transpose(input1[0], (1,2,0))
        img = input1[:, :, :3] * 255
        img = img.astype(np.uint8)
        all_overlays = []
        h = heatmap[0][heatmap_id]
        vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.65, vis, 0.35, 0)
        #y, x = np.unravel_index(h.argmax(), h.shape)
        #heatmap_val = h[y,x]
        #while heatmap_val > 0.5:
        #    overlay = cv2.circle(overlay, (x,y), 4, (255,255,255), -1)
        #    h[max(0,y-15):min(y+15, len(h)), max(0,x-15):min(x+15, len(h[0]))] = 0
        #    y, x = np.unravel_index(h.argmax(), h.shape)
        #    heatmap_val = h[y,x]
        if write_image:
            cv2.imwrite('preds/out%04d.png'%image_id, overlay)
        return overlay

