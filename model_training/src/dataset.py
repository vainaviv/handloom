import torch
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import numpy as np
import os
from datetime import datetime
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import shutil
sys.path.insert(0, '../')
from config import *

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# # Domain randomization
# img_transform = iaa.Sequential([
#     iaa.flip.Fliplr(0.5),
#     iaa.flip.Flipud(0.5),
#     iaa.MultiplyBrightness((0.7, 1.2)),
    # iaa.AddToBrightness((-50, 50)),
    # iaa.Resize({"height": 200, "width": 200}),
    # sometimes(iaa.Affine(
    #     scale = {"x": (0.7, 1.3), "y": (0.7, 1.3)},
    #     rotate=(-30, 30),
    #     shear=(-30, 30)
    # ))
    # ], random_order=False)

# # No randomization
# no_transform = iaa.Sequential([])

# New domain randomization
img_transform_new = iaa.Sequential([
    iaa.flip.Flipud(0.5),
    iaa.flip.Fliplr(0.5),
    # rotate 90, 180, or 270
    iaa.Rot90([0, 1, 2, 3]),
    sometimes(iaa.Affine(
        scale = {"x": (0.7, 1.3), "y": (0.7, 1.3)},
        rotate=(-30, 30),
        shear=(-30, 30)
        ))
    ], random_order=True)
# flips and rot90
augmentation_list = [] #[iaa.flip.Fliplr(0.5), iaa.flip.Flipud(0.5), iaa.Rot90([0, 1, 2, 3])]
 #iaa.MultiplyBrightness((0.7, 1.2)),]
no_augmentation_list = []

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False, single=False):
    if not single:
        U.unsqueeze_(1).unsqueeze_(2)
        V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return (G/G.max()).double() * 2
    return G.double()

def gauss_2d_batch_efficient_np(width, height, sigma, U, V, weights, normalize=False):
    crop_size = 3 * sigma
    ret = np.zeros((height + 2*crop_size, width + 2*crop_size + 1))
    X,Y = np.meshgrid(np.arange(-crop_size, crop_size+1), np.arange(-crop_size, crop_size+1))
    gaussian = np.exp(-((X)**2+(Y)**2)/(2.0*sigma**2))
    for i in range(len(weights)):
        cur_weight = weights[i]
        y, x = int(V[i]) + crop_size, int(U[i]) + crop_size
        if ret[y-crop_size:y+crop_size+1, x-crop_size:x+crop_size+1].shape == gaussian.shape:
           ret[y-crop_size:y+crop_size+1, x-crop_size:x+crop_size+1] = np.max((cur_weight * gaussian, ret[y-crop_size:y+crop_size+1, x-crop_size:x+crop_size+1]), axis=0)

    if normalize:
        ret = ret / ret.max()
    return ret[crop_size:crop_size+height, crop_size:crop_size+width]

def vis_gauss(img, gaussians, i):
    gaussians = gaussians.cpu().detach().numpy().transpose(1, 2, 0)
    img = (img.cpu().detach().numpy().transpose(1, 2, 0) * 255)
    gaussians = np.concatenate((gaussians, np.zeros_like(gaussians[:, :, :1]), np.zeros_like(gaussians[:, :, :1])), axis=2)
    h1 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    crop = np.expand_dims(img[:, :, -1], axis=-1)
    crop_og = np.tile(crop, 3)
    if not os.path.exists('./dataset_py_train'):
        os.mkdir('./dataset_py_train')
    img[:, :, 2] = gaussians[:, :, 0] * 255
    cv2.imwrite(f'./dataset_py_test/test-img_{i:05d}.png', img[...,::-1])
    cv2.imwrite(f'./dataset_py_test/test-crop_{i:05d}.png', crop_og)

def vis_gauss_input_output(img, gaussians, i):
    gaussians = gaussians.cpu().detach().numpy().transpose(1, 2, 0)
    img = (img.cpu().detach().numpy().transpose(1, 2, 0) * 255)
    gaussians = np.concatenate((gaussians, np.zeros_like(gaussians[:, :, :1]), np.zeros_like(gaussians[:, :, :1])), axis=2) * 255
    cv2.imwrite(f'./dataset_py_test2/input_{i:05d}.png', img)
    cv2.imwrite(f'./dataset_py_test2/label_{i:05d}.png', gaussians)

def bimodal_gauss(G1, G2, normalize=False):
    bimodal = torch.max(G1, G2)
    if normalize:
        return normalize(bimodal)
    return bimodal

def get_gauss(w, h, sigma, U, V):
    gaussians = gauss_2d_batch(w, h, sigma, U, V)
    if gaussians.shape[0] > 0:
        mm_gauss = gaussians[0]
        for i in range(1, len(gaussians)):
            mm_gauss = bimodal_gauss(mm_gauss, gaussians[i])
        mm_gauss.unsqueeze_(0)
        return mm_gauss
    return torch.zeros(1, h, w).cuda().double()

def perform_contrast(img):
    # img = img.copy()

    # show a histogram of brightness values in the image
    # values = img[:, :, 0].flatten()
    # plt.hist(values, bins=256, range=(0, 256), fc='k', ec='k')
    # plt.show()
    cable_mask = img[:, :, 1] > (120/255.0)
    pixels_x_normalize, pixels_y_normalize = np.where(cable_mask > 0)
    if len(pixels_x_normalize) == 0:
        return img
    pixel_vals = img[pixels_x_normalize, pixels_y_normalize, :]
    min_px_val = np.min(pixel_vals)
    max_px_val = np.max(pixel_vals)
    # cable_norm = pixels_normalize / np.linalg.norm(pixels_normalize)
    cable_mask = np.array([cable_mask, cable_mask, cable_mask]).transpose((1,2,0))
    # background = img * (1 - cable_mask)
    # print('img mask', img.max(), img.min(), cable_mask.min(), cable_mask.max())
    # print((img * cable_mask).max(), (img * cable_mask).min())
    cable = ((img * cable_mask) - min_px_val) / (max_px_val - min_px_val)
    # cable = cable * 255.0
    cable *= cable_mask
    # print('after mmx:', min_px_val, max_px_val, np.min(cable), np.max(cable))
    return cable #img_contrast

class KeypointsDataset(Dataset):
    def __init__(self, folder, transform, augment=True, sweep=True, seed=1, real_only=False, config=None):
        self.img_height = config.img_height
        self.img_width = config.img_width
        self.gauss_sigma = config.gauss_sigma
        self.transform = transform
        self.sharpen = config.sharpen
        self.contrast = config.contrast
        self.expand_spline = config.expand_spline
        self.mark_crossing = config.mark_crossing

        real_world_transform_list = augmentation_list if augment else no_augmentation_list
        sim_transform_list = list(real_world_transform_list)
        # sim_transform_list.extend([])
        brightness_avg = 10
        if self.sharpen:
            kernel = np.array([[0, -0.25, 0],
                    [-0.25, 2,-0.25],
                    [0, -0.25, 0]])
            noise_avg = 4
            sim_transform_list.extend([iaa.Convolve(matrix=kernel)])
        else:
            noise_avg = 6
        aug_delta = 2 if augment else 0
        brightness_delta = 5 if augment else 0
        sim_transform_list.extend([iaa.AdditiveGaussianNoise(scale=(noise_avg - aug_delta, noise_avg + aug_delta)),
        iaa.WithBrightnessChannels(iaa.Add((brightness_avg - brightness_delta, brightness_avg + brightness_delta)))])
        
        sim_transform_list.append(iaa.Resize({"height": self.img_height, "width": self.img_width}))
        real_world_transform_list.append(iaa.Resize({"height": self.img_height, "width": self.img_width}))

        self.sim_img_transform = iaa.Sequential(sim_transform_list, random_order=False)
        self.real_img_transform = iaa.Sequential(real_world_transform_list, random_order=False)
        self.augment = augment
        self.condition_len = config.condition_len
        self.crop_width = config.crop_width
        self.crop_span = self.crop_width*2
        self.pred_len = config.pred_len
        self.spacing = config.cond_point_dist_px
        self.sweep = sweep
        self.seed = seed
        self.oversample = config.oversample
        self.oversample_rate = config.oversample_rate
        self.rot_cond = False #config.rot_cond
        self.dataset_real = config.dataset_real if not real_only else [True for _ in range(len(folder))]

        self.data = []
        self.expt_type = config.expt_type

        self.weights = np.geomspace(0.5, 1, self.condition_len)
        self.label_weights = np.ones(self.pred_len) # np.geomspace(1, 0.5, self.pred_len)

        self.folder_sizes = []
        dataset_weights = config.dataset_weights if not real_only else [1.0 for _ in range(len(folder))]
        self.folder_weights = np.array(dataset_weights)/np.sum(dataset_weights)
        self.folder_counts = np.zeros(len(self.folder_weights))
        # if self.expt_type == ExperimentTypes.TRACE_PREDICTION:
        folders = folder
        print('Loading data from', folders)
        for folder in folders:
            if os.path.exists(folder):
                count = 0
                for fname in sorted(os.listdir(folder)):
                    # if os.path.isdir(os.path.join(folder, fname)):
                    #     continue
                    self.data.append(os.path.join(folder, fname))
                    count += 1
                self.folder_sizes.append(count)
            else:
                raise FileNotFoundError(f'Folder {folder} does not exist')
        self.folder_sizes = np.array(self.folder_sizes)

    def _get_evenly_spaced_points(self, pixels, num_points, start_idx, spacing, img_size, backward=True, randomize_spacing=True):
        def is_in_bounds(pixel):
            return pixel[0] >= 0 and pixel[0] < img_size[0] and pixel[1] >= 0 and pixel[1] < img_size[1]
        def get_rand_spacing(spacing):
            return spacing * np.random.uniform(0.8, 1.2) if randomize_spacing else spacing
        # get evenly spaced points
        last_point = np.array(pixels[start_idx]).squeeze()
        points = [last_point]
        if not is_in_bounds(last_point):
            return np.array([])
        rand_spacing = get_rand_spacing(spacing)
        while len(points) < num_points and start_idx > 0 and start_idx < len(pixels):
            start_idx -= (int(backward) * 2 - 1)
            cur_spacing = np.linalg.norm(np.array(pixels[start_idx]).squeeze() - last_point)
            if cur_spacing > rand_spacing and cur_spacing < 2*rand_spacing:
                last_point = np.array(pixels[start_idx]).squeeze()
                rand_spacing = get_rand_spacing(spacing)
                if is_in_bounds(last_point):
                    points.append(last_point)
                else:
                    return np.array([])
        return np.array(points)[..., ::-1]

    def rotate_condition(self, img, points, center_around_last=False, index=0):
        img = img.copy()
        angle = 0
        # points = self.deduplicate_points(points)
        if self.rot_cond:
            if center_around_last:
                dir_vec = points[-1] - points[0]
            else:
                dir_vec = points[-self.pred_len-1] - points[-self.pred_len-2]
            angle = np.arctan2(dir_vec[1], dir_vec[0]) * 180/np.pi
            if angle < -90.0:
                angle += 180
            elif angle > 90.0:
                angle -= 180
            # rotate image specific angle using cv2.rotate
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return img, angle

    def get_trp_model_input(self, crop, crop_points, center_around_last=False, is_real_example=False):
        kpts = KeypointsOnImage.from_xy_array(crop_points, shape=crop.shape)
        img, kpts = self.call_img_transform(img=crop, kpts=kpts, is_real_example=is_real_example)

        points = []
        for k in kpts:
            points.append([k.x,k.y])
        points = np.array(points)

        points_in_image = []
        for i, point in enumerate(points):
            px, py = int(point[0]), int(point[1])
            if px not in range(img.shape[1]) or py not in range(img.shape[0]):
                continue
            points_in_image.append(point)
        points = np.array(points_in_image)

        angle = 0
        if self.rot_cond:
            if center_around_last:
                dir_vec = points[-1] - points[-2]
            else:
                dir_vec = points[-self.pred_len-1] - points[-self.pred_len-2]
            angle = np.arctan2(dir_vec[1], dir_vec[0])

            # rotate image specific angle using cv2.rotate
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle*180/np.pi, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


        # rotate all points by angle around center of image
        points = points - np.array([img.shape[1]/2, img.shape[0]/2])
        points = np.matmul(points, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
        points = points + np.array([img.shape[1]/2, img.shape[0]/2])

        if center_around_last:
            img[:, :, 0] = self.draw_spline(img, points[:,1], points[:,0])# * cable_mask
        else:
            img[:, :, 0] = self.draw_spline(img, points[:-self.pred_len,1], points[:-self.pred_len,0])# * cable_mask

        cable_mask = np.ones(img.shape[:2])
        cable_mask[img[:, :, 1] < 0.4] = 0

        return transform(img.copy()).cuda(), points, cable_mask, angle
    
    def get_crop_and_cond_pixels(self, img, condition_pixels, center_around_last=False):
        center_of_crop = condition_pixels[-self.pred_len*(1 - int(center_around_last))-1]
        img = np.pad(img, ((self.crop_width, self.crop_width), (self.crop_width, self.crop_width), (0, 0)), 'constant')
        center_of_crop = center_of_crop.copy() + self.crop_width

        crop = img[max(0, center_of_crop[0] - self.crop_width): min(img.shape[0], center_of_crop[0] + self.crop_width + 1),
                    max(0, center_of_crop[1] - self.crop_width): min(img.shape[1], center_of_crop[1] + self.crop_width + 1)]
        img = crop
        top_left = [center_of_crop[0] - self.crop_width, center_of_crop[1] - self.crop_width]
        condition_pixels = [[pixel[0] - top_left[0] + self.crop_width, pixel[1] - top_left[1] + self.crop_width] for pixel in condition_pixels]

        return img, np.array(condition_pixels)[:, ::-1], top_left

    def deduplicate_points(self, points):
        # print("points to deduplicate: ", points)
        x = points[:,0]
        y = points[:,1]
        x = list(OrderedDict.fromkeys(x))
        y = list(OrderedDict.fromkeys(y))
        tmp = OrderedDict()
        for point in zip(x, y):
            tmp.setdefault(point[:2], point)
        mypoints = np.array(list(tmp.values()))
        return mypoints

    def draw_spline(self, crop, x, y, label=False):
        # x, y = points[:, 0], points[:, 1]
        if len(x) < 2:
            raise Exception("if drawing spline, must have 2 points minimum for label")
        # x = list(OrderedDict.fromkeys(x))
        # y = list(OrderedDict.fromkeys(y))
        tmp = OrderedDict()
        for point in zip(x, y):
            tmp.setdefault(point[:2], point)
        mypoints = np.array(list(tmp.values()))
        x, y = mypoints[:, 0], mypoints[:, 1]
        k = len(x) - 1 if len(x) < 4 else 3
        if k == 0:
            x = np.append(x, np.array([x[0]]))
            y = np.append(y, np.array([y[0] + 1]))
            k = 1

        tck, u = interpolate.splprep([x, y], s=0, k=k)
        xnew, ynew = interpolate.splev(np.linspace(0, 1, 100), tck, der=0)
        xnew = np.array(xnew, dtype=int)
        ynew = np.array(ynew, dtype=int)

        x_in = np.where(xnew < crop.shape[0])
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        x_in = np.where(xnew >= 0)
        xnew = xnew[x_in[0]]
        ynew = ynew[x_in[0]]
        y_in = np.where(ynew < crop.shape[1])
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]
        y_in = np.where(ynew >= 0)
        xnew = xnew[y_in[0]]
        ynew = ynew[y_in[0]]

        spline = np.zeros(crop.shape[:2])
        if label:
            weights = np.ones(len(xnew))
        else:
            weights = np.geomspace(0.5, 1, len(xnew))

        spline[xnew, ynew] = weights
        spline = np.expand_dims(spline, axis=2)
        spline = np.tile(spline, 3)
        if self.expand_spline:
            spline_dilated = cv2.dilate(spline, np.ones((9,9), np.uint8), iterations=1)
        else:
            spline_dilated = cv2.dilate(spline, np.ones((3,3), np.uint8), iterations=1)
        return spline_dilated[:, :, 0]

    def call_img_transform(self, img, kpts=None, is_real_example=False):
        img_transform = self.real_img_transform if is_real_example else self.sim_img_transform
        img = img.copy()
        img = (img * 255.0).astype(np.uint8)
        if kpts:
            img, keypoints = img_transform(image=img, keypoints=kpts)
            img = (img / 255.0).astype(np.float32)
            return img, keypoints
        else:
            img = img_transform(image=img)
            img = (img / 255.0).astype(np.float32)
            return img

    def __getitem__(self, data_index):
        # ignore data_index and get random data
        if data_index >= self.__len__():
            raise IndexError()
        folder_to_sample = np.random.choice(np.arange(len(self.folder_sizes)), p=self.folder_weights)
        data_index = int(self.folder_sizes[:folder_to_sample].sum() + self.folder_counts[folder_to_sample])
        self.folder_counts[folder_to_sample] += 1
        self.folder_counts[folder_to_sample] = self.folder_counts[folder_to_sample] % self.folder_sizes[folder_to_sample]
        is_real_example = self.dataset_real[folder_to_sample]

        loaded_data = np.load(self.data[data_index], allow_pickle=True).item()
        if self.expt_type == ExperimentTypes.TRACE_PREDICTION:
            img = loaded_data['img'][:, :, :3]

            if not is_real_example:
                delta = 0.8 #np.random.uniform(low=0.8, high=1.0)
                img = cv2.resize(img, (int(img.shape[0]*delta), int(img.shape[1]*delta)))
                for idx in loaded_data['pixels'].keys():
                    loaded_data['pixels'][idx] = [(int(loaded_data['pixels'][idx][0][0]*delta), int(loaded_data['pixels'][idx][0][1]*delta))]
                
            if img.max() > 1:
                img = (img / 255.0).astype(np.float32)
            pixels = loaded_data['pixels']
            cable_mask = np.ones(img.shape[:2])
            cable_mask[img[:, :, 1] <= 0.3] = 0.0
            dense_points = loaded_data['dense_points']
            iters = 0
            while True:
                if self.oversample and len(dense_points) > 0 and np.random.random() < self.oversample_rate:
                    start_idx = np.random.choice(dense_points)
                else:
                    start_idx = np.random.randint(0, len(pixels) - (self.condition_len + self.pred_len))
                condition_pixels = self._get_evenly_spaced_points(pixels, self.condition_len + self.pred_len, start_idx, self.spacing, img.shape, backward=True)
                if len(condition_pixels) == self.condition_len + self.pred_len:
                    break
                if iters > 10:
                    return self.__getitem__(np.random.randint(0, len(self.data)))
                iters += 1

            # get crop and crop-relative condition pixels
            if self.augment and not is_real_example:
                # pass
                jitter = np.random.randint(-2, 3, size=condition_pixels.shape) * 0
                jitter[-self.pred_len:] = 0
                condition_pixels = condition_pixels + jitter

            img, cond_pix_array, _ = self.get_crop_and_cond_pixels(img, condition_pixels)

        elif self.expt_type == ExperimentTypes.CAGE_PREDICTION:
            # getting img, pixels, and cage_point 
            img = loaded_data['img'][:, :, :3]
            pixels = loaded_data['pixels']
            cage_point = loaded_data['cage_point']

            # finding pixels within img boundaries
            within_bounds_pixels = []
            img_dim_x, img_dim_y = img.shape[0], img.shape[1]
            for i, pixel in enumerate(pixels):
                px, py = int(pixel[0]), int(pixel[1]) 
                # ignore off-frame pixels - adding the rest in
                if px not in range(img_dim_x) or py not in range(img_dim_y):
                    continue
                within_bounds_pixels.append(pixel)

            # beginning conditioning at 6th pixel (0-indexed) within img boundaries 
            start_idx = 5
            condition_pixels = self._get_evenly_spaced_points(within_bounds_pixels, self.condition_len, start_idx, self.spacing, img.shape, backward=False)            
            condition_pixels_array = np.array(condition_pixels)
            # note: need to flip condition_pixels for augmentation
            condition_pixels_array = condition_pixels_array[:, ::-1]
            
            # adding jitter to all condition_pixels
            jitter = np.random.uniform(-1, 1, size=condition_pixels_array.shape)
            jitter[-1] = 0
            condition_pixels_array = condition_pixels_array + jitter

            # getting array of keypoints (kpts_array) = condition_pixels_array (flipped) + cage_point_array
            cage_point_array = np.array([cage_point])
            kpts_array = np.append(condition_pixels_array, cage_point_array, axis=0)

            # getting final keypoints (final_kpts) post-transformation
            kpts_on_image = KeypointsOnImage.from_xy_array(kpts_array, shape=img.shape)
            img, transformed_kpts = self.call_img_transform(img, kpts=kpts_on_image, is_real_example=is_real_example) #self.img_transform(image=img, keypoints=kpts_on_image)
            final_kpts = []
            for k in transformed_kpts:
                final_kpts.append([k.x, k.y])
            final_kpts = np.array(final_kpts)

            # getting cable mask (cable_mask)
            cable_mask = np.ones(img.shape[:2])
            cable_mask[img[:, :, 1] < 0.35] = 0
        
            # getting img / combined
            img[:, :, 0] =  self.draw_spline(img, final_kpts[:-self.pred_len, 1], final_kpts[:-self.pred_len, 0])
            combined = transform(img.copy()).cuda()

            # generating the gauss / label out of the cage point
            label = torch.as_tensor(gauss_2d_batch_efficient_np(self.img_width, self.img_height, self.gauss_sigma, final_kpts[-self.pred_len:, 0], final_kpts[-self.pred_len:, 1], weights=self.label_weights))
            label = label
            label = label.unsqueeze_(0).cuda()
        
        elif self.expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or self.expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
            img = (loaded_data['crop_img'][:, :, :3]).copy()
            if img.max() > 1:
                img = (img / 255.0).astype(np.float32)

            condition_pixels = np.array(loaded_data['spline_pixels'], dtype=np.float64)

            kpts = KeypointsOnImage.from_xy_array(condition_pixels, shape=img.shape)
            img, kpts = self.call_img_transform(img=img, kpts=kpts, is_real_example=is_real_example)
            condition_pixels = []
            for k in kpts:
                condition_pixels.append([k.x,k.y])
            condition_pixels = np.array(condition_pixels)

            if img.max() > 1:
                img = (img / 255.0).astype(np.float32)
            if self.contrast:
                img = perform_contrast(img)
            # print('after contrast min max', img.min(), img.max())
            cable_mask = np.ones(img.shape[:2])
            cable_mask[img[:, :, 1] < 0.35] = 0
            if self.sweep:
                img[:, :, 0] = self.draw_spline(img, condition_pixels[:, 1], condition_pixels[:, 0], label=True) #* cable_mask
            else:
                img[:, :, 0] = gauss_2d_batch_efficient_np(self.crop_span, self.crop_span, self.gauss_sigma, condition_pixels[:-self.pred_len,0], condition_pixels[:-self.pred_len,1], weights=self.weights)
            if self.mark_crossing:
                img[:, :, 1] = gauss_2d_batch_efficient_np(self.crop_span, self.crop_span, self.gauss_sigma, [self.crop_width], [self.crop_width], weights=[1.0])
            if is_real_example and self.crop_width == 10:
                img = img[1:, 1:, :]
                img = cv2.resize(img, (self.img_height, self.img_width))
            img, _= self.rotate_condition(img, condition_pixels, center_around_last=True, index=data_index)
            combined = transform(img.copy()).cuda()
            label = torch.as_tensor(loaded_data['under_over']).double().cuda()

        if self.expt_type == ExperimentTypes.TRACE_PREDICTION:
            combined, points, cable_mask, _ = self.get_trp_model_input(img, cond_pix_array, is_real_example=is_real_example)

            if self.pred_len == 1:
                label = torch.as_tensor(gauss_2d_batch_efficient_np(self.img_width, self.img_height, self.gauss_sigma, points[-self.pred_len:, 0], points[-self.pred_len:, 1], weights=self.label_weights, normalize=True))
            else:
                try:
                    label = torch.as_tensor(self.draw_spline(img, points[-self.pred_len:,1], points[-self.pred_len:,0], label=True)) 
                except:
                    label = torch.as_tensor(gauss_2d_batch_efficient_np(self.img_width, self.img_height, self.gauss_sigma, points[-self.pred_len:, 0], points[-self.pred_len:, 1], weights=self.label_weights, normalize=True))
            label = label #* cable_mask
            label = label.unsqueeze_(0).cuda()

        elif self.expt_type != ExperimentTypes.CAGE_PREDICTION and self.expt_type != ExperimentTypes.CLASSIFY_OVER_UNDER and self.expt_type != ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
            # input processing
            condition_mask = np.zeros(img.shape)
            for condition_pixel in condition_pixels[:len(condition_pixels)//2]:
                condition_mask[int(condition_pixel[1]), int(condition_pixel[0])] = 1.0
            condition_with_cable = np.where(condition_mask > 0, 1, 0)
            if self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
                end_mask = np.zeros(img.shape)
                for condition in condition_pixels[len(condition_pixels)//2:]:
                    end_mask[int(condition[1]), int(condition[0])] = 1.0
                aug_input_concat_tuple = (img, condition_with_cable, end_mask)

        if self.expt_type == ExperimentTypes.OPPOSITE_ENDPOINT_PREDICTION:
            pull_with_cable_and_img = self.call_img_transform(np.concatenate(aug_input_concat_tuple, axis=2)) #self.img_transform(image=np.concatenate(aug_input_concat_tuple, axis=2))
            # split into img and mask again
            img = pull_with_cable_and_img[:, :, 0:3].copy()
            condition_with_cable = pull_with_cable_and_img[:, :, 3:6].copy()
            combined = cv2.resize(img.astype(np.float64), (self.img_width, self.img_height))
            combined = self.transform(combined).cuda().float()
            condition_with_cable = cv2.resize(condition_with_cable.astype(np.float64), (self.img_width, self.img_height))
            if condition_with_cable.sum() > 0:
                cond_V, cond_U = np.nonzero(condition_with_cable[:, :, 0])
                cond_U, cond_V = torch.from_numpy(np.array([cond_U, cond_V], dtype=np.int32)).cuda()
                combined[0] = 255.0 * get_gauss(self.img_width, self.img_height, self.gauss_sigma, cond_U, cond_V)
            else:
                raise Exception("No condition")
            end_mask = pull_with_cable_and_img[:, :, 6:9].copy()
            end_mask = cv2.resize(end_mask.astype(np.float64), (self.img_width, self.img_height))
            if end_mask.sum() > 0:
                end_V, end_U = np.nonzero(end_mask[:, :, 0])
                end_U, end_V = torch.from_numpy(np.array([end_U, end_V], dtype=np.int32)).cuda()
                label = 1.0 * get_gauss(self.img_width, self.img_height, self.gauss_sigma, end_U, end_V)

        return combined, label
    
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    dataset_test_path = './dataset_py_test'
    if os.path.exists(dataset_test_path):
        shutil.rmtree(dataset_test_path)
    os.mkdir(dataset_test_path)
    os.mkdir(os.path.join(dataset_test_path, 'under'))
    os.mkdir(os.path.join(dataset_test_path, 'over'))
    os.mkdir(os.path.join(dataset_test_path, 'none'))

    # UNDER OVER
    test_config = TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp()
    test_dataset = KeypointsDataset([os.path.join(d, 'test') for d in test_config.dataset_dir],
                                    transform,
                                    augment=False, 
                                    config=test_config)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    for i_batch, sample_batched in enumerate(test_data):
        img, gauss = sample_batched
        gauss = gauss.squeeze(0)
        img = img.squeeze(0)
        vis_gauss(img, gauss, i_batch)
        # label = int(label.detach().squeeze().cpu().numpy().item())
        # print(i_batch, label)
        # img = img.squeeze(0)
        # img = (img.cpu().detach().numpy().transpose(1, 2, 0) * 255)
        # if label == 0:
        #     cv2.imwrite(f'./dataset_py_test/under/test-img_{i_batch:05d}.png', img[...,::-1])
        # elif label == 1:
        #     cv2.imwrite(f'./dataset_py_test/over/test-img_{i_batch:05d}.png', img[...,::-1])
        # else:
        #     cv2.imwrite(f'./dataset_py_test/none/test-img_{i_batch:05d}.png', img[...,::-1])


    # TRACE PREDICTION
    # test_config = TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_Medley_MoreReal_Sharp() #TRCR32_CL3_12_PL1_RotCond_Sharp_Hard2_WReal()
    # test_dataset2 = KeypointsDataset([os.path.join(dir, 'test') for dir in test_config.dataset_dir],
    #                                 transform,
    #                                 augment=False, 
    #                                 config=test_config)
    # test_data = DataLoader(test_dataset2, batch_size=1, shuffle=True, num_workers=1)
    # for i_batch, sample_batched in enumerate(test_data):
    #     print(i_batch)
    #     img, gauss = sample_batched
    #     gauss = gauss.squeeze(0)
    #     img = img.squeeze(0)
    #     vis_gauss(img, gauss, i_batch)


    # # TRACE PREDICTION
    # test_config = TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_WReal()
    # test_dataset2 = KeypointsDataset([os.path.join(test_config.dataset_dir, 'train'), os.path.join(test_config.real_dataset_dir, 'train')],
    #                                 test_config.img_height,
    #                                 test_config.img_width,
    #                                 transform,
    #                                 gauss_sigma=test_config.gauss_sigma, 
    #                                 augment=True, 
    #                                 condition_len=test_config.condition_len,
    #                                 crop_width=test_config.crop_width, 
    #                                 spacing=test_config.cond_point_dist_px,
    #                                 expt_type=ExperimentTypes.TRACE_PREDICTION, 
    #                                 pred_len=1,
    #                                 config=test_config,
    #                                 oversample=test_config.oversample)
    # test_data = DataLoader(test_dataset2, batch_size=1, shuffle=True, num_workers=1)
    # for i_batch, sample_batched in enumerate(test_data):
    #     print(i_batch)
    #     img, gauss = sample_batched
    #     gauss = gauss.squeeze(0)
    #     img = img.squeeze(0)
    #     vis_gauss(img, gauss, i_batch)