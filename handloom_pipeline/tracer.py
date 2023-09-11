import pickle
import cv2
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage
from torchvision import transforms, utils
from collections import OrderedDict
from scipy import interpolate
import colorsys
import shutil
from enum import Enum

sys.path.insert(0, '..')
from model_training.src.model import KeypointsGauss
from model_training.config import *
from analytic_tracer import simple_uncertain_trace_single

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class TraceEnd(Enum):
    EDGE = 1
    ENDPOINT = 2
    FINISHED = 3
    RETRACE = 4

class Tracer:
    def __init__(self) -> None:
        self.trace_config = TRCR32_CL3_12_UNet34_B64_OS_MedleyFix_MoreReal_Sharp()
        self.trace_model =  KeypointsGauss(1, img_height=self.trace_config.img_height, img_width=self.trace_config.img_width, channels=3, resnet_type=self.trace_config.resnet_type, pretrained=self.trace_config.pretrained).cuda()
        self.trace_model.load_state_dict(torch.load('../models/tracer/tracer_model.pth')) # Uncomment for bajcsy
        augs = []
        augs.append(iaa.Resize({"height": self.trace_config.img_height, "width": self.trace_config.img_width}))
        self.real_img_transform = iaa.Sequential(augs, random_order=False)
        self.transform = transforms.Compose([transforms.ToTensor()])
        # TODO: fix
        self.x_buffer = 30
        self.y_buffer = 50
        self.ep_buffer = 15

    def _get_evenly_spaced_points(self, pixels, num_points, start_idx, spacing, img_size, backward=True, randomize_spacing=True):
        pixels = np.squeeze(pixels)
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
        start_idx -= (int(backward) * 2 - 1)
        while start_idx > 0 and start_idx < len(pixels):
            cur_spacing = np.linalg.norm(np.array(pixels[start_idx]).squeeze() - last_point)
            if cur_spacing > rand_spacing and cur_spacing < 2*rand_spacing:
                last_point = np.array(pixels[start_idx]).squeeze()
                rand_spacing = get_rand_spacing(spacing)
                if is_in_bounds(last_point):
                    points.append(last_point)
                else:
                    points = points[-num_points:]
                    return np.array(points)[..., ::-1]
            start_idx -= (int(backward) * 2 - 1)
        points = points[-num_points:]
        return np.array(points)

    def center_pixels_on_cable(self, image, pixels):
        # for each pixel, find closest pixel on cable
        image_mask = image[:, :, 0] > 100
        # erode white pixels
        kernel = np.ones((2,2),np.uint8)
        image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
        white_pixels = np.argwhere(image_mask)
        
        # # visualize this
        # plt.imshow(image_mask)
        # for pixel in pixels:
        #     plt.scatter(*pixel[::-1], c='r')
        # plt.show()

        processed_pixels = []
        for pixel in pixels:
            # find closest pixel on cable
            distances = np.linalg.norm(white_pixels - pixel, axis=1)
            closest_pixel = white_pixels[np.argmin(distances)]
            processed_pixels.append([closest_pixel])
        return np.array(processed_pixels)

    def call_img_transform(self, img, kpts):
        img = img.copy()
        normalize = False
        if np.max(img) <= 1.0:
            normalize = True
        if normalize:
            img = (img * 255.0).astype(np.uint8)
        img, keypoints = self.real_img_transform(image=img, keypoints=kpts)
        if normalize:
            img = (img / 255.0).astype(np.float32)
        return img, keypoints

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
        spline_dilated = cv2.dilate(spline, np.ones((3,3), np.uint8), iterations=1)
        return spline_dilated[:, :, 0]

    def get_crop_and_cond_pixels(self, img, condition_pixels, center_around_last=False):
        center_of_crop = condition_pixels[-self.trace_config.pred_len*(1 - int(center_around_last))-1]
        img = np.pad(img, ((self.trace_config.crop_width, self.trace_config.crop_width), (self.trace_config.crop_width, self.trace_config.crop_width), (0, 0)), 'constant')
        center_of_crop = center_of_crop.copy() + self.trace_config.crop_width

        crop = img[max(0, center_of_crop[0] - self.trace_config.crop_width): min(img.shape[0], center_of_crop[0] + self.trace_config.crop_width + 1),
                    max(0, center_of_crop[1] - self.trace_config.crop_width): min(img.shape[1], center_of_crop[1] + self.trace_config.crop_width + 1)]
        img = crop
        top_left = [center_of_crop[0] - self.trace_config.crop_width, center_of_crop[1] - self.trace_config.crop_width]
        condition_pixels = [[pixel[0] - top_left[0] + self.trace_config.crop_width, pixel[1] - top_left[1] + self.trace_config.crop_width] for pixel in condition_pixels]

        return img, np.array(condition_pixels)[:, ::-1], top_left

    def get_trp_model_input(self, crop, crop_points, center_around_last=False):
        kpts = KeypointsOnImage.from_xy_array(crop_points, shape=crop.shape)
        img, kpts = self.call_img_transform(img=crop, kpts=kpts)

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
        if self.trace_config.rot_cond:
            if center_around_last:
                dir_vec = points[-1] - points[-2]
            else:
                dir_vec = points[-self.trace_config.pred_len-1] - points[-self.trace_config.pred_len-2]
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
            img[:, :, 0] = self.draw_spline(img, points[:-self.trace_config.pred_len,1], points[:-self.trace_config.pred_len,0])# * cable_mask

        cable_mask = np.ones(img.shape[:2])
        cable_mask[img[:, :, 1] < 0.4] = 0

        return self.transform(img.copy()).cuda(), points, cable_mask, angle

    def get_dist_cumsum(self, lst):
        lst_shifted = lst[1:]
        distances = np.linalg.norm(lst_shifted - lst[:-1], axis=1)
        # cumulative sum
        distances_cumsum = np.concatenate(([0], np.cumsum(distances)))
        return distances_cumsum[-1] / 1000

    def _trace(self, image, start_points, exact_path_len, endpoints=None, viz=False, model=None):    
        num_condition_points = self.trace_config.condition_len
        if start_points is None or len(start_points) < num_condition_points:
            raise ValueError(f"Need at least {num_condition_points} start points")
        path = [start_point for start_point in start_points]
        disp_img = (image.copy() * 255.0).astype(np.uint8)

        for iter in range(exact_path_len):
            condition_pixels = [p for p in path[-num_condition_points:]]
            
            crop, cond_pixels_in_crop, top_left = self.get_crop_and_cond_pixels(image, condition_pixels, center_around_last=True)
            ymin, xmin = np.array(top_left) - self.trace_config.crop_width

            model_input, _, cable_mask, angle = self.get_trp_model_input(crop, cond_pixels_in_crop, center_around_last=True)

            crop_eroded = cv2.erode((cable_mask).astype(np.uint8), np.ones((2, 2)), iterations=1)

            if viz:
                # cv2.imshow('model input', model_input.detach().cpu().numpy().transpose(1, 2, 0))
                # cv2.waitKey(1)
                plt.imsave(f'trace_test/model_input_{iter}.png', model_input.detach().cpu().numpy().transpose(1, 2, 0))

            model_output = model(model_input.unsqueeze(0)).detach().cpu().numpy().squeeze()
            model_output *= crop_eroded.squeeze()
            model_output = cv2.resize(model_output, (crop.shape[1], crop.shape[0]))

            # undo rotation if done in preprocessing
            M = cv2.getRotationMatrix2D((model_output.shape[1]/2, model_output.shape[0]/2), -angle*180/np.pi, 1)
            model_output = cv2.warpAffine(model_output, M, (model_output.shape[1], model_output.shape[0]))

            argmax_yx = np.unravel_index(model_output.argmax(), model_output.shape)# * np.array([crop.shape[0] / config.img_height, crop.shape[1] / config.img_width])

            # get angle of argmax yx
            global_yx = np.array([argmax_yx[0] + ymin, argmax_yx[1] + xmin]).astype(int)
            path.append(global_yx)

            if global_yx[0] > (image.shape[0] - self.y_buffer) or global_yx[0] < self.y_buffer or global_yx[1] > (image.shape[1] - self.x_buffer) or global_yx[1] < self.x_buffer: # Uncomment for triton
                return path, TraceEnd.EDGE

            if endpoints is not None:
                for endpoint in endpoints:
                    pix_dist = self.get_dist_cumsum(np.array(path))
                    if (abs(global_yx[0] - endpoint[0])) < self.ep_buffer and (abs(global_yx[1] - endpoint[1])) < self.ep_buffer:
                        if pix_dist > 2.5:
                            return path, TraceEnd.ENDPOINT
                    
            disp_img = cv2.circle(disp_img, (global_yx[1], global_yx[0]), 1, (0, 0, 255), 2)
            # add line from previous to current point
            if len(path) > 1:
                disp_img = cv2.line(disp_img, (path[-2][1], path[-2][0]), (global_yx[1], global_yx[0]), (0, 0, 255), 2)

            if viz:
                # cv2.imshow("disp_img", disp_img)
                # cv2.waitKey(1)
                plt.imsave(f'trace_test/disp_img_{iter}.png', disp_img)
                
            if len(path) > 10:
                p = np.array(path)
                for i in range(0, len(path)-20):
                    diff = np.linalg.norm(p[i:i+10] - p[-10:])
                    diffrev = np.linalg.norm(p[i:i+10] - p[-10:][::-1])
                    if diff < 10 or diffrev < 30:
                        return path[:-10], TraceEnd.RETRACE

        return path, TraceEnd.FINISHED

    def trace(self, img, prev_pixels, endpoints=None, path_len=20, viz=False, idx=0):
        pixels = self.center_pixels_on_cable(img, prev_pixels)
        for j in range(len(pixels)):
            cur_pixel = pixels[j][0]
            if cur_pixel[0] >= 0 and cur_pixel[1] >= 0 and cur_pixel[1] < img.shape[1] and cur_pixel[0] < img.shape[0]:
                start_idx = j
                break
            
        starting_points = self._get_evenly_spaced_points(pixels, self.trace_config.condition_len, start_idx, self.trace_config.cond_point_dist_px, img.shape, backward=False, randomize_spacing=False)

        if len(starting_points) < self.trace_config.condition_len:
            raise Exception("Not enough starting points:", len(starting_points), "Need: ", self.trace_config.condition_len)
            # return
        if img.max() > 1:
            img = (img / 255.0).astype(np.float32)
        
        plt.imshow(img)
        for pt in starting_points:
            plt.scatter(*pt[::-1])
        plt.show()

        spline, trace_end = self._trace(img, starting_points, exact_path_len=path_len, endpoints=endpoints, model=self.trace_model, viz=False)
        if viz:
            img_cp = (img.copy() * 255.0).astype(np.uint8)
            trace_viz = self.visualize_path(img_cp, spline.copy())
            plt.imsave(f'./trace_{idx}.png', trace_viz)
        spline = np.array(spline)
        spline = np.concatenate((starting_points, spline), axis=0)
        return np.array(spline), trace_end

    def visualize_path(self, img, path, black=False):
        img = img.copy()
        def color_for_pct(pct):
            return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
            # return (255*(1 - pct), 150, 255*pct) if not black else (0, 0, 0)
        for i in range(len(path) - 1):
            # if path is ordered dict, use below logic
            if not isinstance(path, OrderedDict):
                pt1 = tuple(path[i].astype(int))
                pt2 = tuple(path[i+1].astype(int))
            else:
                path_keys = list(path.keys())
                pt1 = path_keys[i]
                pt2 = path_keys[i + 1]
            cv2.line(img, pt1[::-1], pt2[::-1], color_for_pct(i/len(path)), 2 if not black else 5)
        return img

    def _is_uncovered_area_touching_before_idx(self, image, points, idx, endpoints):
        if idx is None or endpoints is None:
            return False
        image = image.copy()
        image[650:] = 0.0
        bs = 22
        for endpoint in endpoints:
            image[endpoint[0] - bs: endpoint[0]+bs, endpoint[1] - bs:endpoint[1] + bs] = 0

        uncovered_pixels = self._uncovered_pixels(image, points)
        if len(uncovered_pixels) < 30:
            return False
        image_draw = image.copy()
        for i in range(idx, len(points) - 1):
            cv2.line(image_draw, tuple(points[i])[::-1], tuple(points[i+1])[::-1], 0, 10)
        image_draw_mask = ((image_draw > 100) * 255).astype(np.uint8)

        _, labels, _, _ = cv2.connectedComponentsWithStats(image_draw_mask[..., 0], connectivity=8)
        uncovered_pixel_components = labels[uncovered_pixels[:, 0], uncovered_pixels[:, 1]]
        points_components = labels[points[:idx, 0], points[:idx, 1]]
        difference_matrix = uncovered_pixel_components[:, None] - points_components[None, ...]
        return np.sum(difference_matrix == 0) < 10

class AnalyticTracer(Tracer):
    def trace(self, img, prev_pixels, endpoints=None, path_len=20, viz=False, idx=0):
        pixels = self.center_pixels_on_cable(img, prev_pixels)
        for j in range(len(pixels)):
            cur_pixel = pixels[j][0]
            if cur_pixel[0] >= 0 and cur_pixel[1] >= 0 and cur_pixel[1] < img.shape[1] and cur_pixel[0] < img.shape[0]:
                start_idx = j
                break

        # starting_points = self._get_evenly_spaced_points(pixels, self.trace_config.condition_len,
        #                                                  start_idx, self.trace_config.cond_point_dist_px,
        #                                                  img.shape, backward=False, randomize_spacing=False)
        spline, trace_end = simple_uncertain_trace_single.trace(img, prev_pixels, None,
                                                                exact_path_len=path_len, endpoints=endpoints)
        if spline is None:
            spline = prev_pixels
        return np.array(spline), trace_end

if __name__ == '__main__': 
    trace_test = './trace_test'
    if os.path.exists(trace_test):
        shutil.rmtree(trace_test)
    os.mkdir(trace_test)

    tracer = Tracer()
    analytic_tracer = AnalyticTracer()
    eval_folder = '../data/real_data/real_data_for_tracer/test'
    for i, data in enumerate(np.sort(os.listdir(eval_folder))):
        if i == 0:
            continue
        test_data = np.load(os.path.join(eval_folder, data), allow_pickle=True).item()
        img = test_data['img']
        img_cp = img.copy()
        img[-130:, ...] = 0
        thresh_img = np.where(img[:,:,:3] > 100, 255, 0).astype('uint8')
        start_pixels = np.array(test_data['pixels'][0], dtype=np.uint32)[::-1]
        start_pixels, _ = analytic_tracer.trace(thresh_img, start_pixels, path_len=6, viz=False, idx=i)
        if len(start_pixels) < 5:
            continue
        spline = tracer.trace(img_cp, start_pixels, path_len=200, viz=True, idx=i)
    