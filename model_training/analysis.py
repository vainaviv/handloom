import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.model import KeypointsGauss, ClassificationModel
from src.dataset import KeypointsDataset, transform, gauss_2d_batch, bimodal_gauss, get_gauss
from src.prediction import Prediction
from datetime import datetime, time
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve2d
import argparse
from config import *
import colorsys
from collections import OrderedDict
from annot_real_img import REAL_WORLD_DICT 
import pickle as pkl
import shutil

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def visualize_heatmap_on_image(img, heatmap):
    argmax = list(np.unravel_index(np.argmax(heatmap), heatmap.shape))[::-1]
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    # plot argmax location as white dot
    cv2.circle(cam, argmax, 1, (1,1,1), -1)
    return cam

def center_pixels_on_cable(image, pixels):
    # for each pixel, find closest pixel on cable
    image_mask = image[:, :, 0] > 100
    # erode white pixels
    kernel = np.ones((2,2),np.uint8)
    image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
    white_pixels = np.argwhere(image_mask)

    processed_pixels = []
    for pixel in pixels:
        # find closest pixel on cable
        distances = np.linalg.norm(white_pixels - pixel[::-1], axis=1)
        closest_pixel = white_pixels[np.argmin(distances)]
        processed_pixels.append([closest_pixel])

    # plot pixels on cable
    # plt.imshow(image_mask)
    # for pixel in processed_pixels:
    #     plt.scatter(pixel[0][1], pixel[0][0], c='r')
    # plt.show()

    return np.array(processed_pixels)[:, ::-1]

# parse command line flags
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--checkpoint_file_name', type=str, default='')
parser.add_argument('--trace_if_trp', action='store_true', default=False)
parser.add_argument('--real_world_trace', action='store_true', default=False)
parser.add_argument('--eval_real', action='store_true', default=False)

flags = parser.parse_args()

experiment_time = time.strftime("%Y%m%d-%H%M%S")
checkpoint_path = flags.checkpoint_path
checkpoint_file_name = flags.checkpoint_file_name
trace_if_trp = flags.trace_if_trp
real_world_trace = flags.real_world_trace

if checkpoint_path == '':
    raise ValueError("--checkpoint_path must be specified")

min_loss, min_checkpoint = 100000, None
if checkpoint_file_name == '':
    # choose the one with the lowest loss
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pth"):
            # file is structured as "..._loss.pth", extract loss
            loss = float(file.split('_')[-1].split('.')[-2])
            if loss < min_loss:
                min_loss = loss
                min_checkpoint = os.path.join(checkpoint_path, file)
    checkpoint_file_name = min_checkpoint
else:
    checkpoint_file_name = os.path.join(checkpoint_path, checkpoint_file_name)

# laod up all the parameters from the checkpoint
config = load_config_class(checkpoint_path) #TRCR32_CL3_12_PL1_MED3_UNet34_B64_OS_RotCond_Hard2_WReal()
expt_type = config.expt_type

print("Using checkpoint: ", checkpoint_file_name)
print("Loaded config: ", config)

def trace(image, start_points, viz=True, exact_path_len=None, model=None):    
    viz = True
    num_condition_points = config.condition_len
    if start_points is None or len(start_points) < num_condition_points:
        raise ValueError(f"Need at least {num_condition_points} start points")
    path = [start_point for start_point in start_points]
    disp_img = (image.copy() * 255.0).astype(np.uint8)

    for iter in range(exact_path_len):
        condition_pixels = [p for p in path[-num_condition_points:]]

        crop, cond_pixels_in_crop, top_left = test_dataset.get_crop_and_cond_pixels(image, condition_pixels, center_around_last=True)
        # print('cond pixels', cond_pixels_in_crop)
        ymin, xmin = np.array(top_left) - test_dataset.crop_width
        model_input, _, cable_mask, angle = test_dataset.get_trp_model_input(crop, cond_pixels_in_crop, center_around_last=True, is_real_example=real)

        crop_eroded = cv2.erode((cable_mask).astype(np.uint8), np.ones((2, 2)), iterations=1)
        # print("Model input prep time: ", time.time() - tm)

        if True:
            # cv2.imshow('model input', model_input.detach().cpu().numpy().transpose(1, 2, 0))
            # cv2.waitKey(1)
            plt.imsave(f'./model_inputs/model_input_{iter}.png', model_input.detach().cpu().numpy().transpose(1, 2, 0))

        model_output = model(model_input.unsqueeze(0)).detach().cpu().numpy().squeeze()
        model_output *= crop_eroded.squeeze()
        model_output = cv2.resize(model_output, (crop.shape[1], crop.shape[0]))

        # undo rotation if done in preprocessing
        M = cv2.getRotationMatrix2D((model_output.shape[1]/2, model_output.shape[0]/2), -angle*180/np.pi, 1)
        model_output = cv2.warpAffine(model_output, M, (model_output.shape[1], model_output.shape[0]))

        # mask model output by disc of radius COND_POINT_DIST_PX around the last condition pixel
        tolerance = 2
        last_condition_pixel = cond_pixels_in_crop[-1]

        argmax_yx = np.unravel_index(model_output.argmax(), model_output.shape)# * np.array([crop.shape[0] / config.img_height, crop.shape[1] / config.img_width])

        # get angle of argmax yx
        global_yx = np.array([argmax_yx[0] + ymin, argmax_yx[1] + xmin]).astype(int)
        path.append(global_yx)

        # print("global_yx: ", global_yx)

        if viz:
            # plt.scatter(argmax_yx[1], argmax_yx[0], c='r')
            # plt.imshow(crop)
            # plt.show()

            # cv2.imshow('heatmap on crop', visualize_heatmap_on_image(crop, model_output))
            # cv2.waitKey(1)

            plt.imshow(visualize_heatmap_on_image(crop, model_output))
            plt.savefig("vis_heatmap.png")

            # plt.scatter(global_yx[1], global_yx[0], c='r')
            # plt.imshow(image)
            # plt.show()

        disp_img = cv2.circle(disp_img, (global_yx[1], global_yx[0]), 1, (0, 0, 255), 2)
        # add line from previous to current point
        if len(path) > 1:
            disp_img = cv2.line(disp_img, (path[-2][1], path[-2][0]), (global_yx[1], global_yx[0]), (0, 0, 255), 2)
        # plt.imsave(f'preds/disp_img_{i}.png', disp_img)

        if viz:
            # cv2.imshow("disp_img", disp_img)
            # cv2.waitKey(1)

            pass
    return path

def visualize_path(img, path, black=False):
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
            # print(pt1, pt2)
        cv2.line(img, pt1[::-1], pt2[::-1], color_for_pct(i/len(path)), 2 if not black else 5)
    return img

expt_name = os.path.normpath(checkpoint_path).split(os.sep)[-1]
output_folder_name = f'preds/preds_{expt_name}'
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)

if not flags.trace_if_trp:
    success_folder_name = os.path.join(output_folder_name, 'success')
    if os.path.exists(success_folder_name):
        shutil.rmtree(success_folder_name)
    os.mkdir(success_folder_name)
    failure_folder_name = os.path.join(output_folder_name, 'failure')
    if os.path.exists(failure_folder_name):
        shutil.rmtree(failure_folder_name)
    os.mkdir(failure_folder_name)
else:
    trace_folder_name = os.path.join(output_folder_name, 'full_traces')
    if os.path.exists(trace_folder_name):
        shutil.rmtree(trace_folder_name)
    os.mkdir(trace_folder_name)

# cuda
use_cuda = torch.cuda.is_available()
if use_cuda:
    # torch.cuda.set_device(2)
    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    pass

# model
keypoints_models = []
# for model_ckpt in model_ckpts:
if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
    keypoints = ClassificationModel(num_classes=config.classes, img_height=config.img_height, img_width=config.img_width, resnet_type=config.resnet_type, channels=3)
elif is_point_pred(expt_type):
    keypoints = KeypointsGauss(1, img_height=config.img_height, img_width=config.img_width, channels=3, resnet_type=config.resnet_type, pretrained=config.pretrained)

keypoints_models.append(keypoints)
if use_cuda:
    for keypoints in keypoints_models:
        keypoints = keypoints.cuda()

keypoints.load_state_dict(torch.load(checkpoint_file_name))

predictions = []

for keypoints in keypoints_models:
    prediction = Prediction(keypoints, config.num_keypoints, config.img_height, config.img_width, use_cuda)
    predictions.append(prediction)

transform = transform = transforms.Compose([
    transforms.ToTensor()
])
real = False
if flags.eval_real:
    real_paths = []
    for dir in config.dataset_dir:
        real_path = os.path.join(dir, 'real_test')
        if os.path.exists(real_path):
            real=True
            real_paths.append(real_path)
    for i, real_bool in enumerate(config.dataset_real):
        if real_bool:
            real_paths.append(os.path.join(config.dataset_dir[i], 'test'))
            real = True

real = True
if config.expt_type == ExperimentTypes.TRACE_PREDICTION:
    real_paths  = ['/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test']
elif config.expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or config.expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
    if config.crop_width == 10:
        real_paths = ['/home/vainavi/hulk-keypoints/processed_sim_data/under_over_crossing_set2/real_test']
    elif config.crop_width == 16:
        real_paths = ['/home/kaushiks/hulk-keypoints/processed_sim_data/under_over_REAL_centered_32px/test']

if real:
    test_dataset = KeypointsDataset(real_paths, 
                                    transform,
                                    augment=False,
                                    real_only=True,
                                    config=config)
else:
    test_dataset = KeypointsDataset(['%s/test'%dir for dir in config.dataset_dir],
                                    transform, 
                                    augment=False, 
                                    config=config)

if expt_type == ExperimentTypes.TRACE_PREDICTION and trace_if_trp:
    if not real_world_trace:
        image_folder = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2/test'
        images = os.listdir(image_folder)
    else:
        image_folder = ''
        # trace_annots = '/home/vainavi/hulk-keypoints/logs/real_trace_test.pkl'
        # with open(trace_annots, 'rb') as pickle_file:
        #     more_real_world = pkl.load(pickle_file)
        #     for key in more_real_world.keys():
        #         REAL_WORLD_DICT[key] = more_real_world[key]
        REAL_WORLD_DICT = {}
        even_more_real_world = '/home/vainavi/hulk-keypoints/real_data/real_data_for_tracer/test'
        for file in os.listdir(even_more_real_world):
                file_path = os.path.join(even_more_real_world, file)
                spline = np.load(file_path, allow_pickle=True).item()['pixels']
                REAL_WORLD_DICT[file_path] = spline
        # REAL_WORLD_DICT = {k: v for k, v in REAL_WORLD_DICT.items() if k.endswith('00102.npy')}
        images = list(REAL_WORLD_DICT.keys())
        images.sort()

        # filter REAL_WORLD_DICT into only the image I care about
        # REAL_WORLD_DICT = {k: v for k, v in REAL_WORLD_DICT.items() if k.endswith('00104.npy')}
        # print(REAL_WORLD_DICT)

    for i, image in enumerate(images):
        # if images[i] == '/home/vainavi/hulk-keypoints/eval_imgs/00004.png':
        #     continue
        # if int(images[i][-9:-4]) < 100:
        #     continue
        if image not in REAL_WORLD_DICT:
            continue
        # if i < :
        #     continue
        # print(os.path.join(image_folder, image))
        if not real_world_trace:
            loaded_img = np.load(os.path.join(image_folder, image), allow_pickle=True).item()
            img = loaded_img['img'][:, :, :3]
            pixels = loaded_img['pixels']
        else:
            if image.endswith('.npy'):
                img = np.load(os.path.join(image_folder, image), allow_pickle=True)
                if 'real_data_for_tracer' in image:
                    img = img.item()['img'][:, :, :3]
            else:
                img = cv2.imread(os.path.join(image_folder, image))
            pixels = center_pixels_on_cable(img, REAL_WORLD_DICT[image])[..., ::-1]

        # now get starting points
        # print(pixels, img.shape)
        for j in range(len(pixels)):
            cur_pixel = pixels[j][0]
            if cur_pixel[0] >= 0 and cur_pixel[1] >= 0 and cur_pixel[1] < img.shape[0] and cur_pixel[0] < img.shape[1]:
                start_idx = j
                break

        # print(start_idx, pixels)
        try:
            starting_points = test_dataset._get_evenly_spaced_points(pixels, config.condition_len, start_idx + 1, config.cond_point_dist_px, img.shape, backward=False, randomize_spacing=False)
        except:
            starting_points = []
        if len(starting_points) < config.condition_len:
            print("Not enough starting points")
            continue

        # plt.imshow(img)
        # for pt in starting_points:
        #     plt.scatter(pt[1], pt[0], c='r')
        # plt.show()

        # normalize image from 0 to 1
        if img.max() > 1:
            img = (img / 255.0).astype(np.float32)

        spline = trace(img, starting_points, exact_path_len=80, model=keypoints_models[0], viz=False)
        # plt.imshow(img)
        # for pt in spline:
        #     plt.scatter(pt[1], pt[0], c='r')
        # plt.show()

        img_cp = (img.copy() * 255.0).astype(np.uint8)
        trace_viz = visualize_path(img_cp, spline)
        plt.imsave(f'{trace_folder_name}/trace_{i}.png', trace_viz)

else:
    preds = []
    gts = []
    total = 0
    class_thresholds = np.linspace(0.0, 1.0, 21)
    hits = [0 for _ in range(len(class_thresholds))]
    for i, f in enumerate(test_dataset):
        print(i)
        f = list(f)
        img_t = f[0]
        if (len(img_t.shape) < 4):
            img_t = img_t.unsqueeze(0)

        plt.clf()
        # plt.imshow(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0))
        # plt.savefig(f'{output_folder_name}/input_img_{i}.png'.format(i=i))

        # plot one heatmap for each model with matplotlib
        # plt.figure()

        # if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
        #     pass
        # else: 
        #     input_img_np = img_t.detach().cpu().numpy()[0, 0:3, ...]
        #     plt.clf()
        #     plt.imshow(input_img_np.transpose(1,2,0))
        #     plt.savefig(f'{output_folder_name}/input_img_{i}.png')

        heatmaps = []
        # create len(predictions) subplots
        for j, prediction in enumerate(predictions):
            output = prediction.predict(img_t[0])

        if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
            pred = output.detach().squeeze().cpu().numpy()
            if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
                pred = np.argmax(pred)
            preds.append(pred)
            gt = f[1].detach().cpu().numpy().item()
            gts.append(gt)
            plt.clf()
            plt.title(f'Pred: {preds[-1]}, GT: {gts[-1]}')
            plt.imshow(img_t[0].detach().cpu().numpy().transpose(1, 2, 0))
            save_path = os.path.join(failure_folder_name, f'output_img_{i}.png')
            save_path_og = os.path.join(failure_folder_name, f'output_img_{i}_og.png')
            for k, thresh in enumerate(class_thresholds):
                if pred < thresh:
                    pred_rounded = 0
                elif pred >= thresh and pred < (1 + thresh):
                    pred_rounded = 1
                else:
                    pred_rounded = 2
                if pred_rounded == int(gt):
                    hits[k] += 1
                    save_path = os.path.join(success_folder_name, f'output_img_{i}.png')
                    save_path_og = os.path.join(success_folder_name, f'output_img_{i}_og.png')
            plt.savefig(save_path)
            plt.clf()
            input_img_np = img_t.detach().cpu().numpy()[0, 1, ...]
            # plt.imshow(input_img_np * 255.0)
            # plt.savefig(save_path_og)
            cv2.imwrite(save_path_og, input_img_np * 255.0)
        
        elif is_point_pred(expt_type):
            argmax_yx = np.unravel_index(np.argmax(output.detach().cpu().numpy()[0, 0, ...]), output.detach().cpu().numpy()[0, 0, ...].shape)
            output_yx = np.unravel_index(np.argmax(f[1][0].detach().cpu().numpy()), f[1][0].detach().cpu().numpy().shape)
            output_heatmap = output.detach().cpu().numpy()[0, 0, ...]
            output_image = f[0][0:3, ...].detach().cpu().numpy().transpose(1,2,0)
            # output_image[:, :, 2] = output_heatmap * 255
            output_image[:, :, 2] = output_heatmap 
            output_image = output_image.copy()

            vis_image = visualize_heatmap_on_image(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0), output_heatmap)
            vis_image = cv2.circle(vis_image, (output_yx[1], output_yx[0]), 1, (0, 255, 255), -1)
            
            # output_image = (output_image * 255.0).astype(np.uint8)
            overlay = output_image
            #adding white circle for argmax of cage point prediction because gaussian heatmap is too uncertain
            if(expt_type == ExperimentTypes.CAGE_PREDICTION):
                cv2.circle(output_image, (argmax_yx[1], argmax_yx[0]), 2, (255, 255, 255), -1)
            plt.imshow(overlay)

            save_path = os.path.join(failure_folder_name, f'output_img_{i}.png')
            if np.linalg.norm((np.array(argmax_yx) - np.array(output_yx)), 2) < 5: #*(config.img_height/96.0):
                hits += 1
                save_path = os.path.join(success_folder_name, f'output_img_{i}.png')
            plt.savefig(save_path)

        # check if the gt at argmax is 1
        total += 1

    if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER or expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER_NONE:
        # calculate auc score
        import sklearn.metrics as metrics
        fpr, tpr, thresholds = metrics.roc_curve(gts, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("Classification AUC:", auc)
        for i, hit in enumerate(hits):
            thresh = class_thresholds[i]
            print(f"Classification Accuracy for {thresh}:", hit/total)
    elif is_point_pred(expt_type):
        print("Mean within threshold accuracy:", hits/total)