import cv2
import os
import torch
from torchvision import transforms
from src.model import KeypointsGauss, ClassificationModel
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import time
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from config import *
import colorsys
from collections import OrderedDict
import shutil

os.environ["CUDA_VISIBLE_DEVICES"]="2"

REAL_WORLD_DICT = {'../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_9.npy': [(995, 274), (994, 267), (989, 260), (983, 256), (978, 250), (973, 245), (967, 238), (959, 230), (952, 226), (945, 220), (936, 214), (930, 211), (926, 206), (920, 202), (912, 198), (906, 193), (903, 190), (898, 187), (893, 184), (886, 179)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_0.npy': [(522, 208), (530, 216), (538, 220), (548, 221), (557, 227), (562, 232), (569, 232), (575, 232), (585, 235), (590, 236), (597, 237), (605, 236), (613, 236)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_7.npy': [(540, 41), (531, 48), (527, 53), (523, 62), (516, 71), (511, 81), (509, 92), (501, 108), (497, 116), (492, 128), (491, 136), (486, 141), (481, 150), (477, 161), (473, 168), (468, 175)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_43.npy': [(741, 474), (733, 469), (729, 463), (721, 461), (716, 457), (711, 454), (707, 450), (706, 449), (699, 443), (696, 439), (690, 437), (685, 431), (682, 430), (675, 417), (669, 413), (665, 404), (659, 400)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_44.npy': [(783, 392), (783, 383), (783, 372), (780, 362), (776, 354), (773, 346), (770, 334), (769, 324), (767, 319), (764, 311), (762, 308), (757, 301), (755, 297), (751, 293), (746, 284), (744, 278), (740, 271), (733, 262), (730, 259)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_38.npy': [(503, 415), (514, 422), (529, 430), (538, 431), (552, 433), (563, 439), (578, 443), (592, 443), (604, 443), (626, 443), (638, 443), (646, 444), (656, 444), (667, 444), (686, 440), (701, 439), (718, 431)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_36.npy': [(465, 503), (469, 503), (474, 503), (475, 503), (478, 504), (483, 504), (489, 504), (495, 504), (500, 500), (511, 500), (520, 498), (525, 498), (525, 498), (530, 498), (536, 498), (543, 496), (548, 492), (553, 490), (557, 488), (559, 488), (565, 485), (568, 482), (575, 480), (584, 476), (588, 473), (592, 470), (595, 467), (598, 464), (601, 462), (605, 460), (610, 457), (613, 456)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_31.npy': [(412, 257), (420, 247), (424, 235), (429, 226), (434, 220), (437, 215), (440, 210), (447, 201), (451, 193), (454, 191), (459, 186), (468, 181), (474, 175), (485, 170), (490, 165), (497, 158), (504, 157), (512, 152), (519, 148), (529, 146), (536, 141), (543, 137), (545, 137), (552, 134), (563, 130), (571, 127), (582, 127), (587, 122), (597, 119), (608, 119), (616, 115)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_20.npy': [(694, 512), (683, 504), (671, 499), (662, 494), (655, 490), (645, 488), (638, 479), (628, 473), (618, 467), (612, 462), (602, 455), (599, 448), (587, 438), (584, 431), (579, 423), (576, 418), (566, 406), (563, 395), (560, 388), (555, 375), (553, 371), (552, 361), (552, 353)], '../data/real_data/real_world_images/oct_17_figure_eight_overhand/color_27.npy': [(432, 275), (441, 278), (450, 279), (459, 279), (466, 279), (477, 279), (489, 276), (492, 275), (501, 275), (510, 272), (519, 271), (528, 265), (537, 263), (543, 259), (554, 254), (562, 251)]}

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--checkpoint_file_name', type=str, default='')
parser.add_argument('--trace_full_cable', action='store_true', default=False)
parser.add_argument('--real_world_trace', action='store_true', default=False)
parser.add_argument('--eval_real', action='store_true', default=False)

flags = parser.parse_args()

experiment_time = time.strftime("%Y%m%d-%H%M%S")
checkpoint_path = flags.checkpoint_path
checkpoint_file_name = flags.checkpoint_file_name
trace_full_cable = flags.trace_full_cable
real_world_trace = flags.real_world_trace

def visualize_heatmap_on_image(img, heatmap):
    argmax = list(np.unravel_index(np.argmax(heatmap), heatmap.shape))[::-1]
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

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
    return np.array(processed_pixels)[:, ::-1]

if checkpoint_path == '':
    raise ValueError("--checkpoint_path must be specified")

min_loss, min_checkpoint = 100000, None
if checkpoint_file_name == '':
    # choose the one with the lowest loss
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pth"):
            loss = float(file.split('_')[-1].split('.')[-2])
            if loss < min_loss:
                min_loss = loss
                min_checkpoint = os.path.join(checkpoint_path, file)
    checkpoint_file_name = min_checkpoint
else:
    checkpoint_file_name = os.path.join(checkpoint_path, checkpoint_file_name)

# laod up all the parameters from the checkpoint
config = load_config_class(checkpoint_path)
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
        ymin, xmin = np.array(top_left) - test_dataset.crop_width
        model_input, _, cable_mask, angle = test_dataset.get_trp_model_input(crop, cond_pixels_in_crop, center_around_last=True, is_real_example=real)

        crop_eroded = cv2.erode((cable_mask).astype(np.uint8), np.ones((2, 2)), iterations=1)

        model_output = model(model_input.unsqueeze(0)).detach().cpu().numpy().squeeze()
        model_output *= crop_eroded.squeeze()
        model_output = cv2.resize(model_output, (crop.shape[1], crop.shape[0]))

        # undo rotation if done in preprocessing
        M = cv2.getRotationMatrix2D((model_output.shape[1]/2, model_output.shape[0]/2), -angle*180/np.pi, 1)
        model_output = cv2.warpAffine(model_output, M, (model_output.shape[1], model_output.shape[0]))
        argmax_yx = np.unravel_index(model_output.argmax(), model_output.shape)

        # get angle of argmax yx
        global_yx = np.array([argmax_yx[0] + ymin, argmax_yx[1] + xmin]).astype(int)
        path.append(global_yx)

        if viz:
            plt.imshow(visualize_heatmap_on_image(crop, model_output))
            plt.savefig("vis_heatmap.png")

        disp_img = cv2.circle(disp_img, (global_yx[1], global_yx[0]), 1, (0, 0, 255), 2)
        # add line from previous to current point
        if len(path) > 1:
            disp_img = cv2.line(disp_img, (path[-2][1], path[-2][0]), (global_yx[1], global_yx[0]), (0, 0, 255), 2)
    return path

def visualize_path(img, path, black=False):
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
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

expt_name = os.path.normpath(checkpoint_path).split(os.sep)[-1]
output_folder_name = f'preds/preds_{expt_name}'
if not os.path.exists('preds'):
    os.mkdir('preds')
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)

if not flags.trace_full_cable:
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

# model
keypoints_models = []
if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
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
    real_paths  = ['../data/real_data/real_data_for_tracer/test']
elif config.expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
    if config.crop_width == 10:
        real_paths = ['../data/real_data/under_over_crossing_real_test_set']

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

if expt_type == ExperimentTypes.TRACE_PREDICTION and trace_full_cable:
    if not real_world_trace:
        image_folder = '/home/kaushiks/hulk-keypoints/processed_sim_data/trace_dataset_hard_2/test'
        images = os.listdir(image_folder)
    else:
        image_folder = ''
        REAL_WORLD_DICT = {}
        even_more_real_world = '../data/real_data/real_data_for_tracer/test'
        for file in os.listdir(even_more_real_world):
                file_path = os.path.join(even_more_real_world, file)
                spline = np.load(file_path, allow_pickle=True).item()['pixels']
                REAL_WORLD_DICT[file_path] = spline
        images = list(REAL_WORLD_DICT.keys())
        images.sort()

    for i, image in enumerate(images):
        if not real_world_trace:
            loaded_img = np.load(os.path.join(image_folder, image), allow_pickle=True).item()
            img = loaded_img['img'][:, :, :3]
            pixels = loaded_img['pixels']
        else:
            if image not in REAL_WORLD_DICT:
                continue
            if image.endswith('.npy'):
                img = np.load(os.path.join(image_folder, image), allow_pickle=True)
                if 'real_data_for_tracer' in image:
                    img = img.item()['img'][:, :, :3]
            else:
                img = cv2.imread(os.path.join(image_folder, image))
            pixels = center_pixels_on_cable(img, REAL_WORLD_DICT[image])[..., ::-1]

        # now get starting points
        for j in range(len(pixels)):
            cur_pixel = pixels[j][0]
            if cur_pixel[0] >= 0 and cur_pixel[1] >= 0 and cur_pixel[1] < img.shape[0] and cur_pixel[0] < img.shape[1]:
                start_idx = j
                break

        try:
            starting_points = test_dataset._get_evenly_spaced_points(pixels, config.condition_len, start_idx + 1, config.cond_point_dist_px, img.shape, backward=False, randomize_spacing=False)
        except:
            starting_points = []
        if len(starting_points) < config.condition_len:
            print("Not enough starting points")
            continue

        # normalize image from 0 to 1
        if img.max() > 1:
            img = (img / 255.0).astype(np.float32)

        spline = trace(img, starting_points, exact_path_len=80, model=keypoints_models[0], viz=False)

        img_cp = (img.copy() * 255.0).astype(np.uint8)
        trace_viz = visualize_path(img_cp, spline)
        plt.imsave(f'{trace_folder_name}/trace_{i}.png', trace_viz)

else:
    preds = []
    gts = []
    total = 0
    class_thresholds = np.linspace(0.0, 1.0, 21)
    if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
        hits = [0 for _ in range(len(class_thresholds))]
    else:
        hits = 0
    for i, f in enumerate(test_dataset):
        print(i)
        f = list(f)
        img_t = f[0]
        if (len(img_t.shape) < 4):
            img_t = img_t.unsqueeze(0)
        plt.clf()

        heatmaps = []
        for j, prediction in enumerate(predictions):
            output = prediction.predict(img_t[0])

        if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
            pred = output.detach().squeeze().cpu().numpy()
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
            cv2.imwrite(save_path_og, input_img_np * 255.0)
        
        elif is_point_pred(expt_type):
            argmax_yx = np.unravel_index(np.argmax(output.detach().cpu().numpy()[0, 0, ...]), output.detach().cpu().numpy()[0, 0, ...].shape)
            output_yx = np.unravel_index(np.argmax(f[1][0].detach().cpu().numpy()), f[1][0].detach().cpu().numpy().shape)
            output_heatmap = output.detach().cpu().numpy()[0, 0, ...]
            output_image = f[0][0:3, ...].detach().cpu().numpy().transpose(1,2,0)
            output_image[:, :, 2] = output_heatmap 
            output_image = output_image.copy()

            vis_image = visualize_heatmap_on_image(img_t[0].squeeze().detach().cpu().numpy().transpose(1,2,0), output_heatmap)
            vis_image = cv2.circle(vis_image, (output_yx[1], output_yx[0]), 1, (0, 255, 255), -1)
            
            overlay = output_image
            plt.imshow(overlay)

            save_path = os.path.join(failure_folder_name, f'output_img_{i}.png')
            if np.linalg.norm((np.array(argmax_yx) - np.array(output_yx)), 2) < 5: #*(config.img_height/96.0):
                hits += 1
                save_path = os.path.join(success_folder_name, f'output_img_{i}.png')
            plt.savefig(save_path)

        # check if the gt at argmax is 1
        total += 1

    if expt_type == ExperimentTypes.CLASSIFY_OVER_UNDER:
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