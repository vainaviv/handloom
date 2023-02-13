import numpy as np
import cv2
import os
import pickle as pkl

glob_points = []
image = None
REAL_WORLD_DICT = {'/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_9.npy': [(995, 274), (994, 267), (989, 260), (983, 256), (978, 250), (973, 245), (967, 238), (959, 230), (952, 226), (945, 220), (936, 214), (930, 211), (926, 206), (920, 202), (912, 198), (906, 193), (903, 190), (898, 187), (893, 184), (886, 179)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_0.npy': [(522, 208), (530, 216), (538, 220), (548, 221), (557, 227), (562, 232), (569, 232), (575, 232), (585, 235), (590, 236), (597, 237), (605, 236), (613, 236)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_7.npy': [(540, 41), (531, 48), (527, 53), (523, 62), (516, 71), (511, 81), (509, 92), (501, 108), (497, 116), (492, 128), (491, 136), (486, 141), (481, 150), (477, 161), (473, 168), (468, 175)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_43.npy': [(741, 474), (733, 469), (729, 463), (721, 461), (716, 457), (711, 454), (707, 450), (706, 449), (699, 443), (696, 439), (690, 437), (685, 431), (682, 430), (675, 417), (669, 413), (665, 404), (659, 400)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_44.npy': [(783, 392), (783, 383), (783, 372), (780, 362), (776, 354), (773, 346), (770, 334), (769, 324), (767, 319), (764, 311), (762, 308), (757, 301), (755, 297), (751, 293), (746, 284), (744, 278), (740, 271), (733, 262), (730, 259)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_38.npy': [(503, 415), (514, 422), (529, 430), (538, 431), (552, 433), (563, 439), (578, 443), (592, 443), (604, 443), (626, 443), (638, 443), (646, 444), (656, 444), (667, 444), (686, 440), (701, 439), (718, 431)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_36.npy': [(465, 503), (469, 503), (474, 503), (475, 503), (478, 504), (483, 504), (489, 504), (495, 504), (500, 500), (511, 500), (520, 498), (525, 498), (525, 498), (530, 498), (536, 498), (543, 496), (548, 492), (553, 490), (557, 488), (559, 488), (565, 485), (568, 482), (575, 480), (584, 476), (588, 473), (592, 470), (595, 467), (598, 464), (601, 462), (605, 460), (610, 457), (613, 456)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_31.npy': [(412, 257), (420, 247), (424, 235), (429, 226), (434, 220), (437, 215), (440, 210), (447, 201), (451, 193), (454, 191), (459, 186), (468, 181), (474, 175), (485, 170), (490, 165), (497, 158), (504, 157), (512, 152), (519, 148), (529, 146), (536, 141), (543, 137), (545, 137), (552, 134), (563, 130), (571, 127), (582, 127), (587, 122), (597, 119), (608, 119), (616, 115)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_20.npy': [(694, 512), (683, 504), (671, 499), (662, 494), (655, 490), (645, 488), (638, 479), (628, 473), (618, 467), (612, 462), (602, 455), (599, 448), (587, 438), (584, 431), (579, 423), (576, 418), (566, 406), (563, 395), (560, 388), (555, 375), (553, 371), (552, 361), (552, 353)], '/home/kaushiks/hulk-keypoints/real_world_images/oct_17_figure_eight_overhand/color_27.npy': [(432, 275), (441, 278), (450, 279), (459, 279), (466, 279), (477, 279), (489, 276), (492, 275), (501, 275), (510, 272), (519, 271), (528, 265), (537, 263), (543, 259), (554, 254), (562, 251)]}


def click_points_on_real_image(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        glob_points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

def get_real_world_image_points(image_path):
    global image
    image = np.load(image_path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_points_on_real_image)
    while(1):
        cv2.imshow('image', image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return glob_points


if __name__ == '__main__':
    image_path = '/home/vainavi/hulk-keypoints/eval_imgs'
    all_images = os.listdir(image_path)
    image_annot_dict = {}
    for img_name in all_images:
        if 'depth' in img_name:
            continue
        file_path = os.path.join(image_path, img_name)
        get_real_world_image_points(file_path)
        image_annot_dict[file_path] = glob_points
        glob_points = []
        print(image_annot_dict)