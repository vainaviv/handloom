import numpy as np 
import os
import cv2

train_path = '/home/vainavi/tusk/data/real_data_for_tracer/train'
test_path = '/home/vainavi/tusk/data/real_data_for_tracer/test'
# os.mkdir('./real_data_for_tracer_viz/')
# os.mkdir('./real_data_for_tracer_viz/train')
# os.mkdir('./real_data_for_tracer_viz/test')

for i, f in enumerate(np.sort(os.listdir(train_path))):
    datapath = os.path.join('./real_data_for_tracer/train', f)
    img = np.load(datapath, allow_pickle=True).item()['img']
    filename = './real_data_for_tracer_viz/train/' + str(i) + ".png"
    cv2.imwrite(filename, img)

for i, f in enumerate(np.sort(os.listdir(test_path))):
    datapath = os.path.join('./real_data_for_tracer/test', f)
    img = np.load(datapath, allow_pickle=True).item()['img']
    filename = './real_data_for_tracer_viz/test/' + str(i) + ".png"
    cv2.imwrite(filename, img)