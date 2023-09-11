import enum
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

class Graspability():
    def __init__(self):
        self.radius = 10 #TODO: tune
        self.i = 0

    def find_pixel_point_graspability(self, point, crossings, img):
            # total_points = 4*(self.radius**2)
            crop = img[point[0]-self.radius:point[0]+self.radius, point[1]-self.radius:point[1]+self.radius, :]
            # cv2.imwrite(f'/home/vainavi/hulk-keypoints/triton_trace_files/crops/crop_{self.i}.png', crop)
            self.i += 1
            crop_mask = (crop[:, :, 0] > 100)    
            viz_mask = np.array([crop_mask, crop_mask, crop_mask]).transpose(1,2,0) * 255.0
            # cv2.imwrite(f'/home/vainavi/hulk-keypoints/triton_trace_files/crops/crop_mask_{self.i}.png', viz_mask)
            penalty = 0
            all_crossing_locs = []
            for crossing in crossings:
                all_crossing_locs.append(crossing['loc'])
            all_crossing_locs = np.array(all_crossing_locs)
            # if near a crossing, penalize
            if np.min(np.linalg.norm(point[None, :] - all_crossing_locs, axis=-1)) < 15:
                penalty = 100
            return  np.sum(crop_mask) + penalty

if __name__ == '__main__':
    # parse command line flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_index', type=str, default='')

    flags = parser.parse_args()
    data_index = flags.data_index 

    if data_index == '':
        raise Exception('Please provide the file number (e.g. --data_index 00000) as a command-line argument!')

    data_path = f"../data/real_data/real_data_for_tracer/train/{data_index}.npy"
    test_data = np.load(data_path, allow_pickle=True).item()
    g = Graspability()
    img = test_data['img']
    for i in range(20):
        point = test_data['pixels'][i]
        g.find_pixel_point_graspability(point, img)
        # cv2.imwrite(f'./crops/crop_{i}.png', img[point[1]-g.radius:point[1]+g.radius, point[0]-g.radius:point[0]+g.radius, :])

    # tkd._set_data(test_data['img'], test_data['pixels'][:10], test_data['pixels'])
    # print(data_path)
    # print()
    # tkd.perception_pipeline()
    # tkd._visualize_full()
    # tkd._visualize_crossings()
    # if tkd.knot:
    #     print()
    #     print("knot: ", tkd.knot)
    #     print("knot confidence: ", tkd._get_knot_confidence())
    #     tkd._visualize_knot()