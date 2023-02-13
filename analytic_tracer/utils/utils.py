import numpy as np
import matplotlib.pyplot as plt
import cv2
import colorsys
import matplotlib.pyplot as plt
from collections import deque, OrderedDict

WORKSPACE_MIN_X = 0
WORKSPACE_MAX_X = -1
WORKSPACE_MIN_Y = 0
WORKSPACE_MAX_Y = 600

def get_edge_mask(img, x_min=None, x_max=None, y_min=None, y_max=None, addtl_padding=20):
    x_min = x_min if x_min is not None else WORKSPACE_MIN_X
    x_max = x_max if x_max is not None else WORKSPACE_MAX_X
    y_min = y_min if y_min is not None else WORKSPACE_MIN_Y
    y_max = y_max if y_max is not None else WORKSPACE_MAX_Y

    copied_img = img.copy()
    copied_img[y_min+addtl_padding:y_max-addtl_padding, x_min+addtl_padding:x_max-addtl_padding, :] = 0
    return (copied_img > 0)[:, :, 0]

def get_all_edge_candidates(img, x_min=None, x_max=None, y_min=None, y_max=None):
    edge_mask = get_edge_mask(img, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, addtl_padding=1)
    ret = cv2.connectedComponentsWithStats(edge_mask.astype(np.uint8), 4, cv2.CV_32S)
    num_components, labels, stats, centroids = ret
    successors = []
    for i in range(1, num_components):
        successors.append(closest_nonzero_pixel(centroids[i][::-1], edge_mask))
    return np.array(successors)

def cable_inaccessible(img, visited):
    visited_mask = np.zeros(img.shape[:2], np.uint8)
    for pt in visited.keys():
        visited_mask[pt[0], pt[1]] = 1
    visited_mask = (cv2.dilate(visited_mask, np.ones((24, 24), np.uint8)) > 0)
    img_dilated = (img > 0).astype(np.uint8)[:, :, 0] > 0
    untraversed_cable = cv2.dilate((img_dilated & ~visited_mask).astype(np.uint8), np.ones((24, 24), np.uint8))
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(untraversed_cable, 4, cv2.CV_32S)
    num_components_not_touching_edge = 0
    num_total_components = 0
    edge_mask = get_edge_mask(img, addtl_padding=1)
    for i in range(1, num_components):
        component_size = stats[i, cv2.CC_STAT_AREA]
        if component_size > 50:
            component_mask = labels == i
            if np.sum(component_mask * edge_mask) == 0:
                num_components_not_touching_edge += 1
            num_total_components += 1
    return num_total_components > 1 and num_components_not_touching_edge > 0

def get_dist_cumsum(lst):
    lst_shifted = np.array(lst[1:])
    distances = np.linalg.norm(lst_shifted - np.array(lst[:-1]), axis=1)
    # cumulative sum
    distances_cumsum = np.concatenate(([0], np.cumsum(distances)))
    return distances_cumsum

def black_on_path(color_img, pt, next_pt, num_to_check=10, dilate=True):
    # dilation should be precomputed
    img_to_use = cv2.dilate(color_img, np.ones((2, 2), np.uint8)) if dilate else color_img#.copy()
    # if np.linalg.norm(pt - next_pt) < 5:
    #     return 0.0
    num_black = 0
    # check if the line between pt and next_pt has a black pixel, using 10 samples spaced evenly along the line
    for i in range(num_to_check):
        cur_pt = pt + (next_pt - pt) * (i / num_to_check)
        if img_to_use[int(cur_pt[0]), int(cur_pt[1])] == 0:
            num_black += 1
    return num_black/num_to_check

def erode_image(img, kernel=(1, 1)):
    img = img.astype(np.uint8)
    kernel = np.ones(kernel, np.uint8)
    return cv2.erode(img, kernel)

def remove_specks(color_img):
    window_size = 5
    # remove tiny regions of stray white pixels 
    for i in range(color_img.shape[0] - window_size):
        for j in range(color_img.shape[1] - window_size):
            if np.sum(color_img[i:i+window_size, j:j+window_size, 0] > 0) < 3:
                color_img[i+window_size//2, j+window_size//2, :] = 0

    # zero out all edges of image
    color_img[0:window_size, :, :] = 0
    color_img[-window_size:, :, :] = 0
    color_img[:, 0:window_size, :] = 0
    color_img[:, -window_size:, :] = 0

    return color_img

def closest_nonzero_pixel(pt, depth_img):
    # find the closest nonzero pixel to pt
    nonzero_pixels = np.nonzero(depth_img)
    # print(nonzero_pixels[0].shape)
    pts_combined = np.array([nonzero_pixels[0], nonzero_pixels[1]]).T
    distances = np.sqrt((pts_combined[:, 0] - pt[0]) ** 2 + (pts_combined[:, 1] - pt[1]) ** 2)
    return pts_combined[np.argmin(distances)]

def normalize(vec):
    return vec / np.linalg.norm(vec)

def pixel_to_dist_from_nearest_black_point(image):

    # for each pixel, compute distance to nearest black pixel
    all_black = np.nonzero(image == 0)
    # add all black points to queue
    dq = deque()
    for i in range(len(all_black[0])):
        dq.append(np.array((all_black[0][i], all_black[1][i])))
    
    # initialize distances to infinity
    distances = np.full(image.shape, np.inf)
    distances[all_black] = 0

    # run dijkstra's algorithm
    # while len(q) > 0:
    #     print("Iter")
    #     closest_point, closest_dist = q[0], distances[tuple(q[0])]
    #     for i in range(1, len(q)):
    #         cur_pt_tuple = tuple(q[i])
    #         # print(distances.shape, q[i].shape, distances[tuple(q[i])])
    #         if distances[cur_pt_tuple] < closest_dist:
    #             closest_point, closest_dist = q[i], distances[cur_pt_tuple]
    #     q.remove(closest_point)

    #     # update distances
    #     for i in range(len(q)):
    #         distances[cur_pt_tuple] = min(distances[cur_pt_tuple], closest_dist + np.linalg.norm(q[i] - closest_point))
    
    # run BFS
    iters = 0
    while len(dq) > 0:
        iters += 1
        if iters % 100000 == 0:
            print("Iter", iters)
        next_pt = dq.popleft()
        
        # update distances
        for i in range(-1, 2):
            for j in range(-1, 2):
                cur_pt = next_pt + np.array((i, j))
                if not (cur_pt[0] < 0 or cur_pt[1] < 0 or cur_pt[0] >= image.shape[0]
                        or cur_pt[1] >= image.shape[1]):
                    if (distances[tuple(cur_pt)] == np.inf):
                        distances[tuple(cur_pt)] = distances[tuple(next_pt)] + 1
                        dq.append(cur_pt)
    return distances

def smooth_depth(depth_img):
    depth_cpy = depth_img.copy()
    # smooth the depth image with an average blur of non-zero values in a 3x3 window
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            if depth_img[i, j] == 0:
                continue
            cnt = 0
            depth_img[i, j] = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if (i + di >= 0 and i + di < depth_img.shape[0] and \
                        j + dj >= 0 and j + dj < depth_img.shape[1] and \
                        depth_cpy[i + di, j + dj] > 0):
                        depth_img[i, j] += depth_cpy[i + di, j + dj]
                        cnt += 1
            depth_img[i, j] /= cnt
    return depth_img

def visualize_depth_map_in_3d(depth):
    plt.imshow(depth)
    plt.clim(np.min(depth[np.nonzero(depth)]), np.max(depth[np.nonzero(depth)]))
    plt.show()

    points = []
    counter = 0
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            depth_val = depth[i, j]
            if depth_val <= 0.1:
                continue
            counter += 1
            if counter % 1 != 0:
                continue
            points.append(np.array([i, j, depth_val]))
    print("showing " + str(len(points)))
    points = np.array(points)

    # # fig = plt.figure()
    # # ax = plt.axes(projection='3d')
    lz = list(zip(*points))
    x = np.array(lz[0]).squeeze()
    y = np.array(lz[1]).squeeze()
    z = np.array(lz[2]).squeeze()

    data = [go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=2,
        )
    )]
    # show the plot
    fig = go.Figure(data=data)
    fig.show()
    # exit()

def visualize_spline_in_3d(img, path, plotly=True):
    points = []
    for pt in path:
        pt = pt.astype(int)
        pt = closest_nonzero_pixel(pt, img[:, :, 3])
        points.append(np.array([pt[0], pt[1], img[pt[0], pt[1], 3]]))
    
    lz = list(zip(*points))
    xs = np.array(lz[0]).squeeze()
    ys = np.array(lz[1]).squeeze()
    zs = np.array(lz[2]).squeeze()

    if plotly:
        fig = go.Figure(data=[go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers+lines',
            marker=dict(
                size=2,
                color=[i for i in range(len(xs))],
            ),
            line=dict(
                color=[i for i in range(len(xs))],
            )
        )])
        fig.show()
    else:
        ax = plt.axes(projection='3d')
        for i in range(len(xs) - 1):
            ax.plot3D([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], [zs[i], zs[i + 1]], c = [i/len(xs), 0, 1 - i/len(xs)])
        plt.show()

def dedup_and_center(image, points, dedup_dist):
    # greedy deduplicate points within distance 
    filtered_points = []
    too_close = False
    for pt in points:
        for i in range(len(filtered_points)):
            if np.linalg.norm(pt - filtered_points[i]) < dedup_dist:
                too_close = True
                break
        if not too_close:
            filtered_points.append(pt)
    
    centered_points = []
    for pt in filtered_points:
        centered_points.append(closest_nonzero_pixel(pt, image[:, :, 0]))

    return np.array(centered_points)
    

def grid_cable(image, vis=False, res=20, dedup_dist=14):
    orig_image = image.copy()
    image[:, :, :3] = cv2.dilate(image[:, :, :3], np.ones((3, 3), np.uint8))

    points = []
    U, V = np.nonzero(image[:, :, 0] > 0)
    pts = np.array([U, V]).T
    # choose pts with x or y divisible by 5

    for pt in pts:
        if pt[0] % res == 0 or pt[1] % res == 0:
            points.append(pt)
    points = np.array(points)
    points = dedup_and_center(orig_image, points, dedup_dist)

    if vis:
        plt.imshow(orig_image[:, :, :3])
        plt.scatter(points[:, 1], points[:, 0], s=10)
        plt.show()

    return points

def grid_cable_bfs(image, vis=False, res=40):
    queue = deque()

    points = []
    visited = np.zeros(image.shape[:2])
    counter = 0
    while visited.sum() < (image[:, :, 0] > 0).sum():
        start_point = closest_nonzero_pixel(np.array([0, 0]), (image[:, :, 0] > 0) - visited)
        visited[start_point[0], start_point[1]] = 1
        queue.append((start_point, 0))
        while len(queue) > 0:
            cur_pt = queue.popleft()
            cur_dist = cur_pt[1]
            if cur_dist % res == 0:
                points.append(cur_pt[0])
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    next_pt = cur_pt[0] + np.array((i, j))
                    if next_pt[0] < 0 or next_pt[1] < 0 or next_pt[0] >= image.shape[0] or next_pt[1] >= image.shape[1]:
                        continue
                    if visited[next_pt[0], next_pt[1]] == 1:
                        continue
                    if image[next_pt[0], next_pt[1], 0] > 0:
                        queue.append((next_pt, cur_dist + 1))
                        visited[next_pt[0], next_pt[1]] = 1

    points = np.array(points)
    points = dedup_and_center(image, points, 5)

    if vis:
        plt.imshow(image[:, :, :3])
        plt.scatter(points[:, 1], points[:, 0], s=10)
        plt.show()

    return points

def visualize_path(img, path, black=False):
    def color_for_pct(pct):
        return colorsys.hsv_to_rgb(pct, 1, 1)[0] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[1] * 255, colorsys.hsv_to_rgb(pct, 1, 1)[2] * 255
        # return (255*(1 - pct), 150, 255*pct) if not black else (0, 0, 0)
    img = img.copy()[:, :, :3].astype(np.uint8)
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

def score_path(color_img, depth_img, points):
    # get the MLE score for a given path through the image
    # find the farthest distance of a white pixel from all the points
    # total_score = 10000
    color_img[600:] = 0
    white_pixels = np.argwhere(color_img > 100)
    points = np.array(points)
    distances = white_pixels[None, :, :2] - points[:, None, :]
    # distances = np.sqrt((white_pixels[0][None, :] - points[:, 0, None]) ** 2 + (white_pixels[1][None, :] - points[:, 1, None]) ** 2)
    # print(distances.shape)
    # print(distances.shape)
    distances = np.linalg.norm(distances, axis=-1)
    max_distance = np.max(np.min(distances, axis=0))
    coverage_score = -max_distance / 15 if max_distance > 30 else 0

    # coverage_score = 0

    # now assess sequential probability of the path by adding log probabilities
    total_angle_change = 0
    cur_dir = normalize(points[1] - points[0])
    for i in range(1, len(points) - 1):
        new_dir = normalize(points[i+1] - points[i])
        # print(new_dir.dot(cur_dir))
        total_angle_change += abs(np.arccos(np.clip(new_dir.dot(cur_dir), -1, 1)))**2
        cur_dir = new_dir
    total_angle_change /= len(points)
    return -total_angle_change*5 + coverage_score

def get_best_path(image, finished_paths, stop_when_crossing=False):
    # init best score to min possible python value
    best_score, best_path = float('-inf'), None
    for path in finished_paths:
        score = score_path(image[:, :, :3], image[:, :, 3], path, partial_paths=stop_when_crossing)
        if score > best_score:
            best_score = score
            best_path = path
    print("Best score", best_score, "Best path", best_path is not None)
    return best_path

def sort_paths_by_score(image, finished_paths, stop_when_crossing=False):
    # return a list of paths sorted by score
    scores = []
    for path in finished_paths:
        scores.append(score_path(image[:, :, :3], image[:, :, 3], path, partial_paths=stop_when_crossing))
    return np.array(finished_paths)[np.argsort(scores)]

def visualize_edges(image, edges):
    image = image.copy()
    for pt in edges.keys():
        plt.scatter(*pt[::-1], s=10, c='r')
        for second_pt in edges[pt]:
            cv2.line(image, pt[::-1], second_pt[0][::-1], (0, 0, 255), 1)
    return image

def delete_overlap_points(path, threshold=4):
    # delete points that are far apart in the spline
    # but too close to each other
    def within_threshold(pt1, pt2):
        return np.linalg.norm(pt1 - pt2) < threshold

    new_path = []
    for i in range(len(path)):
        add_point = True
        for j in range(len(new_path)):
            if within_threshold(path[i], new_path[j]):
                new_path.pop(j)
                add_point = False
                break
        if add_point:
            new_path.append(path[i])
    return np.array(new_path)


if __name__ == "__main__":
    img_path = 'data_bank/series_simple/1640295900/color_0.npy' #'data_bank/large_overhand_drop/1640297206/color_0.npy' #'data_bank/large_figure8_simple/1640297369/color_0.npy'
    color_img = np.load(img_path)

    color_img[600:, :, :] = 0
    color_img[:, :100, :] = 0

    color_img = remove_specks(np.where(color_img < 80, 0, 255))

    grid_cable_bfs(color_img, vis=True)

