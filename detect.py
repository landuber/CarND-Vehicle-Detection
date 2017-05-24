import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os.path
from scipy.ndimage.measurements import label
from detection_functions import *

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
color_space = dist_pickle["color_space"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]


ystart = 400
ystop = 656
scales = [1.5]
threshold = 1
out_dir = 'output_images'

#img = mpimg.imread('test_images/test1.jpg')
video = cv2.VideoCapture('project_video.mp4')

if not video.isOpened():
    print('Could not open video')
    sys.exit()

ok, img = video.read()
if not ok:
    sys.exit()

height, width, channels = img.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracked_video.mp4', fourcc, 20.0, (width, height))

count = 0
while True:
    bbox_list = []
    ok, img = video.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not ok:
        break

    for scale in scales:
        _, bboxes = find_cars(img, ystart, ystop, scale, svc, 
                                       color_space, X_scaler, orient, pix_per_cell, 
                                       cell_per_block, spatial_size, hist_bins)
        bbox_list.extend(bboxes)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_RGB2BGR)
    count += 1
    #mpimg.imsave(os.path.join(out_dir, str(count) + '.jpg'), draw_img)
    out.write(draw_img)

video.release()
out.release()


