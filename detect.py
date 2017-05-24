import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os.path
from collections import deque
from  moviepy.editor import VideoFileClip
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
scales = [1.2, 1.8]
threshold = 1
out_dir = 'output_images'
d = deque(maxlen = 5)

def transform(img):
    bbox_list = []
    for scale in scales:
        _, bboxes = find_cars(img, ystart, ystop, scale, svc, 
                                       color_space, X_scaler, orient, pix_per_cell, 
                                       cell_per_block, spatial_size, hist_bins)
        bbox_list.extend(bboxes)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, bbox_list)
    d.append(heat)


    heat = reduce(lambda x, y: np.add(x, y), d)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, len(d) * threshold)

    # Visualize the heatmap when displaying
    #heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

clip = VideoFileClip('project_video.mp4')#.subclip(25, 30)
newclip = clip.fl(lambda gf, t: transform(gf(t)))
newclip.write_videofile("final_video.mp4")
