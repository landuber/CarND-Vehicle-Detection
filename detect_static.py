import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
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


images = glob.glob('test_images/*.jpg')
out_dir = 'output_images'
out_dir2 = 'output_images2'

for image in images:
    bbox_list = []
    img = mpimg.imread(image)

    for scale in scales:
        out_img, bboxes = find_cars(img, ystart, ystop, scale, svc, 
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
    mpimg.imsave(os.path.join(out_dir, os.path.basename(image)), draw_img)
    mpimg.imsave(os.path.join(out_dir2, os.path.basename(image)), out_img)



