**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Histogram of Oriented Gradients (HOG)

The code for this step is contained  in lines 19 through 36 of the file called `detection_functions.py`. This functions is used both in train.py
and detect.py for training and detection.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.


I tried various combinations of parameters and chose the ones that improved the accuracy on the test data. Parameters are listed in train.py between lines 19 and 25.


I trained a linear SVM using spatial, historgram and hog features - line 59 in train.py. I used the `extract_features` function in `detection_functions.py` file (between lines 106 and 154)


I decided to use the `find_car` function provided in the lessons with modified `cells_per_step = 1`. I decided on the parameter after trying several ones and seeing the bounding box generated.

Using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Examples outputs are found in the `output_images` folder. 
---

Here's a [link to my video result](./tracked_video.mp4)


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  This is done in file detect.py in lines 58 to 69.

---

The solution can be improved by using multiple scales which might cause performance issues. But that can be tackled a with a native C/C++ implementation using CUDA. Also the number of false positives can be reduced by considering multiple frames instead of just one frame.

