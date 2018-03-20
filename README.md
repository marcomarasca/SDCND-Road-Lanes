# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

[//]: # (Image References)

[image1]: ./examples/input.jpg "Input image"
[image2]: ./examples/output_7_final.jpg "Final image"
[image3]: ./examples/output_1_color_mask.jpg "Color masking"
[image4]: ./examples/output_2_grayscale.jpg "Grayscale conversion"
[image5]: ./examples/output_3_blurred.jpg "Gaussian blur"
[image6]: ./examples/output_4_edges.jpg "Edge detection"
[image7]: ./examples/output_5_masked.jpg "Region of interest"
[image8]: ./examples/output_5_masked_region.jpg "Masked region"
[image9]: ./examples/output_6_lanes.jpg "Lane detection"
[image10]: ./examples/output_6_lines.jpg "Detected lines"

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project we implement a simple lane detection pipeline in images using Python and OpenCV. OpenCV ([Open-Source Computer Vision](https://opencv.org/)), which is a package that has many useful tools for analyzing images.

The code is included in the [jupyter notebook](P1.ipynb) in the repository.

The pipeline that is implemented in this project consists of the follwing steps:

1. Color masking for white and yellow regions converting to HSL colorspace
2. Grayscale conversion for the edge detection
3. Gaussian blur to reduce noise
4. Edge detection using Canny transform
5. Region of interest masking
6. Lines detection in the masked region using a hough transform
7. Road lane detection averaging the lines from the previous step

![alt text][image1] ![alt text][image2]

## Color Masking

A simple pre-processing step that can be performed is to find the yellow and white regions in the input image, while this is not probably the best apporach it works well for simple scenarios. An interesting way to work with colors in images is actually to examin different color spaces. RGB is not very flexible as each channel affects each other. A simple guide for color selection in different color spaces can be found at https://www.learnopencv.com/color-spaces-in-opencv-cpp-python. 

OpenCV provides a simple way to select ranges of colors using the [inRange](https://docs.opencv.org/3.4.1/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981) function.
```python
img_converted = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
white_mask = cv2.inRange(img_converted, np.array([0, 200, 0]), np.array([255, 255, 255]))
yellow_mask = cv2.inRange(img_converted, np.array([10, 100, 150]), np.array([40, 255, 255]))
```

In our case we simple convert the input image to HSL and we select a range for the yellow and white regions using mostly the L(uminance) channel for the whites and the H(ue) channel for the yellow.

We later add the two masks:

```python
mask = cv2.bitwise_or(white_mask, yellow_mask)
```

And finally apply the mask to the input image:

```python
return cv2.bitwise_and(img, img, mask = mask)
```

![alt text][image3]

## Edge Detection

For the edge detection we use a [Canny transform](https://en.wikipedia.org/wiki/Canny_edge_detector), the input is an 8 bit image so we need to convert first to grayscale. We also apply an additional Gaussian filter in order to further reduce the noise in the image and avoid outliers.

```python
grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel_size = 15
blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
![alt text][image4] ![alt text][image5]

Note that the higher the kernel size the more the image will be blurred.

```python
low_threshold = 50
high_threshold = 150
edge_img = cv2.Canny(img, low_threshold, high_threshold)
```
![alt text][image6]

## Region of Interest

To further improve the accuracy we apply a region of interest to the image masking out a polygon that would represent the road in front of the camera:
```python
x = img.shape[1]
y = img.shape[0]

region_center_gap = 50
vertical_clip_ratio = 0.62
side_clip_ratio = 0.08

region = np.array([[(x * side_clip_ratio, y * (1 - side_clip_ratio)), 
                    (x // 2 - region_center_gap, y * vertical_clip_ratio), 
                    (x // 2 + region_center_gap, y * vertical_clip_ratio), 
                    (x * (1 - side_clip_ratio), y * (1 - side_clip_ratio))]],
                    dtype=np.int32)

mask = np.zeros_like(img)   
    
cv2.fillPoly(mask, region, 255)
```
![alt text][image8] ![alt text][image7]

## Lane Detection

Now the last part of the pipeline tries to detect the road lanes from the image above, first we can detect all the lines in the image using a [hough transform](https://en.wikipedia.org/wiki/Hough_transform):

```python
lines = cv2.HoughLinesP(img, 1, np.pi/180, 12, np.array([]), minLineLength=20, maxLineGap=200)
```
![alt text][image10]

Finally we can proceed in detecting the road lanes using an average of the lines extracted from the hough transform. The hough transform returns a set of lines as points coordinates representing the lines, we can therefore compute the slope and intercept of each line:

```python
slope = (y2 - y1) / (x2 - x1)
intercept = y1 - slope * x1
```

and add it to different lists according to the slope (e.g. to divide left and right lines according to the direction):

```python
# Skips steeps angles
if angle < 15 or angle > 75:
    continue
if slope > 0: # Y is inverted (top-down)
    right_lines.append((slope, intercept))
elif slope < 0:
    left_lines.append((slope, intercept))
```

Finally we compute the mean:

```python
left_mean = np.mean(left_lines, axis = 0) if len(left_lines) > 0 else None
right_mean = np.mean(right_lines, axis = 0) if len(right_lines) > 0 else None    
```

From the average of the slope and intercept we can compute the sinlge left and right lanes:

```python
def get_line_coordinates(line_slope_intercept, y1, y2):

    slope, intercept = line_slope_intercept

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return x1, y1, x2, y2
    
def extract_lanes(img, lines, vertical_clip_ratio = 0.62):

    # This computes the average slope and intercept from the set of lines
    # left_line and right_line are tuple containing the (slope, intercept)
    left_line, right_line = mean_slope_intercept(lines)
    
    img_shape = img.shape
    lanes = []
    
    if left_line is not None:
        left_lane = get_line_coordinates(left_line, int(img_shape[1]), int(img_shape[0] * vertical_clip_ratio))
        lanes.append([left_lane])
        
    if right_line is not None:
        right_lane = get_line_coordinates(right_line, int(img_shape[1]), int(img_shape[0] * vertical_clip_ratio))
        lanes.append([right_lane])
    
    return lanes
```
![alt text][image9] ![alt text][image2]

The pipeline is then applied to video files, processing each frame and writing a new annotated video:

[![Lane Detection Pipeline](http://img.youtube.com/vi/aQgzi_cLuFM/0.jpg)](http://www.youtube.com/watch?v=aQgzi_cLuFM "Lane Detection Pipeline")

