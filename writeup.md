# **Finding Lane Lines on the Road** 

The goal of the project is to take in input an RGB image and output an annotated version of the image that oulines the road lanes detected in the input image. Finally the pipeline is applied to all the frames of a set of input videos in order to produce an annotated video with the detected lanes.

[//]: # (Image References)

[image1]: ./examples/output_1_color_mask.jpg "Color mask"
[image2]: ./examples/output_2_grayscale.jpg "Grayscale"
[image3]: ./examples/output_3_blurred.jpg "Blurred version"
[image4]: ./examples/output_4_edges.jpg "Edge detection"
[image5]: ./examples/output_5_masked.jpg "Region of interest mask"
[image6]: ./examples/output_6_lanes.jpg "Lanes detection"
[image7]: ./examples/output_7_final.jpg "Final result"
[image8]: ./examples/output_failing.jpg "Failing edge detection"
[image9]: ./examples/output_failing_correct.jpg "Fixed edge detection"

---

## Pipeline description

The pipeline I implemented consists of the following steps:

1. Color masking for white and yellow regions converting to HSL colorspace
2. Gaussian blur to reduce noise
3. Grayscale conversion for the edge detection
4. Edge detection using Canny transform
5. Region of interest masking
6. Lines detection in the masked region using a hough transform
7. Road lane detection averaging the lines from the previous step

For the **edge detection** I use a Canny transform and the input of the algorithm is an 8 bit image, therefore the image is first converted to **grayscale**. Using the suggested ratio 1:3 for the thresholds of the algorithm yields good results when a low threshold of 50 and an high threshold of 150 are selected.

![alt text][image2]

The openCV implementation specifies that a **gaussian blur** is applied to the image before running the Canny transform. I found that the default was not enough so I added an additional Gaussian blur to further reduce the noise in the image before feeding it to the edge detector with a relatively high kernel size (15). I found that using higher kernel sizes yield better results (at least on the test images) since it reduced the amount of outliers.

![alt text][image3]

Finally I also added an additional pre-processing step using a color mask to detect the yellow and white parts of the image in order to improve the overall performance of the pipeline. The initial implementation didn't include this step and it was working fine for the first two videos. As soon as I moved to the challenge video I noticed that the pipeline was failing when lighting and contrast of of the image were not good enough. I extracted a few frames from the video to better debug the issue and run the pipeline step by step. 

![alt text][image8]

It was clear that the failing step was the edge detector, I could improve the output reducing the kernel size of the gaussian blur filter but this would have worsened the overall performance of the pipeline. At this point I started doing some research and found a nice article about color spaces and open CV (https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/). This led me to test out various approaches working in different color spaces to better isolate the yellow and white lanes. Initially I tried working with the HSV that produced great result for filtering masking the yellow regions, but was a bit more complicated for the white regions. I then thought of converting in two different color spaces to process the yellow and white regions separately but I actually found a similar solution from another student that got better results using the HSL color space only (therefore reducing the processing required). I finally adopted this solution in the final pipeline.

![alt text][image1] 
![alt text][image9]

After the edge detection the pipeline applies a mask for the region of interest that focuses on what is in front of the camera util a certain distance (which is basically defined by a threshold for the vertical size of a frame).

![alt text][image5]

The pipeline continues with the final step of **lane detection** that is split in two main parts: 

First, from the ouput of the canny transform we can detect lines using the **hough line transform**. Playing with the various parameters we can obtain great results already. The output is a set of lines defined as 2 points coordinates, the goal of the final step is take this set of lines and find an aproximation of the left and right lanes. In order to achive this result the algorithm simply computes the **mean of the slope and intercept** of each detected line and then using a set of partial starting coordinates we can simply extract the two lines (clipping to a certain limit). The slope is used to detect which lines are part of the right or left lane so that the average can be computed separatly. Moreover lines that are within a certain angle range are not taken into consideration (e.g. horizontal and vertical lines) while computing the mean of the slope and intercept.

![alt text][image6]
![alt text][image7]
---
## Pipeline shortcomings and possible improvements

The pipeline provides good results for the example images and videos, the final implementation provides decent results also for the last challenge video. There are several issues and limitation in the current implementation.

Several parameters are hardcoded and should be probably parameterized and/or modified according to external feedback. This implementation assume a static and controlled environment which is not applicable in real world scenario.  For example the lighting condition may change drastically and more advanced technique may be applied to preprocess the input images, maybe using a some sort of normalization for the light conditions. Additionally the region masking of the image makes strong assumptions about the position of the camera/car in respect to the road.

Another issue that is currently present is the that the lanes are detected in each frame separatly, on the video this leads to lanes slightly jumping without a smooth transition between frames. An interesting solution that I saw implemented is to average the lines not only within the set of detected lines by the hough transform but also including lanes detected in previous frames in order to smooth out the result.

Another shortcoming is the lane detection itself, a few disturbance in the image (e.g. a car driving in front of the camera) may easily disrupt the pipeline. The lanes may also be of different colors or missing all together (e.g. an offroad trail). The pipeline would probably fail also on very curvy roads or on steep roads.

