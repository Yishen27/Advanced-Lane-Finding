## Writeup Advanced Lane Finding Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I wrote all the function codes in one file, the "Advanced lane finding code final.py", so I'll only discuss how I implanted them.  

I prepared object points, for the chessboard(9*6) corners. As we learned the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

After I got all `objpoints` and `imgpoints` by interate through all images in "camera_cal", I computed the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the "undistorted.jpg" in "/output_images".

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I wrote the undistorted function in the function "cal_img()", as discribed in the last part. The example is "undistorted.jpg" in "/output_images". By the way, all my image examples are in "/output_images".

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

For binary image, I used a combination of HLS color and gradient thresholds to generate a binary image. To be more specific, I first only combined S channel and sobel X, to get a higher efficiency. It worked for most test figure until I got to try the video. A problem occured as "problem image.png", I tuned the two thresholds but didn't work. So I added L channel into the combination, and solved the problem, as "solved image2.png".  



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called "warp_img()" (start from line 67).  As we're dealing with images with same shape, I set the src and dst points as fixed.  I chose the source and destination points as follows:

```python
offset = 250
src = np.float32([[600,445],[675,445], [1060,690], [250,690]])
dst = np.float32([[offset, 0], [img_size[0] - 350, 0],
                      [img_size[0] - 350, img_size[1]],
                      [offset, img_size[1]]])
```

The example of warped image is "warped.png".


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After warp images, I wrote a function "find_lane_pixels()" to identify lane pixels with histogram. It took the lower half of the warped image, found the lane position and use slid windows to identify lane pixels. The pixels are fitted with a polynomial by the function "fit_polynomial()". The example from this step is "fit.png".  And for video precessing, I defined a class "Lane" to track recent detected values, with functions to check the sanity by compare the detected radius and lane width with previous data. And a "search_around_poly()" function is used to detecte from previous results. The function first check if anything is detected in last frame, if yes, continue with previous data and generate new data. Then do the sanity check, if passed update the data in the lane class, if not get back to the lane pixel detection.



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Then I measure curvature radius with the function learned in the class, and convert the data to real-world value in function measure_curvature_real(). I calculated the average radius of left and right polynomial as the radius.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step with function "project()", project the lane and data onto the image. The example is "final.png".

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The final result of my video is "project_video8.mp4".

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I have in this project is the measurement of the radius. The result of radius seems to be off (not always around 1 km) I checked many times but didn't figure out why. However, the offset from lane center and lane width are pretty accurate. Can I get a hint?
Another problem is the efficiency. My software has to take like 40 mins to process a video. If I got time, this will be the part I want to optimize.
