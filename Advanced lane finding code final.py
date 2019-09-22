import numpy as np
import cv2
import glob
from pylab import *
from moviepy.editor import VideoFileClip

images = glob.glob('camera_cal/calibration*.jpg')
img_points = []
obj_points = []

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# iterate through all calibration images to find the corners
for i in images:
    t_img = cv2.imread(i)
    gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        img_points.append(corners)
        obj_points.append(objp)

# get coefficients
cal = plt.imread('test_images/test4.jpg')
g = cv2.cvtColor(cal, cv2.COLOR_BGR2GRAY)
g_size = (g.shape[1], g.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, g_size, None, None)

# function to calibration camera and undistort image
def cal_img(img):

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# identify lane lines
def binary(undist, s_thresh=(120, 255), sx_thresh=(30, 100), sobel_kernel=5):

    img = np.copy(undist)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # Threshold color channel L and S
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 120) & (l_channel <= 255)] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1)& (l_binary == 1))| (sxbinary == 1)] = 1
    return combined_binary


#warp binary image
def warp_img(img):

    img_size = (img.shape[1], img.shape[0])

    offset = 290
    src = np.float32([[600, 445], [675, 445], [1060, 690], [250, 690]])
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]],
                      [offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


# find lane pixels with histogram
def find_lane_pixels(warped):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
    out_img = np.dstack((warped, warped, warped))
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # hyperparameters
    nwindows = 9
    margin = 100
    minpix = 50

    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows
    for window in range(nwindows):
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current- margin
        win_xleft_high = leftx_current+ margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        print('The function failed to find indicies!')
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


# fit polynomials with pixels found
def fit_polynomial(leftx, lefty, rightx, righty):

    # Fit polynomial

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    ## visualization codes to check polynomial
    # ploty = lefty
    # try:
    #     left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #     right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # except TypeError:
    #     # Avoids an error if `left` and `right_fit` are still none or incorrect
    #     print('The function failed to fit a line!')
    #     left_fitx = 1 * (ploty) ** 2 + 1 * ploty
    #     right_fitx = 1 * (ploty)** 2 + 1 * ploty
    #
    # # Colors in the left and right lane regions
    # # out_img[lefty, leftx] = [255, 0, 0]
    # # out_img[righty, rightx] = [0, 0, 255]
    #
    # # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit


# meausre and calculate values we want
def measure_curvature_real(shape, left_fit, right_fit, lefty, righty):
    ym = 30 / 720
    xm = 3.7 / 700

    # Define y-value where we want radius of curvature
    y_eval_l = np.max(lefty) * ym
    y_eval_r = np.max(righty) * ym


    left_curverad = ((1 + (2 * left_fit[0] * (xm / (ym ** 2)) * y_eval_l + left_fit[1] * (xm / ym)) ** 2) ** 1.5) / np.absolute(2 * left_fit[0] * (xm / (ym ** 2)))
    right_curverad = ((1 + (2 * right_fit[0] * (xm / (ym ** 2)) * y_eval_r + right_fit[1] * (xm / ym)) ** 2) ** 1.5) / np.absolute(2 * right_fit[0] * (xm / (ym ** 2)))

    average_radius = (left_curverad + right_curverad) / 2
    left_x = left_fit[0] * (y_eval_l / ym) ** 2 + left_fit[1] * (y_eval_l / ym) + left_fit[2]
    right_x = right_fit[0] * (y_eval_r / ym) ** 2 + right_fit[1] * (y_eval_r / ym) + right_fit[2]
    offset = ((shape[1] / 2) - (left_x + right_x) / 2) * xm
    width = (left_x - right_x) * xm

    return average_radius, offset, width


# define a class to track our results
class Lane():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the most recent fit
        self.current_L_fit = [np.array([False])]
        self.current_R_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        # measured lane width
        self.lane_width = None

    # sanity check function, decide whether to search from start or not
    def Sanity_Check(self, radius, lane_width):
        if abs((radius-self.radius_of_curvature)/self.radius_of_curvature)>0.3 and abs((lane_width-self.lane_width)/self.lane_width)> 0.2:
            self.detected = False
            return False
        else:
            self.detected = True
            return True

    # update key values
    def update(self, L_fit, R_fit, radius, offset, lane_width):
        self.current_L_fit = L_fit
        self.current_R_fit = R_fit
        self.radius_of_curvature = radius
        self.line_base_pos = offset
        self.lane_width = lane_width
        self.detected = True


# search from detected values
def search_around_poly(binary_warped, lane):
    # HYPERPARAMETER
    margin = 80
    shape = binary_warped.shape

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if lane.detected == True:

        left_fit = lane.current_L_fit
        right_fit = lane.current_R_fit

        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit, right_fit= fit_polynomial(leftx, lefty, rightx, righty)
        R_c, offset, width = measure_curvature_real(shape, left_fit, right_fit, lefty, righty)

        s_c = lane.Sanity_Check(R_c, width)
        if  s_c == True:
            lane.update(left_fit, right_fit, R_c, offset, width)
            return left_fit, right_fit, R_c, offset
        else:
            pass

    else:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
        left_fit, right_fit = fit_polynomial(leftx, lefty, rightx, righty)
        R_c, offset, width = measure_curvature_real(shape, left_fit, right_fit, lefty, righty)
        lane.update(left_fit, right_fit, R_c, offset, width)
        return left_fit, right_fit, R_c, offset


# project what we had to images
def project(undist, warped, left_fit, right_fit, R_c, offset):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_size = (undist.shape[1], undist.shape[0])

    dst = np.float32([[600, 445], [675, 445], [1060, 690], [250, 690]])
    src = np.float32([[290, 0], [img_size[0] - 290, 0],
                      [img_size[0] - 290, img_size[1]],
                      [290, img_size[1]]])
    Minv = cv2.getPerspectiveTransform(src, dst)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)


    #add data into the image
    R_c = str(R_c)
    offset = str(offset)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    line1 = 'Radius of curv is ' + R_c + 'm'
    line2 = 'Vehicle is ' + offset + 'm to the center'
    img = cv2.putText(result, line1, (0, 40), font, 1.2, (255, 255, 255),2)  # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
    img = cv2.putText(result, line2, (0, 80), font, 1.2, (255, 255, 255), 2)

    # plt.imshow(img)
    # plt.show()

    return img


# define the class as a globel class
lane1 = Lane()


# whole process pipline
def lane_detection(img):
    global lane1
    undist = cal_img(img)
    b = binary(undist)
    w = warp_img(b)
    left_fit, right_fit, R_c, offset = search_around_poly(w, lane1)
    result = project(undist, w, left_fit, right_fit, R_c, offset)

    return result


# test with video
test_output = 'output_videos/project_video8.mp4'

#test and tune with clips
# clip1 = VideoFileClip("project_video.mp4").subclip(23,26)
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(lane_detection)
output_clip.write_videofile(test_output, audio=False)
