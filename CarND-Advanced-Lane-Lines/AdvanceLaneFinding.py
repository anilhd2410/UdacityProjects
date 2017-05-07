#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import PIL
import glob
import pickle
from PIL import Image
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#play video or test on images
Play_video = True

#Save images to a file
Save_image = False

#Variable name for saving images 
image_count = 0

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

#Previous position of lines
gleft_fit = 0
gright_fit = 0 

Calibrate_Image = False


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

#function to calibrate camera using chessboard images
#save result into the file so can be applied later to camera images
def ImageCalibration():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    #images = glob.glob('camera_cal/calibration*.jpg')
    images = os.listdir("camera_cal")
  
    dist_pickle = {}

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread('camera_cal/'+fname)
        image = np.copy(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            #Undistort using mtx and dist
            Uimg = cv2.undistort(image, mtx, dist, None, mtx)
            if Save_image == True:
                PrintImg = Image.fromarray(Uimg)
                #PrintImg.save('output_images/'+ fname)

                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()
                ax1.imshow(image)
                ax1.set_title('Original Image', fontsize=50)
                ax2.imshow(Uimg)
                ax2.set_title('Undistorted Image', fontsize=50)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                plt.savefig('output_images/'+ fname)

            # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
            dist_pickle["mtx"] = mtx
            dist_pickle["dist"] = dist
    
    pickle.dump( dist_pickle, open( "calibrate_pickle.p", "wb" ) )



def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#Function to perform image distortion and perspective transform
def corners_unwarp(img, mtx, dist):
    #Undistort using mtx and dist
    Uimg = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])
    #points on source image
    src = np.float32([[580,460],[730,460],[1170,img_size[1]],[150,img_size[1]]])

    offset=200
    #points on destination image
    dst = np.float32([[offset, 0],[img_size[0]-offset, 0],[img_size[0]-offset,img_size[1]],[offset, img_size[1]]])
    
    #Transform image    
    M = cv2.getPerspectiveTransform(src, dst)
    #Inverse Transform of a image
    MinV = cv2.getPerspectiveTransform(dst, src)
    #Warp an image using the perspective transform, M
    wraped = cv2.warpPerspective(Uimg, M, img_size, flags=cv2.INTER_LINEAR)

    return wraped, M, MinV, Uimg, src, dst


#Finction to perform and combining color and gradient fliters of an image
def AppyColorAndGradient(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsl[:,:,1]
    s_channel = hsl[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    #The final image color_binary is a combination of binary thresholding the S channel (HLS) and 
    #binary thresholding the result of applying the Sobel operator in the x direction on the original image
    color_binary = np.zeros_like(sxbinary)
    color_binary[(sxbinary==1)|(s_binary==1)]=1
    return color_binary

#For first image frame find lane by window sliding
def FindLanesSlidingWindow(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #prepare an image for lotting
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img,left_fit,right_fit, left_fitx, right_fitx, ploty

#Function to find lane from previous line findings results
#It improves performance over window sliding for every video frame
def FindLanesMarginArround(binary_warped, left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit,right_fit,left_fitx, right_fitx, ploty
    
#Function to find left and right line curvature
def FindLaneCurvature(left_fitx, right_fitx, ploty):
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # values in m
    return left_curverad, right_curverad

#Function to find vehicle position to the center from the lane
def FindVehiclePosition(ImageCenter,newwarp_pts):
    #Get the left and right closest point fromm the wrapper around bottom of the image
    left  = np.min(newwarp_pts[(newwarp_pts[:,1] < ImageCenter) & (newwarp_pts[:,0] > 700)][:,1])
    right = np.max(newwarp_pts[(newwarp_pts[:,1] > ImageCenter) & (newwarp_pts[:,0] > 700)][:,1])
    #Center point between left and right points
    center = (left + right)/2
    return center


#Function which excutes steps to Find lanes and plot results
def ImagePipeline(image):
    global image_count
    global gleft_fit
    global gright_fit
    image_count = image_count + 1
    # Define a kernel size and apply Gaussian smoothing
    img = gaussian_blur(image,5)    

    imgshape = img.shape
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of ins
    vertices = np.array([[(.51*imgshape[1],imgshape[0]*.58),(.49*imgshape[1],imgshape[0]*.58) \
                              ,(0,imgshape[0]),(imgshape[1],imgshape[0])]],dtype=np.int32)
    

    result = AppyColorAndGradient(img,s_thresh=(175, 250), sx_thresh=(30, 150))

    img = region_of_interest(result,vertices)
    if Save_image == True:
        PrintImg = Image.fromarray(img * 255)
        PrintImg.save('output_images/Color_Gradient'+str(image_count)+'.jpg')

    #Retrieve the saved mtx and dst to apply undistortion
    dist_pickle = pickle.load( open( "calibrate_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    binary_warped, perspective_M, perspective_MinV, undist, src, dst = corners_unwarp(img, mtx, dist)
    if Save_image == True:
        xPoints = [src[0][0],src[1][0],src[2][0],src[3][0],src[0][0]]
        yPoints = [src[0][1],src[1][1],src[2][1],src[3][1],src[0][1]]
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        # Showing the image from pipeline with marked areas
        ax1.set_title('Unstorted Wrapped_Src', fontsize=40)
        ax1.plot(xPoints,yPoints,'r-',lw=2)
        ax1.imshow(image)
        
        xPoints = [dst[0][0],dst[1][0],dst[2][0],dst[3][0],dst[0][0]]
        yPoints = [dst[0][1],dst[1][1],dst[2][1],dst[3][1],dst[0][1]]
        ax2.plot(xPoints,yPoints,'r-',lw=2)
        ax2.imshow(binary_warped, cmap='gray')
        ax2.set_title('Unstorted Wrapped dest', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig('output_images/AreaOnImage'+ str(image_count)+'.jpg')

    #for first image apply sliding window, only applicable for video
    if(image_count ==1 or Play_video==False):
        out_img, gleft_fit,gright_fit, left_fitx, right_fitx, ploty = FindLanesSlidingWindow(binary_warped)
        if Save_image == True:
            PrintImg = Image.fromarray(out_img, 'RGB')
            PrintImg.save('output_images/SlidingWindow_output'+str(image_count)+'.jpg')
    else:
        #rest of the images use findings from previous frame
        gleft_fit,gright_fit, left_fitx, right_fitx, ploty = FindLanesMarginArround(binary_warped, gleft_fit,gright_fit)

    left_curv, right_curv = FindLaneCurvature(left_fitx, right_fitx, ploty)

    # Read in a thresholded image
    warped = np.copy(binary_warped)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, perspective_MinV, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    #Draw curvatue information on an image
    curvature = round((left_curv + right_curv)/2,2)
    font = cv2.FONT_HERSHEY_DUPLEX 
    text = "Radius of the curvature {}m".format(curvature)
    cv2.putText(result,text,(20,50), font, 1,(255,255,255),1)

    #center of the image, image width divide by 2
    ImageCenter = img.shape[1]/2
    center = FindVehiclePosition(ImageCenter,np.argwhere(newwarp[:,:,1]))
    #position from center of the image and convert from picek to meter 
    CarPosition = (ImageCenter - center) * xm_per_pix
    #Draw vehcile position curvatue information on an image
    if CarPosition > 0:
        text = "Car Position from center: {}m".format(round(CarPosition,2))
    else:
        text = "Car Position from center {}m".format(round(-CarPosition,2))
    cv2.putText(result,text,(20,80), font, 1,(255,255,255),1)

    return result


if Play_video == True:
    ### Import everything needed to edit/save/watch video clips
    # Set up lines for left and right
    white_output = 'white.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(ImagePipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

else:

    #Run first time  to save wide_dist_pickle
    #ImageCalibration()
    #(1) get the file names from given folder
    if Calibrate_Image == True:
        ImageCalibration();
    else:
        file_list = os.listdir("test_images")
        print(file_list)
    
        for i, file_name in enumerate(file_list):
            #reading in an image
            image = mpimg.imread('test_images/' + file_name)
            result_img = ImagePipeline(image)
            if Save_image == True:
                PrintImg = Image.fromarray(result_img, 'RGB')
                PrintImg.save('output_images/Output'+str(image_count)+'.jpg')
    

