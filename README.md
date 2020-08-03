# Skewered
OpenCV-based Image alignment tool for fixed point cameras
## Introduction
Quantative comparison between two photos taken with a fixed point camera requires image to image alignment based on pixel to pixel matching. skewered.py uses lens distortion models and homography translations to minimize the distance between matching points of two images. For local feature matching, I use AKAZE descripter and FLANN matcher of OpenCV. I use a lens distortion model similar to the cv2.calibratecamera. The parameters k1, k2, k3, k4, k5, k6, p1, p2 are estimated.

## Work flow
1. AKAZE local feature detection
2. FLANN matching
3. Select matched point by homography estimation with RANSAC
4. Estimate lens distortion and homography translation
5. Map the source image to the destination image 

## Dependencies
Python3  
opencv-python  
numpy

## Usage
### python code
```
src = "1009.jpeg"
dst = "1908.jpeg"
rslt = "result_1009.png"

# result
im1_dh = homography_lensdist(src, dst, rslt)

# make a diff image
im2 = cv2.imread(dst)
im_diff = im2.astype(int) - im1_dh.astype(int)
cv2.imwrite("diff.png", np.abs(im_diff))
```
### images
source and destination images are taken from [NIES homepage](http://db.cger.nies.go.jp/gem/ja/mountain/station.html?id=2)

source image
![](1009.jpeg)
destination image
![](1908.jpeg)
aligned source image
![](result_1009.png)
abs(aligned - destination)
![](diff.png)