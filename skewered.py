import cv2
import numpy as np
import math
import copy
from scipy import optimize as opt

def homography_lensdist(source_path, destination_path, result_path):
  # Read images
  im1 = cv2.imread(source_path)
  im2 = cv2.imread(destination_path)
  # Akaze descripter
  akaze = cv2.AKAZE_create()
  kp1, des1 = akaze.detectAndCompute(im1, None)
  kp2, des2 = akaze.detectAndCompute(im2, None)
  # Flann matching
  FLANN_INDEX_LSH = 6
  index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6,  
                     key_size = 12,     
                     multi_probe_level = 1) 
  search_params = dict(checks = 50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k = 2)
  ratio = 0.8
  good = []
  for m, n in matches:
    if m.distance < ratio * n.distance:
      good.append(m)
  
  pts1 = np.float32([ kp1[match.queryIdx].pt for match in good ])
  pts2 = np.float32([ kp2[match.trainIdx].pt for match in good ])
  pts1 = pts1.reshape(-1,1,2)
  pts2 = pts2.reshape(-1,1,2)
  hm, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 100)

  pts1 = pts1[mask.astype('bool')]
  pts2 = pts2[mask.astype('bool')]
  
  def distort(pts, params):
    k1 = params[0]
    k2 = params[1]
    k3 = params[2]
    p1 = params[3]
    p2 = params[4]
    k4 = params[5]
    k5 = params[6]
    k6 = params[7]
    s1 = 0 #params[8]
    s2 = 0 #params[9]
    s3 = 0 #params[10]
    s4 = 0 #params[11]
    w = im1.shape[1]
    h = im1.shape[0]
    centre = np.array([(w - 1) / 2, (h - 1) / 2], dtype = 'float32')
    x1 = (pts[:,0] - centre[0]) / centre[0]
    y1 = (h / w) * (pts[:,1] - centre[1]) / centre[1]
    r = (x1**2 + y1**2)**0.5
    r2 = r**2
    r4 = r**4
    r6 = r**6
    
    x1_d = x1 * (1 + k1*r2 + k2*r4 + k3*r6)/(1 + k4*r2 + k5*r4 + k6*r6) + 2*p1*x1*y1 + p2*(r2*2*x1**2) + s1*r2 + s2*r4
    y1_d = y1 * (1 + k1*r2 + k2*r4 + k3*r6)/(1 + k4*r2 + k5*r4 + k6*r6) + 2*p2*x1*y1 + p1*(r2*2*y1**2) + s3*r2 + s4*r4
    x1_d = x1_d * centre[0] + centre[0]
    y1_d = (w / h) * y1_d * centre[1] + centre[1]
    pts_d = np.stack([x1_d, y1_d], axis = 0).T
    return pts_d
  
  def homography(pts1, pts2):
    pts1 = pts1.reshape(-1,1,2)
    pts2 = pts2.reshape(-1,1,2)
    hmat, mask = cv2.findHomography(pts1, pts2)
    pts1 = pts1.reshape(-1,2)
    pts2 = pts2.reshape(-1,2)

    pts1 = np.insert(pts1, 2, 1, axis = 1)
    pts1 = np.dot(hmat, pts1.T).T
    pts1[:,0] = pts1[:,0] / pts1[:,2]
    pts1[:,1] = pts1[:,1] / pts1[:,2]
    pts1 = pts1[:,0:2]
    rmse = np.mean(((pts1[:,0] - pts2[:,0])**2 + (pts1[:,1] - pts2[:,1])**2)**0.5) 

    return pts1, rmse, hmat

  def distort_rmse(params):
    pts1_d = distort(pts1, params)
    pts1_dh = homography(pts1_d, pts2)
    return pts1_dh[1]
  
  res = opt.minimize(distort_rmse, x0 = [0] * 8, method = 'Nelder-Mead')

  height, width, channels = im1.shape
  map_x, map_y  = np.meshgrid(np.arange(width), np.arange(height))
  grid = np.stack([map_x.flatten(), map_y.flatten()]).T
  p = copy.copy(res.x)
  p = -p
  grid_d = distort(grid, p)
  map_d = grid_d.T.reshape([2, height, width]).astype('float32')
  im1_d = cv2.remap(im1, map_d[0,:,:], map_d[1,:,:], interpolation = cv2.INTER_LINEAR)
  pts1_d = distort(pts1, res.x)
  pts1_dh, rmse, hmat = homography(pts1_d, pts2)
  
  im1_dh = cv2.warpPerspective(im1_d, hmat, (im2.shape[1], im2.shape[0]))
  cv2.imwrite(result_path, im1_dh)
  return im1_dh

# example

src = "1009.jpeg"
dst = "1908.jpeg"
rslt = "result_1009.png"

# result
im1_dh = homography_lensdist(src, dst, rslt)

# make a diff image
im2 = cv2.imread(dst)
im_diff = im2.astype(int) - im1_dh.astype(int)
cv2.imwrite("diff.png", np.abs(im_diff))
