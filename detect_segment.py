import matplotlib.pyplot as plt
import cv2
import numpy as np

def convert_binary_img(img,png):
  if len(img.shape) == 2:
    min_value = float("inf")
    max_value = -float("inf")
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        if img[i,j] < min_value:
          min_value = img[i,j]
        if img[i,j] > max_value:
          max_value = img[i,j]
    
    thesh = (min_value+max_value)/2.0
    if png:
      _,img_binary = cv2.threshold(img,thesh,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
      _,img_binary = cv2.threshold(img,thesh,255,cv2.THRESH_BINARY)
  elif len(img.shape) == 3:
    min_value = float("inf")
    max_value = -float("inf")
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        if img[i,j][0] < min_value:
          min_value = img[i,j][0]
        if img[i,j][0] > max_value:
          max_value = img[i,j][0]
    thesh = (min_value+max_value)/2.0
    if png:
      _,img_binary = cv2.threshold(img,thesh,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
      _,img_binary = cv2.threshold(img,thesh,255,cv2.THRESH_BINARY)
  return img_binary

import math
def find_corners(contour):
  top_left = [float("inf"),float("inf")]
  bot_right = [-float("inf"),-float("inf")]
  for point in contour:
    if ((point[0])[0]+(point[0])[1]) <= (top_left[0]+top_left[1]):
      top_left[0] = (point[0])[0]
      top_left[1] = (point[0])[1]
    if ((point[0])[0]+(point[0])[1]) >= (bot_right[0]+bot_right[1]):
      bot_right[0] = (point[0])[0]
      bot_right[1] = (point[0])[1]
  return top_left,bot_right

import math

def rectangle_filter(contours_list):
  rectangle_list = []
  for sub in contours_list:
    top_left,bot_right = find_corners(sub)
    top_right = [-float("inf"),float("inf")]
    bot_left = [float("inf"),-float("inf")]
    area = cv2.contourArea(sub)
    top_right[0] = bot_right[0]
    top_right[1] = top_left[1]
    bot_left[0] = top_left[0]
    bot_left[1] = bot_right[1]
    row_max = bot_right[1]
    row_min = top_left[1]
    col_max = bot_right[0]
    col_min = top_left[0]
    outlier = 0
    for point in sub:
      if (point[0])[1] >= row_max+5 or (point[0])[1] <= row_min-5:
        outlier+=1
      elif (point[0])[0] >= col_max+5 or (point[0])[0] <= col_min-5:
        outlier+=1
    side_width = math.sqrt(pow(abs(bot_left[0]-bot_right[0]),2)+pow(abs(bot_left[1]-bot_right[1]),2))
    side_height = math.sqrt(pow(abs(top_left[0]-bot_left[0]),2)+pow(abs(top_left[1]-bot_left[1]),2))
    std_rectangle_are = side_width*side_height
    if_rectangle = True
    if outlier > (len(sub)/3.0):
      if_rectangle = False
    if area < 0.8*std_rectangle_are or area > 1.2*std_rectangle_are:
      if_rectangle = False
    if abs(top_left[1]-top_right[1]) > 10:
      if_rectangle = False
    if abs(bot_left[1]-bot_right[1]) > 10:
      if_rectangle = False
    if abs(top_left[0]-bot_left[0]) > 10:
      if_rectangle = False
    if abs(top_right[0]-bot_right[0]) > 10:
      if_rectangle = False
    if side_width < 30:
      if_rectangle = False
    if side_height < 30:
      if_rectangle = False
    if float(side_width)/side_height < 1:
      if_rectangle = False
    if float(side_width)/side_height > 3:
      if_rectangle = False
    if if_rectangle == True:
      rectangle_list.append(sub)
  return rectangle_list

def possible_contoures(img,kernel_size_parameter):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  img_size = gray_img.shape
  kernel = np.ones((int(img_size[0]/kernel_size_parameter),int(img_size[1]/kernel_size_parameter)),np.uint8)
  dilation_Erosion_img = cv2.morphologyEx(gray_img,cv2.MORPH_OPEN,kernel)
  diff_img = cv2.absdiff(dilation_Erosion_img,gray_img)
  binary_img = convert_binary_img(diff_img,False)
  canny_img = cv2.Canny(binary_img,255,255/2)
  filter_kernel = np.ones((10,25),np.uint8)
  closing_img = cv2.morphologyEx(canny_img,cv2.MORPH_CLOSE,filter_kernel)
  open_img = cv2.morphologyEx(closing_img,cv2.MORPH_OPEN,filter_kernel)
  img_copy = img.copy()
  _, contours_s,_ = cv2.findContours(open_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  rect_list = rectangle_filter(contours_s)
  return rect_list

def segmentation_index_row(seg,contours):
  row_list = []
  count_1 = 0
  minimum_count_row = float("inf")

  for i in range(seg.shape[0]):
    for j in range(seg.shape[1]):
      if seg[i,j] == 0:
        count_1 += 1
    if minimum_count_row > count_1:
      minimum_count_row = count_1
    row_list.append(count_1)
    count_1 = 0
  credict_point_row = []

  if minimum_count_row == 0:
    minimum_count_row = 1
  for i in range(len(row_list)):
    if not (i == 0 or i == len(row_list)-1):
      if row_list[i] < (minimum_count_row*3):
        if row_list[i+1] > row_list[i]:
          if not i-1 in credict_point_row:
            credict_point_row.append(i)
          #else:
            #if i-1 in credict_point_row:
              #credict_point_row.remove(i-1)
            #credict_point_row.append(i)
  credict_point_row.sort()
  return credict_point_row,row_list

def segmentation_index_col(seg,credict_point_row,contours):
  min_row = float("inf")
  max_row = -float("inf")
  max_interval = 0
  for i in range(0,len(credict_point_row)-1):
    row_s = credict_point_row[i]
    row_e = credict_point_row[i+1]
    dis = abs(row_e-row_s)
    if dis > max_interval:
      max_interval = dis
      min_row = credict_point_row[i]
      max_row = credict_point_row[i+1]
  if min_row == float("inf") or max_row == -float("inf"):
    digits = seg[:,:]
  elif abs(min_row-max_row) < seg.shape[0]/3.0:
    digits = seg[:,:]
  else:
    digits = seg[min_row:max_row,:]
  col_list = []
  count_2 = 0
  minimum_count = float("inf")
  for j in range(digits.shape[1]):
    for i in range(digits.shape[0]):
      if digits[i,j] == 0:
        count_2 += 1
    if minimum_count > count_2:
      minimum_count = count_2
    col_list.append(count_2)
    count_2 = 0

  credict_point_col = []
  if minimum_count == 0:
    minimum_count = 1
  for i in range(len(col_list)):
    if not (i == 0 or i == len(col_list)-1):
      if col_list[i] < (minimum_count*1):
        if col_list[i+1] > col_list[i]:
          if not i-1 in credict_point_col:
            credict_point_col.append(i)
          #else:
            #if i-1 in credict_point_col:
              #credict_point_col.remove(i-1)
            #credict_point_col.append(i)
  credict_point_col.append(digits.shape[0])          
  credict_point_col.sort()
  return credict_point_col,col_list

def segmentation_digit(seg,credict_point_row,credict_point_col):
  credict_point_col.sort()
  min_row = float("inf")
  max_row = -float("inf")
  max_interval = 0
  for i in range(0,len(credict_point_row)-1):
    row_s = credict_point_row[i]
    row_e = credict_point_row[i+1]
    dis = abs(row_e-row_s)
    if dis > max_interval:
      max_interval = dis
      min_row = credict_point_row[i]
      max_row = credict_point_row[i+1]
  if min_row == float("inf") or max_row == -float("inf"):
    digits = seg[:,:]
  elif abs(min_row-max_row) < seg.shape[0]/3.0:
    digits = seg[:,:]
  else:
    digits = seg[min_row:max_row,:]
  seg_list = []
  credict_point_col.insert(0,0)
  credict_point_col.append(digits.shape[1])
  for i in range(len(credict_point_col)-1):
    start = credict_point_col[i]
    end = credict_point_col[i+1]
    if not abs(start-end) < (digits.shape[1]/12):
      seg_list.append(digits[:,start:end])
  return seg_list,digits

from skimage.transform import  ProjectiveTransform

def transform_seg(img,top_left,bot_right,top_right,bot_left):
  point_list_left = [top_left,top_right,bot_left,bot_right]
  point_list_right = []
  point_list_right.append([0,0])
  height = abs(top_left[1]-bot_left[1])
  point_list_right.append([height*2,0])
  point_list_right.append([0,height])
  point_list_right.append([height*2,height])
  corners = np.array([point_list_left],dtype='float32')
  target = np.array([point_list_right],dtype='float32')
  H = cv2.getPerspectiveTransform(corners, target)
  transformed_img = cv2.warpPerspective(img,H,(height*2,height))
  return transformed_img

import math
def find_other_corners(row_min,row_max,col_min,col_max,contour,img_col):
  top_right = [-float("inf"),float("inf")]
  bot_left = [float("inf"),-float("inf")]
  min_dis = float("inf")
  max_dis = -float("inf")
  for point in contour:
    if ((point[0])[0] -(point[0])[1]) >= top_right[0]-top_right[1]:
      if ((point[0])[0] <= col_max+5) and ((point[0])[1] >= row_min-5):
        top_right[0] = (point[0])[0]
        top_right[1] = (point[0])[1]
    if ((point[0])[0] -(point[0])[1]) <= bot_left[0]-bot_left[1]:
      if ((point[0])[0] >= col_min-5) and ((point[0])[1] <= row_max+5):
        bot_left[0] = (point[0])[0]
        bot_left[1] = (point[0])[1]
  return top_right,bot_left

def zero_one_binary(img):
  copy_img = img.copy()
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i][j] > 255/2.0:
        copy_img[i][j] = 255
      else:
        copy_img[i][j] = 0
  return copy_img

import os
import shutil
from predict import predict_plate

def find_license_plate(path):
  img_path = path
  car_img = cv2.imread(img_path)
  car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)
  gray_img = cv2.cvtColor(car_img,cv2.COLOR_RGB2GRAY)
  #possible contours for the license plate
  rect_list = possible_contoures(car_img,10)

  #plot image with contours
  con_img = cv2.drawContours(car_img.copy(),rect_list,-1,(255,0,0),6)
  count = 0
  max_length = 0
  top_contour = []
  top_license_plate = ''
  top_bot_left = []
  top_top_left = []
  top_bot_right = []
  for contour in rect_list:
    binary_img = convert_binary_img(gray_img,True)
  #find possible position of the plate
    top_left,bot_right = find_corners(contour)
    top_right,bot_left = find_other_corners(top_left[1],bot_right[1],top_left[0],bot_right[0],contour,binary_img.shape[1])
#transform the license plate based on the index abow
    seg_seg = transform_seg(binary_img,top_left,bot_right,top_right,bot_left)
    ori_seg = transform_seg(car_img,top_left,bot_right,top_right,bot_left)

#find the segmentation index for license plate
    credict_point_row,row_list = segmentation_index_row(seg_seg,contour)
    credict_point_col,col_list = segmentation_index_col(seg_seg,credict_point_row,contour)

#segmented plate
    seg_list,digits = segmentation_digit(seg_seg,credict_point_row,credict_point_col)
    ori_seg_list,digits = segmentation_digit(ori_seg,credict_point_row,credict_point_col)
    root = "English/pred/img"+ str(count)
    folder_name = root+"/img"
    if not os.path.exists(folder_name):
      os.mkdir(folder_name)
    else:
      shutil.rmtree(folder_name)
      os.mkdir(folder_name)
    for i in range(len(seg_list)):
      seg_path = folder_name+"/seg"+str(i)+".jpg"
      zero_one = zero_one_binary(seg_list[i])
      #color_seg = cv2.cvtColor(zero_one,cv2.COLOR_RGB2BGR)
      cv2.imwrite(seg_path,zero_one)
    count+=1
    detection = predict_plate(root)
    if len(detection) > max_length:
      max_length = len(detection)
      top_contour = contour
      top_license_plate = detection
      top_bot_left = bot_left
      top_top_left = top_left
      top_bot_right = bot_right
  final_img = cv2.rectangle(car_img.copy(), (top_top_left[0], top_top_left[1]), (top_bot_right[0], top_bot_right[1]), (255,0,0), 3)
  cv2.putText(final_img,detection,(top_bot_left[0],top_bot_left[1]+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
  final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)
  con_img = cv2.cvtColor(con_img,cv2.COLOR_RGB2BGR)
  cv2.imwrite('final_img.jpg',final_img)
  cv2.imwrite('possible_plates.jpg',con_img)
  return rect_list,seg_list,bot_left,row_list,col_list

import sys

if __name__ == '__main__':
  path = sys.argv[1]
  if not os.path.isfile(path):
    print ("Input path was not valid file")
  rect_list,con_img,bot_left,row_list,col_list = find_license_plate(path)
