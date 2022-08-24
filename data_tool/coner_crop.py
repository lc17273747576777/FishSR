import math
import cv2
import numpy as np
import math
import os

# 定义FisheyeIm类
HR_path = "F:/WJ_project/dataset/realSR/ori/Flickr2K_HR"
LR_path = "F:/WJ_project/dataset/realSR/ori/Flickr2K_LR"
output_path = "F:/WJ_project/dataset/realSR/crop576"
start_num = 0

def crop_coner(img, idx, cropsize=576, border=4):
    h, w, c = img.shape
    assert (h > cropsize and w > cropsize)
    if idx == 1:
        outimg = img[border:border+cropsize, border:border+cropsize, :]
        return outimg
    elif idx == 2:
        outimg = cv2.flip(img[border:border+cropsize, -border-cropsize:-border, :], flipCode=1)
        return outimg
    elif idx == 3:
        outimg = cv2.flip(img[-border-cropsize:-border, border:border+cropsize, :], flipCode=0)
        return outimg
    elif idx == 4:
        outimg = cv2.flip(img[-border-cropsize:-border, -border-cropsize:-border, :], flipCode=-1)
        return outimg
    else:
        print("wrong idx")


output_HR_path = os.path.join(output_path, 'HR')
output_LR_path = os.path.join(output_path, 'LR')
if not os.path.exists(output_HR_path):
    os.makedirs(output_HR_path)
if not os.path.exists(output_LR_path):
    os.makedirs(output_LR_path)

for dirpath, dirnames, filenames in os.walk(HR_path):
    for filename in filenames:
        if start_num > 1:
            start_num -= 1
            continue
        print(filename)
        HR_file = os.path.join(dirpath, filename)
        LR_file = HR_file.replace('.png', '_sim.png')
        LR_file = LR_file.replace('Flickr2K_HR', 'Flickr2K_LR')
        input_img_HR = cv2.imread(HR_file)
        input_img_LR = cv2.imread(LR_file)
        filename = filename.replace('.png', '')
        filename = filename.replace('._', '')
        for idx in range(1, 5):
            crop_HR = crop_coner(input_img_HR, cropsize=576, idx=idx)
            crop_LR = crop_coner(input_img_LR, cropsize=576, idx=idx)
            # img_name_fisheye = img_name + str(k) + '_fisheye.jpg'
            # save_img_name0 = os.path.join(output_path, img_name_fisheye)
            # cv2.imwrite(save_img_name0, output)
            # distort = Distort_restore(output, 1920, 1080)
            crop_name = filename + '_' + str(idx) + '.png'
            HR_save_path = os.path.join(output_HR_path, crop_name)
            LR_save_path = os.path.join(output_LR_path, crop_name)
            cv2.imwrite(HR_save_path, crop_HR)
            cv2.imwrite(LR_save_path, crop_LR)


