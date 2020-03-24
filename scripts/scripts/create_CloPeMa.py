# coding: utf-8
import pandas as pd
import numpy as np
import scipy.io as sp_io
import os
import re
import glob
import math
import cv2
from src.conf.clopema import base_path
#base_path = '/home/zieglert/ETH/SA-FL/data/CloPeMa/clothes_dataset_RH/'

# get all mat files in folder
files = [f for f in glob.glob(base_path + "**/[0-9]*/info.mat", recursive=True)]

dic_folder_category = {1: 't-shirt',
                       2: 't-shirt',
                       3: 't-shirt', # polo
                       4: 't-shirt', # polo
                       5: 't-shirt',
                       6: 'shirt', # short sleeve
                       7: 'shirt',
                       8: 'shirt',
                       9: 'shirt',
                       10: 'thick-sweater',
                       11: 'thick-sweater',
                       12: 'jean',
                       13: 'thick-sweater',
                       14: 'jean',
                       15: 'jean',
                       16: 'jean',
                       17: 'thick-sweater', # cardigan
                       18: 'thick-sweater',
                       19: 'jean',
                       20: 'jean',
                       21: 'towel',
                       22: 'shirt', # short sleeve
                       23: 'shirt',
                       24: 'towel',
                       25: 'towel',
                       26: 'jean',
                       27: 't-shirt', # polo
                       28: 't-shirt',
                       29: 't-shirt',
                       30: 't-shirt', # polo
                       31: 'jean',
                       32: 'shirt',
                       33: 'shirt', # short sleeve
                       34: 'shirt',
                       35: 'shirt', # short sleeve
                       36: 'jean',
                       37: 'jean',
                       38: 'towel',
                       39: 'towel',
                       40: 'towel',
                       41: 'towel',
                       42: 'towel',
                       43: 'towel',
                       44: 'towel',
                       45: 't-shirt',
                       46: 'thick-sweater',
                       47: 'thick-sweater',
                       48: 'thick-sweater',
                       49: 'thick-sweater',
                       50: 'thick-sweater'}

dic_category_label = {'t-shirt': 0,
                      'shirt': 1,
                      'thick-sweater': 2,
                      'jean': 3}

dic_category_type = {'t-shirt': 0,
                     'shirt': 0,
                     'thick-sweater': 0,
                     'jean': 1}


#dic_evaluation_status = {0: 'train',
#                         1: 'val',
#                         2: 'test'}

random_state = np.random.RandomState(17)

values=[]
for f in files:
    # get folder name
    folder = f.split("/")[-2]

    # load mat file
    mat_file = sp_io.loadmat(f)
    category_name = mat_file['category'][0]
    fabric = mat_file['fabric'][0]

    # Ignore towels
    if category_name == 'towel':
        continue

    # create data frame values
#    evaluation_status = dic_evaluation_status[random_state.choice(3, p=[0.6, 0.2, 0.2])]
    evaluation_status = folder
    category_label = dic_category_label[category_name]
    category_type = dic_category_type[category_name]

    x_1, y_1 = 850, 675
    x_2, y_2 = 3975, 2900

    landmark_postions = [0, 0] * 8
    landmark_visibilities = [0] * 8
    landmark_in_pic = [0] * 8
    attr = [0]*1000

    # set fabric as attribute
    if fabric == 'thin-cotton':
        attr[0]=0
    elif fabric == 'jaconet':
        attr[0]=1
    elif fabric == 'thick-knitting':
        attr[0]=2
    elif fabric == 'jean':
        attr[0]=3
    else:
        print('>>>>>>>', fabric)

    # get all images
    images = [img for img in glob.glob(base_path + folder + "/*rgb.png", recursive=True)]

    for img in images:
        # get relative path
        image_name = '/'.join(img.split("/")[-2:])
#        image = cv2.imread(base_path+image_name, 1)
#        cv2.namedWindow("image")
#        cv2.imshow("image", image)
#        cv2.waitKey(0)
#
#        evaluation_status = dic_evaluation_status[random_state.choice(3, p=[0.6, 0.2, 0.2])]

        line_value = [image_name, x_1, y_1, x_2, y_2, evaluation_status, category_label, category_name, category_type]
        line_value.extend(landmark_postions)
        line_value.extend(landmark_visibilities)
        line_value.extend(landmark_in_pic)
        line_value.extend(attr)

        values.append(line_value)


column_names = ['image_name', 'x_1', 'y_1', 'x_2', 'y_2', 'evaluation_status', 'category_label', 'category_name', 'category_type']

column_names.extend(['lm_lc_x', 'lm_lc_y', 'lm_rc_x', 'lm_rc_y',
             'lm_ls_x', 'lm_ls_y', 'lm_rs_x', 'lm_rs_y',
             'lm_lw_x', 'lm_lw_y', 'lm_rw_x', 'lm_rw_y',
             'lm_lh_x', 'lm_lh_y', 'lm_rh_x', 'lm_rh_y'])

column_names.extend([
    'lm_lc_vis', 'lm_rc_vis',
    'lm_ls_vis', 'lm_rs_vis',
    'lm_lw_vis', 'lm_rw_vis',
    'lm_lh_vis', 'lm_rh_vis',
])

column_names.extend([
    'lm_lc_in_pic', 'lm_rc_in_pic',
    'lm_ls_in_pic', 'lm_rs_in_pic',
    'lm_lw_in_pic', 'lm_rw_in_pic',
    'lm_lh_in_pic', 'lm_rh_in_pic',
])

column_names.extend(['attr_%d' % i for i in range(1000)])


df_info = pd.DataFrame(values, columns=column_names)
df_info.to_csv(base_path + 'info.csv', index=False)
