# coding: utf-8
import pandas as pd
import numpy as np
import os
import re
import glob
import yaml
import math
import cv2
from src.conf.inf import base_path


# get all yaml files in folder
files = [f for f in glob.glob(base_path + "**/*.yaml", recursive=True)]

values = []

dic_category_label = {'bluse': 0,
                      'hoody': 1,
                      'pants': 2,
                      'polo': 3,
                      'polo-long': 4,
                      'skirt': 5,
                      'tshirt': 6,
                      'tshirt-long': 7}

dic_category_type = {'bluse': 0,
                     'hoody': 0,
                     'pants': 1,
                     'polo': 0,
                     'polo-long': 0,
                     'skirt': 1,
                     'tshirt': 0,
                     'tshirt-long': 0}


dic_evaluation_status = {0: 'train',
                         1: 'val',
                         2: 'test'}

random_state = np.random.RandomState(17)

values=[]
for f in files:
    # open yaml file
    with open(f, 'r') as stream:
        info = yaml.safe_load(stream)

    image_name = info['path_c']

    # check that image excists
    if not os.path.isfile(base_path+image_name):
#        print('image: {} does not exist'.format(base_path+image_name))
        continue

    category = info['type']

    # Ignore towels
    if category == 'towel':
        continue

    polygon_points = info['poly_c']

    image = cv2.imread(base_path+image_name, 1)
    height, width = image.shape[:2]

    x_1 = np.inf
    y_1 = np.inf
    x_2 = 0.
    y_2 = 0.


    for p in polygon_points:
        x_1 = np.minimum(p[0]-100, x_1)
        x_2 = np.maximum(p[0]+100, x_2)
        y_1 = np.minimum(p[1]-100, y_1)
        y_2 = np.maximum(p[1]+100, y_2)

    if math.isnan(x_1) or x_1 < 0.:
        x_1 = 0
    if math.isnan(y_1) or y_1 < 0.:
        y_1 = 0
    if math.isnan(x_2) or x_2 > width:
        x_2 = width
    if math.isnan(y_2) or y_2 > height:
        y_2 = height


    evaluation_status = dic_evaluation_status[random_state.choice(3, p=[0.6, 0.2, 0.2])]
    category_name = category
    category_label = dic_category_label[category_name]
    category_type = dic_category_type[category_name]

    landmark_postions = [0, 0] * 8
    landmark_visibilities = [0] * 8
    landmark_in_pic = [0] * 8
    attr = [0]*1000

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






#partition = pd.read_csv(base_path + 'Eval/list_eval_partition.txt', skiprows=1, sep='\s+')
#
#category = pd.read_csv(
#    base_path + 'Anno/list_category_img.txt', skiprows=1, sep='\s+')
#category_type = pd.read_csv(
#    base_path + 'Anno/list_category_cloth.txt', skiprows=1, sep='\s+')
#category_type['category_label'] = range(1, len(category_type) + 1)
#category = pd.merge(category, category_type, on='category_label')
#
## parse landmarks
#with open(base_path + 'Anno/list_landmarks.txt') as f:
#    f.readline()
#    f.readline()
#    values = []
#    for line in f:
#        info = re.split('\s+', line)
#        image_name = info[0].strip()
#        clothes_type = int(info[1])
#        # 1: ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
#        # 2: ["left waistline", "right waistline", "left hem", "right hem"]
#        # 3: ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].
#        landmark_postions = [(0, 0)] * 8
#        landmark_visibilities = [1] * 8
#        landmark_in_pic = [1] * 8
#        landmark_info = info[2:]
#        if clothes_type == 1:  # upper body
#            convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 7}
#        elif clothes_type == 2:
#            convert = {0: 4, 1: 5, 2: 6, 3: 7}
#        elif clothes_type == 3:
#            convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
#        for i in convert:
#            x = int(landmark_info[i * 3 + 1])
#            y = int(landmark_info[i * 3 + 2])
#            vis = int(landmark_info[i * 3])
#            if vis == 2:
#                in_pic = 0
#            elif vis == 1:
#                in_pic = 1
#            else:
#                in_pic = 1
#            if vis == 2:
#                vis = 0
#            elif vis == 1:
#                vis = 0
#            else:
#                vis = 1
#            landmark_postions[convert[i]] = (x, y)
#            landmark_visibilities[convert[i]] = vis
#            landmark_in_pic[convert[i]] = in_pic
#        tmp = []
#        for pair in landmark_postions:
#            tmp.append(pair[0])
#            tmp.append(pair[1])
#        landmark_postions = tmp
#
#        line_value = []
#        line_value.extend([image_name, clothes_type])
#        line_value.extend(landmark_postions)
#        line_value.extend(landmark_visibilities)
#        line_value.extend(landmark_in_pic)
#        values.append(line_value)
#
#name = ['image_name', 'clothes_type']
#name.extend(['lm_lc_x', 'lm_lc_y', 'lm_rc_x', 'lm_rc_y',
#             'lm_ls_x', 'lm_ls_y', 'lm_rs_x', 'lm_rs_y',
#             'lm_lw_x', 'lm_lw_y', 'lm_rw_x', 'lm_rw_y',
#             'lm_lh_x', 'lm_lh_y', 'lm_rh_x', 'lm_rh_y'])
#
#name.extend([
#    'lm_lc_vis', 'lm_rc_vis',
#    'lm_ls_vis', 'lm_rs_vis',
#    'lm_lw_vis', 'lm_rw_vis',
#    'lm_lh_vis', 'lm_rh_vis',
#])
#
#name.extend([
#    'lm_lc_in_pic', 'lm_rc_in_pic',
#    'lm_ls_in_pic', 'lm_rs_in_pic',
#    'lm_lw_in_pic', 'lm_rw_in_pic',
#    'lm_lh_in_pic', 'lm_rh_in_pic',
#])
#
#landmarks = pd.DataFrame(values, columns=name)
#
## attribute
#attr = pd.read_csv(base_path + 'Anno/list_attr_img.txt', skiprows=2, sep='\s+', names=['image_name'] + ['attr_%d' % i for i in range(1000)])
#attr.replace(-1, 0, inplace=True)
#
## bbox
#bbox = pd.read_csv(base_path + 'Anno/list_bbox.txt', skiprows=1, sep='\s+')
#
## merge all information
#assert (category['category_type'] == landmarks['clothes_type']).all()
#landmarks = landmarks.drop('clothes_type', axis=1)
#category['category_type'] = category['category_type'] - 1  # 0-based
#category['category_label'] = category['category_label'] - 1  # 0-based
#info_df = pd.merge(category, landmarks, on='image_name', how='inner')
#info_df = pd.merge(info_df, attr, on='image_name', how='inner')
#info_df = pd.merge(partition, info_df, on='image_name', how='inner')
#info_df = pd.merge(bbox, info_df, on='image_name', how='inner')
#
#info_df.to_csv(base_path + 'info.csv', index=False)
