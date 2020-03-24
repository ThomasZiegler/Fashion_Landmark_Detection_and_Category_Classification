# coding: utf-8
import pandas as pd
import numpy as np
import os
import re
import glob
import math
import sys
import cv2
#from src.conf.robot import base_path
base_path = '/home/zieglert/ETH/SA-FL/data/robot_upside_down/cropped/'

# get all mat files in folder
files = [f for f in glob.glob(base_path + "*.jpg", recursive=True)]



lm_to_name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']


dic_category_label = {'Anorak': 0,
                      'Blazer': 1,
                      'Blouse': 2,
                      'Bomber': 3,
                      'Button-Down': 4,
                      'Cardigan': 5,
                      'Flannel': 6,
                      'Halter': 7,
                      'Henley': 8,
                      'Hoodie': 9,
                      'Jacket': 10,
                      'Jersey': 11,
                      'Parka': 12,
                      'Peacoat': 13,
                      'Poncho': 14,
                      'Sweater': 15,
                      'Tank': 16,
                      'Tee': 17,
                      'Top': 18,
                      'Turtleneck': 19,
                      'Capris': 20,
                      'Chinos': 21,
                      'Culottes': 22,
                      'Cutoffs': 23,
                      'Gauchos': 24,
                      'Jeans': 25,
                      'Jeggings': 26,
                      'Jodhpurs': 27,
                      'Joggers': 28,
                      'Leggings': 29,
                      'Sarong': 30,
                      'Shorts': 31,
                      'Skirt': 32,
                      'Sweatpants': 33,
                      'Sweatshorts': 34,
                      'Trunks': 35,
                      'Caftan': 36,
                      'Cape': 37,
                      'Coat': 38,
                      'Coverup': 39,
                      'Dress': 40,
                      'Jumpsuit': 41,
                      'Kaftan': 42,
                      'Kimono': 43,
                      'Nightdress': 44,
                      'Onesie': 45,
                      'Robe': 46,
                      'Romper': 47,
                      'Shirtdress': 48,
                      'Sundress': 49}

dic_category_robot = {'Tank': 0,
                      'Tee': 1,
                      'Jeans': 2,
                      'Jacket': 3,
                      'Sweater': 4,
                      'Hoodie': 5}


dic_category_type = {'Anorak': 0,
                     'Blazer': 0,
                     'Blouse': 0,
                     'Bomber': 0,
                     'Button-Down': 0,
                     'Cardigan': 0,
                     'Flannel': 0,
                     'Halter': 0,
                     'Henley': 0,
                     'Hoodie': 0,
                     'Jacket': 0,
                     'Jersey': 0,
                     'Parka': 0,
                     'Peacoat': 0,
                     'Poncho': 0,
                     'Sweater': 0,
                     'Tank': 0,
                     'Tee': 0,
                     'Top': 0,
                     'Turtleneck': 0,
                     'Capris': 1,
                     'Chinos': 1,
                     'Culottes': 1,
                     'Cutoffs': 1,
                     'Gauchos': 1,
                     'Jeans': 1,
                     'Jeggings': 1,
                     'Jodhpurs': 1,
                     'Joggers': 1,
                     'Leggings': 1,
                     'Sarong': 1,
                     'Shorts': 1,
                     'Skirt': 1,
                     'Sweatpants': 1,
                     'Sweatshorts': 1,
                     'Trunks': 1,
                     'Caftan': 2,
                     'Cape': 2,
                     'Coat': 2,
                     'Coverup': 2,
                     'Dress': 2,
                     'Jumpsuit': 2,
                     'Kaftan': 2,
                     'Kimono': 2,
                     'Nightdress': 2,
                     'Onesie': 2,
                     'Robe': 2,
                     'Romper': 2,
                     'Shirtdress': 2,
                     'Sundress': 2}



#dic_category_label = {'bluse': 0,
#                      'hoody': 1,
#                      'pants': 2,
#                      'polo': 3,
#                      'polo-long': 4,
#                      'skirt': 5,
#                      'tshirt': 6,
#                      'tshirt-long': 7}
#
#dic_category_type = {'bluse': 0,
#                     'hoody': 0,
#                     'pants': 1,
#                     'polo': 0,
#                     'polo-long': 0,
#                     'skirt': 1,
#                     'tshirt': 0,
#                     'tshirt-long': 0}
#
dic_evaluation_status = {0: 'train',
                         1: 'val',
                         2: 'test'}



lm_pos_name = ['lm_lc_x', 'lm_lc_y', 'lm_rc_x', 'lm_rc_y',
               'lm_ls_x', 'lm_ls_y', 'lm_rs_x', 'lm_rs_y',
               'lm_lw_x', 'lm_lw_y', 'lm_rw_x', 'lm_rw_y',
               'lm_lh_x', 'lm_lh_y', 'lm_rh_x', 'lm_rh_y']



lm_vis_name = ['lm_lc_vis', 'lm_rc_vis',
               'lm_ls_vis', 'lm_rs_vis',
               'lm_lw_vis', 'lm_rw_vis',
               'lm_lh_vis', 'lm_rh_vis']


lm_inpic_name = ['lm_lc_in_pic', 'lm_rc_in_pic',
                 'lm_ls_in_pic', 'lm_rs_in_pic',
                 'lm_lw_in_pic', 'lm_rw_in_pic',
                 'lm_lh_in_pic', 'lm_rh_in_pic']


column_names = ['image_name', 'x_1', 'y_1', 'x_2', 'y_2', 'evaluation_status', 'category_label', 'category_name', 'category_type']

column_names.extend(lm_pos_name)
column_names.extend(lm_vis_name)
column_names.extend(lm_inpic_name)
column_names.extend(['attr_%d' % i for i in range(1000)])

if os.path.exists(base_path+'info.csv'):
    dataframe = pd.read_csv(base_path + 'info.csv')
else:
    dataframe = pd.DataFrame(columns=column_names)
    dataframe.to_csv(base_path + 'info.csv', index=False)


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global tmp_bbox, bbox, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp_bbox = []
        tmp_bbox = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        tmp_bbox.append((x, y))
        cropping = False
        bbox = tmp_bbox

        # draw a rectangle around the region of interest
        cv2.rectangle(image, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

def click_landmark(event, x, y, flags, param):
    global lm_pos, next_lm, lm_inpic, lm_vis

    if event == cv2.EVENT_LBUTTONDOWN:
        lm_pos.append((x, y))
        lm_inpic.append(1)
        lm_vis.append(1)
        next_lm = True

    if event == cv2.EVENT_RBUTTONDOWN:
        lm_pos.append((0,0))
        lm_inpic.append(0)
        lm_vis.append(0)
        next_lm = True


random_state = np.random.RandomState(17)
values=[]
for row_idx, f in enumerate(files):
    print(row_idx)
    # get image_name from absolut path
    image_name = '/'.join(f.split("/")[-1:])

    # skip if image already in dataset
    if dataframe['image_name'].str.contains(image_name).sum():
        continue

    image = cv2.imread(base_path+image_name, 1)


    clone = image.copy()
    random_state = np.random.RandomState(17)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    cropping = False
    while True:
        while cropping:
            cv2.waitKey(10)

        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            if len(bbox) == 2:
                break

    waiting_for_correct_input = True
    while(waiting_for_correct_input):
        print('select category type: ', ['{}: {}'.format(i, key) for i, key in enumerate(dic_category_robot.keys())] )

        category_id = input('Category id: ')
        category_name = [key for i, key in enumerate(dic_category_robot.keys()) if i == int(category_id)][0]
        print(category_name)
        if category_name in dic_category_label.keys():
            category_label = dic_category_label[category_name]
            category_type = dic_category_type[category_name]
            evaluation_status = 'val'
            waiting_for_correct_input = False


    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(bbox) == 2:
        roi = clone[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        cv2.imshow("ROI", roi)


    clone = roi.copy()
    cv2.setMouseCallback("ROI", click_landmark)
    next_lm = False
    lm_nr = 0
    lm_pos = []
    lm_vis = []
    lm_inpic = []
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord("r"):
            # reset
            next_lm = False
            lm_nr = 0
            lm_pos = []
            lm_vis = []
            lm_inpic = []
            roi = clone.copy()

        elif key == ord("c"):
            break;

        while lm_nr < 8:
            do_lm = False
            if (lm_nr==0 or lm_nr==1) and (category_type==0 or category_type==2):
                do_lm = True
            elif (lm_nr==2 or lm_nr==3) and (category_type==0 or category_type==2):
                do_lm = True
            elif (lm_nr==4 or lm_nr==5) and (category_type==1 or category_type==2):
                do_lm = True
            elif (lm_nr==6 or lm_nr==7):
                do_lm = True

            if do_lm:
                print(lm_to_name[lm_nr])
                sys.stdout.flush()
                while not next_lm:
                    # wait for a keypress
                    cv2.imshow("ROI", roi)
                    key = cv2.waitKey(100) & 0xFF

                next_lm = False
            else:
                lm_pos.append((0,0))
                lm_inpic.append(0)
                lm_vis.append(0)

            print( lm_pos[lm_nr])
            print('-----------------------')
            sys.stdout.flush()
            cv2.circle(roi, lm_pos[lm_nr], 15, (0, 0, 255), -1)
            cv2.imshow("ROI", roi)
            lm_nr += 1

    dataframe.at[row_idx, 'image_name'] = image_name
    dataframe.at[row_idx, 'x_1'] = bbox[0][0]
    dataframe.at[row_idx, 'y_1'] = bbox[0][1]
    dataframe.at[row_idx, 'x_2'] = bbox[1][0]
    dataframe.at[row_idx, 'y_2'] = bbox[1][1]
    dataframe.at[row_idx, 'evaluation_status'] = evaluation_status
    dataframe.at[row_idx, 'category_name'] = category_name
    dataframe.at[row_idx, 'category_label'] = category_label
    dataframe.at[row_idx, 'category_type'] = category_type

    for i in range(0, 16, 2):
        # lm position
        dataframe.at[row_idx, lm_pos_name[i]] = lm_pos[i//2][0]+bbox[0][0]
        dataframe.at[row_idx, lm_pos_name[i+1]] = lm_pos[i//2][1]+bbox[0][1]

        # lm visibility
        dataframe.at[row_idx, lm_vis_name[i//2]] = lm_vis[i//2]
        # lm in picture
        dataframe.at[row_idx, lm_inpic_name[i//2]] = lm_inpic[i//2]

    for i in range(1000):
        dataframe.at[row_idx, 'attr_%d' % i] = 0

    x_1 = bbox[0][0]
    y_1 = bbox[0][1]
    x_2 = bbox[0][0]
    y_2 = bbox[1][1]


    dataframe.to_csv(base_path + 'info.csv', index=False)

    # close all open windows
    cv2.destroyAllWindows()

