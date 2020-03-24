import cv2
import numpy as np
import pandas as pd
import os
import sys



lm2name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']


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


base_path = '/home/zieglert/ETH/SA-FL/data/CTU/'
datafile_name = base_path + 'info.csv'
dataframe = pd.read_csv(datafile_name)

if os.path.exists(base_path+'new_info.csv'):
    new_dataframe = pd.read_csv(base_path + 'new_info.csv')
else:
    new_dataframe = dataframe
    new_dataframe.to_csv(base_path + 'new_info.csv', index=False)

for row_idx, row in dataframe.iterrows():
    # skip line if landmarks already exist in new file
    if new_dataframe.at[row_idx, 'lm_lh_x'] != 0:
        continue

    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not
    bbox = []
    cropping = False

    x_1 = int(row['x_1'])
    x_2 = int(row['x_2'])
    y_1 = int(row['y_1'])
    y_2 = int(row['y_2'])

    bbox = [(x_1,y_1), (x_2,y_2)]

    image_name = row['image_name']
    image_path = base_path + image_name
    print(row_idx, image_name)
    sys.stdout.flush()

    category_type = row['category_type']

    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(image_path)

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    # load initial bbox
    cv2.rectangle(image, bbox[0], bbox[1], (0, 255, 0), 2)

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
                print(lm2name[lm_nr])
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

    new_dataframe.at[row_idx, 'x_1'] = bbox[0][0]
    new_dataframe.at[row_idx, 'y_1'] = bbox[0][1]
    new_dataframe.at[row_idx, 'x_2'] = bbox[1][0]
    new_dataframe.at[row_idx, 'y_2'] = bbox[1][1]

    for i in range(0, 16, 2):
        # lm position
        new_dataframe.at[row_idx, lm_pos_name[i]] = lm_pos[i//2][0]+bbox[0][0]
        new_dataframe.at[row_idx, lm_pos_name[i+1]] = lm_pos[i//2][1]+bbox[0][1]

        # lm visibility
        new_dataframe.at[row_idx, lm_vis_name[i//2]] = lm_vis[i//2]
        # lm in picture
        new_dataframe.at[row_idx, lm_inpic_name[i//2]] = lm_inpic[i//2]




    new_dataframe.to_csv(base_path + 'new_info.csv', index=False)

    # close all open windows
    cv2.destroyAllWindows()
