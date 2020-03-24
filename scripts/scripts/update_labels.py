import cv2
import numpy as np
import pandas as pd
import os
import sys



lm_pos_name = ['lm_lc_x', 'lm_lc_y', 'lm_rc_x', 'lm_rc_y',
               'lm_ls_x', 'lm_ls_y', 'lm_rs_x', 'lm_rs_y',
               'lm_lw_x', 'lm_lw_y', 'lm_rw_x', 'lm_rw_y',
               'lm_lh_x', 'lm_lh_y', 'lm_rh_x', 'lm_rh_y']



base_path = '/home/zieglert/ETH/SA-FL/data/CTU/'
datafile_name = base_path + 'new_info.csv'
dataframe = pd.read_csv(datafile_name)

new_dataframe = dataframe

for row_idx, row in dataframe.iterrows():

    x_1 = int(row['x_1'])
    x_2 = int(row['x_2'])
    y_1 = int(row['y_1'])
    y_2 = int(row['y_2'])



    for i in range(0, 16, 2):
        # lm position
        x = dataframe.at[row_idx, lm_pos_name[i]]+x_1
        y = dataframe.at[row_idx, lm_pos_name[i+1]]+y_1

        new_dataframe.at[row_idx, lm_pos_name[i]] = x
        new_dataframe.at[row_idx, lm_pos_name[i+1]] = y

new_dataframe.to_csv(base_path + 'new_new_info.csv', index=False)
