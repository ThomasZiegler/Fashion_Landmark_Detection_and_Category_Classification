from src.dataset import DeepFashionCAPDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const
from tensorboardX import SummaryWriter
import os
import time

if __name__ == '__main__':
    start_time = time.time()
    parse_args_and_merge_const()


    if const.RANDOM_SEED != None:
        torch.manual_seed(const.RANDOM_SEED)
        np.random.seed(const.RANDOM_SEED)
        random_state = np.random.RandomState(const.RANDOM_SEED)


#    const.DATASET_PROC_METHOD_TRAIN = 'ELASTIC_ROTATION_BBOXRESIZE'
#    const.DATASET_PROC_METHOD_TRAIN = 'ROTATION_BBOXRESIZE'
    const.DATASET_PROC_METHOD_TRAIN = 'BBOXRESIZE'
    const.USE_CSV = 'info.csv'
#    const.USE_CSV = 'debug_info.csv'
    if os.path.exists('models') is False:
        os.makedirs('models')

    print(const.base_path)
    df = pd.read_csv(const.base_path + const.USE_CSV)
    train_df = df[df['evaluation_status'] == 'val']
    train_dataset = DeepFashionCAPDataset(train_df,
                                          base_path=const.base_path,
                                          random_state=random_state,
                                          mode=const.DATASET_PROC_METHOD_TRAIN)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=const.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=4)
#    val_df = df[df['evaluation_status'] == 'val']
#    val_dataset = DeepFashionCAPDataset(val_df, mode=const.DATASET_PROC_METHOD_VAL)
#    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=const.VAL_BATCH_SIZE, shuffle=False, num_workers=1)

#    for i in range(100):
#        train_dataset.plot_sample(i)
#
    step = 0
    for i, sample in enumerate(train_dataloader):
        step += 1
        print(i,'================================================')
        for key in sample:
            sample[key] = sample[key].to(const.device)


    print('loading time: ', time.time()-start_time)
