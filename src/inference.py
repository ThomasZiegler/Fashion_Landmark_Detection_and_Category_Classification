from src.dataset import DeepFashionCAPDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const, unnormalize_image
from tensorboardX import SummaryWriter
import matplotlib.artist as artists
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
import os
import sys
import time



def transparent_cmap(cmap, N=255):
    """
        Transparent color map to overlay heatmap on image.
        Source: https://stackoverflow.com/questions/42481203/heatmap-on-top-of-image
    """
    new_cmap = cmap
    new_cmap._init()
    new_cmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return new_cmap

if __name__ == '__main__':
    parse_args_and_merge_const()

    random_state = np.random.RandomState(const.RANDOM_SEED)

    if os.path.exists('models') is False:
        os.makedirs('models')

    df = pd.read_csv(const.base_path + const.USE_CSV)
    inf_df = df

    inf_dataset = DeepFashionCAPDataset(inf_df,
                                        random_state=random_state,
                                        mode=const.DATASET_PROC_METHOD_INF,
                                        base_path = const.base_path)
    inf_dataloader = torch.utils.data.DataLoader(inf_dataset,
                                                 batch_size=const.INF_BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=6)
    inf_step = len(inf_dataloader)

    net = const.USE_NET(const.USE_IORN)
    net = net.to(const.device)
    net.load_state_dict(torch.load(const.INIT_MODEL), strict=False)

    writer = SummaryWriter(const.INF_DIR)

    inf_step = len(inf_dataloader)

    with torch.no_grad():
        net.eval()
        evaluator = const.EVALUATOR()
        for sample_idx, sample in enumerate(inf_dataloader):
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = net(sample)
            all_lines = evaluator.add(output, sample)
            gt = all_lines[0][0]
            pred = all_lines[0][1]
            correct = all_lines[0][-1]

#            _, pred = output['category_output'].topk(5, 1, True, True)
#            pred.t()
#            print(pred)
#            print(sample['category_label'])
#
#            # add pictures
#            if (sample_idx + 1) % 5 == 0:
            category_type = sample['category_type'][0].cpu().numpy()

            lm_size = int(output['lm_pos_map'].shape[2])
            heatmaps_pred = output['lm_pos_map'][0,:,:,:].cpu().detach().numpy()
            lm_pos_pred = output['lm_pos_output'][0,:,:].cpu().detach().numpy()*lm_size

            nr_img, height, width = heatmaps_pred.shape
            y, x = np.mgrid[0:height, 0:width]
            new_cmap = transparent_cmap(plt.cm.Reds)

            image = unnormalize_image(sample['image'][0,:,:,:].cpu())
#            writer.add_image('input_image', image, sample_idx)

            # [C,H,W] => [H,W,C] for Matplotlib
            image = image.numpy()
            image = image.transpose((1,2,0))

            text_pred = TextArea('gt: '+gt, textprops=dict(color='blue'))
            if correct==1:
                text_gt = TextArea('pred: '+pred, textprops=dict(color='green'))
            else:
                text_gt = TextArea('pred: '+pred, textprops=dict(color='red'))
            box = HPacker(children=[text_pred, text_gt], align='left', pad=5, sep=5)

            fig = plt.figure(10, figsize=(5,5))
            ax = fig.add_subplot(111)
            ax.imshow(image)

            anchored_box = AnchoredOffsetbox(loc=2,
                                             child=box, pad=0.,
                                             frameon=True,
                                             bbox_to_anchor=(0.,1.1),
                                             bbox_transform=ax.transAxes,
                                             borderpad=0.)
            ax.add_artist(anchored_box)
            fig.subplots_adjust(top=0.8)
#            fig.subplots_adjust(top=0.8)

            writer.add_figure('input_image', fig, sample_idx)


            for i, visible in enumerate(sample['landmark_vis'][0]):
                fig = plt.figure(i, figsize=(5,5))
                fig.suptitle(const.lm2name[i])

                # only plot heatmaps when landmark is in cloth type
                do_plot = False
                if (i==0 or i==1) and (category_type==0 or category_type==2):
                    do_plot = True
                elif (i==2 or i==3) and (category_type==0 or category_type==2):
                    do_plot = True
                elif (i==4 or i==5) and (category_type==1 or category_type==2):
                    do_plot = True
                elif (i==6 or i==7):
                    do_plot = True

                if do_plot:
                    ax = fig.add_subplot(111)
                    ax.imshow(image)
                    cb = ax.contourf(x, y, heatmaps_pred[i,:,:].reshape(x.shape[0], y.shape[1]), 15, cmap=new_cmap)
    #                ax.imshow(heatmaps_pred[i,:,:], cmap='gray')
                    ax.set_title('prediction')
                    ax.scatter(lm_pos_pred[i,0], lm_pos_pred[i,1], s=40, marker='x', c='r')

                writer.add_figure('heatmaps/{}'.format(const.lm2name[i]), fig, sample_idx)

            print('Val Step [{}/{}]'.format(sample_idx + 1, inf_step))

        ret = evaluator.evaluate()

        for topk, accuracy in ret['category_accuracy_topk'].items():
            print('metrics/category_top{}'.format(topk), accuracy)
            writer.add_scalar('metrics/category_top{}'.format(topk), accuracy, sample_idx)


