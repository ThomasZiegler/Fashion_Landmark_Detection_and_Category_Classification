from src.dataset import DeepFashionCAPDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from src import const
from src.utils import parse_args_and_merge_const, unnormalize_image
from tensorboardX import SummaryWriter
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker
import os
import sys
import random


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
    # parse flags from const.py file and overwrite the ones given in the conf/xxx.py file
    parse_args_and_merge_const()

    # set stdout to file
    if hasattr(const, 'STDOUT_FILE') and const.STDOUT_FILE != None:
        dirname = os.path.dirname(const.STDOUT_FILE)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        sys.stdout = open(const.STDOUT_FILE, 'w')


    # Initialize random generators with given seed
    if const.RANDOM_SEED != None:
        torch.manual_seed(const.RANDOM_SEED)
        np.random.seed(const.RANDOM_SEED)
        random.seed(const.RANDOM_SEED)
        random_state = np.random.RandomState(const.RANDOM_SEED)

    if os.path.exists('models') is False:
        os.makedirs('models')

    # order the item labels randomly for each category
    random_label_order = []
    random_label_order.append(random_state.choice(const.SWEATER_LABELS, 10, replace=False))
    random_label_order.append(random_state.choice(const.JEAN_LABELS, 10, replace=False))
    random_label_order.append(random_state.choice(const.SHIRT_LABELS, 10, replace=False))
    random_label_order.append(random_state.choice(const.TSHIRT_LABELS, 10, replace=False))
    random_label_order = np.array(random_label_order)

    df = pd.read_csv(const.base_path + const.USE_CSV)


    total_results = {'best_epoch_nr': [0,0,0,0,0],
                     'best_category_top1': [0,0,0,0,0],
                     'best_category_top2': [0,0,0,0,0],
                     'best_category_top3': [0,0,0,0,0],
                     'best_attr_top1': [0,0,0,0,0],
                     'best_attr_top2': [0,0,0,0,0],
                     'best_attr_top3': [0,0,0,0,0]}

    for validation_cycle in range(5):
        # get randomly 2 out of 10 labels without replacement for each category
        validation_labels = random_label_order[:,2*validation_cycle:2*validation_cycle+1].ravel()

        train_df = df[~df['evaluation_status'].isin(validation_labels)]
        train_dataset = DeepFashionCAPDataset(train_df,
                                              random_state=random_state,
                                              mode=const.DATASET_PROC_METHOD_TRAIN,
                                              base_path=const.base_path)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=const.BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=4)
        val_df = df[df['evaluation_status'].isin(validation_labels)]
        val_dataset = DeepFashionCAPDataset(val_df,
                                            random_state=random_state,
                                            mode=const.DATASET_PROC_METHOD_VAL,
                                            base_path=const.base_path)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=const.VAL_BATCH_SIZE,
                                                     shuffle=False,
                                                     num_workers=4)

        # load network either based on VGG or IORN
        net = const.USE_NET(const.USE_IORN)
        net = net.to(const.device)

        # load LM network if given
        if hasattr(const, 'LM_INIT_MODEL') and const.LM_INIT_MODEL is not None:
            net.load_state_dict(torch.load(const.LM_INIT_MODEL), strict=False)

        # Freeze VGG layer Conv1 - Conv4 and LM Network
        if hasattr(const, 'FREEZE_LM_NETWORK') and const.FREEZE_LM_NETWORK:
            child_nr = 0
            for child in net.children():
                child_nr += 1
                # VGG Conv1 - Conv4 and LM network are child 1 & 2
                if child_nr <=2:
                    for param in child.parameters():
                        param.requires_grad = False
            print('VGG16 Conv1-Conv4 and LM network are froozen')
        # setup optimizer
        learning_rate = const.LEARNING_RATE
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()),
                                     lr=learning_rate)

        writer = SummaryWriter(const.TRAIN_DIR)

        # init variables
        total_step = len(train_dataloader)
        val_step = len(val_dataloader)
        step = 0
        best_epoch = 0
        best_lm = 10
        best_cat = {}
        best_attr = {}
        min_epoch_loss = float('Inf')
        for epoch in range(const.NUM_EPOCH):
            net.train()
            epoch_loss = 0
            for sample_idx, sample in enumerate(train_dataloader):
                step += 1
                # move samples to GPU/CPU
                for key in sample:
                    sample[key] = sample[key].to(const.device)
                output = net(sample)
                loss = net.cal_loss(sample, output)
                epoch_loss += loss['all'].item()

                optimizer.zero_grad()
                loss['all'].backward()
                optimizer.step()


                # log scalars
                if (sample_idx + 1) % const.LOG_INTERVAL_SCALAR == 0:
                    if 'category_loss' in loss:
                        writer.add_scalar('loss/category_loss', loss['category_loss'], step)
                        writer.add_scalar('loss_weighted/category_loss', loss['weighted_category_loss'], step)
                    if 'attr_loss' in loss:
                        writer.add_scalar('loss/attr_loss', loss['attr_loss'], step)
                        writer.add_scalar('loss_weighted/attr_loss', loss['weighted_attr_loss'], step)
                    if 'lm_vis_loss' in loss:
                        writer.add_scalar('loss/lm_vis_loss', loss['lm_vis_loss'], step)
                        writer.add_scalar('loss_weighted/lm_vis_loss', loss['weighted_lm_vis_loss'], step)
                    if 'lm_pos_loss' in loss:
                        writer.add_scalar('loss/lm_pos_loss', loss['lm_pos_loss'], step)
                        writer.add_scalar('loss_weighted/lm_pos_loss', loss['weighted_lm_pos_loss'], step)
                    writer.add_scalar('loss_weighted/all', loss['all'], step)
                    writer.add_scalar('global/learning_rate', learning_rate, step)

                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1,
                                                                             const.NUM_EPOCH,
                                                                             sample_idx + 1,
                                                                             total_step,
                                                                             loss['all'].item()) + '\033[1A\r')


                # log images
                if (sample_idx + 1) % const.LOG_INTERVAL_IMAGE == 0:
                    # get shapes
                    img_height = int(output['lm_pos_map'].shape[2])
                    img_width = int(output['lm_pos_map'].shape[3])
                    hm_size = img_height

                    # get predictions and groundtruth
                    heatmaps_gt = sample['landmark_map%d' % hm_size][0,:,:,:].cpu().numpy()
                    heatmaps_pred = output['lm_pos_map'][0,:,:,:].cpu().detach().numpy()
                    lm_pos_gt = sample['landmark_pos'][0,:,:].cpu().numpy()
                    lm_pos_pred = output['lm_pos_output'][0,:,:].cpu().detach().numpy()*hm_size

                    category_gt = sample['category_label'][0]
                    category_gt_name = const.CATEGORY_NAMES[int(category_gt)]
                    category_type = sample['category_type'][0].cpu().numpy()

                    # get category prediction if it is in network output and create text box
                    if 'category_loss' in loss:
                        _, category_pred = output['category_output'][0].topk(1, 0, True, True)
                        category_pred_name = const.CATEGORY_NAMES[int(category_pred)]

                        if (category_pred == category_gt):
                            text_pred = TextArea('pred: '+category_pred_name, textprops=dict(color='green'))
                        else:
                            text_pred = TextArea('pred: '+category_pred_name, textprops=dict(color='red'))
                    else:
                        text_pred = TextArea('pred: ----', textprops=dict(color='red'))
                    text_gt = TextArea('gt: '+category_gt_name, textprops=dict(color='blue'))
                    text_box = HPacker(children=[text_gt, text_pred], align='left', pad=5, sep=5)

                    # prepare input image
                    image = unnormalize_image(sample['image'][0,:,:,:].cpu())
                    image = image.numpy()
                    image = image.transpose((1,2,0)) # [C,H,W] => [H,W,C] for Matplotlib



                    # create figure and add input image & text_box
                    fig = plt.figure(10, figsize=(5,5))
                    ax = fig.add_subplot(111)
                    ax.imshow(image)
                    anchored_box = AnchoredOffsetbox(loc=2,
                                                     child=text_box, pad=0.,
                                                     frameon=True,
                                                     bbox_to_anchor=(0.,1.1),
                                                     bbox_transform=ax.transAxes,
                                                     borderpad=0.)
                    ax.add_artist(anchored_box)
                    fig.subplots_adjust(top=0.8)

                    # ad figure to logfile
                    writer.add_figure('input_image', fig, step)

                    # add log for each landmark

                    # create color map
                    new_cmap = transparent_cmap(plt.cm.Reds)
                    # create mesh grid
                    mesh_y, mesh_x = np.mgrid[0:img_height, 0:img_width]
                    for i, visible in enumerate(sample['landmark_vis'][0]):
                        fig = plt.figure(i, figsize=(5,5))
                        fig.suptitle(const.lm2name[i])

                        # only plot heatmaps when landmark is in current cloth type
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
                            # create figure and plot gt & pred heatmaps
                            ax1 = fig.add_subplot(121)
                            ax1.set_title('groundtruth')
                            ax1.imshow(image)
                            cb = ax1.contourf(mesh_x, mesh_y,
                                              heatmaps_gt[i,:,:].reshape(mesh_x.shape[0], mesh_y.shape[1]),
                                              15,
                                              cmap=new_cmap)
                            ax2 = fig.add_subplot(122)
                            ax2.set_title('prediction')
                            ax2.imshow(image)
                            cb = ax2.contourf(mesh_x, mesh_y,
                                              heatmaps_pred[i,:,:].reshape(mesh_x.shape[0], mesh_y.shape[1]),
                                              15,
                                              cmap=new_cmap)

                            # add gt & pred landmarks into figure
                            if visible == 1:
                                ax1.scatter(lm_pos_pred[i,0], lm_pos_pred[i,1], s=40, marker='x', c='g')
                                ax2.scatter(lm_pos_pred[i,0], lm_pos_pred[i,1], s=40, marker='x', c='g')
                                ax1.scatter(lm_pos_gt[i,0], lm_pos_gt[i,1], s=40, marker='.', c='y')
                                ax2.scatter(lm_pos_gt[i,0], lm_pos_gt[i,1], s=40, marker='.', c='y')
                            else:
                                ax1.scatter(lm_pos_pred[i,0], lm_pos_pred[i,1], s=40, marker='x', c='r')
                                ax2.scatter(lm_pos_pred[i,0], lm_pos_pred[i,1], s=40, marker='x', c='r')
                                ax1.scatter(lm_pos_gt[i,0], lm_pos_gt[i,1], s=40, marker='.', c='b')
                                ax2.scatter(lm_pos_gt[i,0], lm_pos_gt[i,1], s=40, marker='.', c='b')

                        # add figure to logfile
                        writer.add_figure('heatmaps/{}'.format(const.lm2name[i]), fig, step)


                # save and evaluate model
                if (sample_idx + 1) == total_step:
#                if (sample_idx + 1)%5 == 0:
                    print('\nSaving Model....')
                    net.set_buffer('step', step)
                    torch.save(net.state_dict(), 'models/' + const.MODEL_NAME)
                    print('OK.')
                    if const.VAL_WHILE_TRAIN:
                        print('Now Evaluate..')
                        with torch.no_grad():
                            net.eval()
                            evaluator = const.EVALUATOR(category_topk=(1,2,3), attr_topk=(1,2,3))
                            for j, sample in enumerate(val_dataloader):
                                # move samples to GPU/CPU
                                for key in sample:
                                    sample[key] = sample[key].to(const.device)
                                # perform inference
                                output = net(sample)
                                # add result to evaluator
                                evaluator.add(output, sample)

                                if (j + 1) % 100 == 0:
                                    print('Val Step [{}/{}]'.format(j + 1, val_step))

                            # get result from evaluator
                            ret = evaluator.evaluate()

                            # print results when it exists
                            for topk, accuracy in ret['category_accuracy_topk'].items():
                                print('metrics/category_top{}'.format(topk), accuracy)
                                writer.add_scalar('metrics/category_top{}'.format(topk), accuracy, step)

                            if 'attr_accuracy_topk' in ret:
                                for topk, accuracy in ret['attr_accuracy_topk'].items():
                                    print('metrics/attr_top{}'.format(topk), accuracy)
                                    writer.add_scalar('metrics/attr_top{}'.format(topk), accuracy, step)


                            for topk, accuracy in ret['attr_group_recall'].items():
                                for attr_type in range(1, 6):
                                    print('metrics/attr_top{}_type_{}_{}_recall'.format(
                                        topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1]
                                    )
                                    writer.add_scalar('metrics/attr_top{}_type_{}_{}_recall'.format(
                                        topk, attr_type, const.attrtype2name[attr_type]), accuracy[attr_type - 1], step
                                    )
                                print('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk])
                                writer.add_scalar('metrics/attr_top{}_all_recall'.format(topk), ret['attr_recall'][topk], step)

                            if ret['lm_dist'] != {}:
                                for i in range(8):
                                    print('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i])
                                    writer.add_scalar('metrics/dist_part_{}_{}'.format(i, const.lm2name[i]), ret['lm_individual_dist'][i], step)
                                print('metrics/dist_all', ret['lm_dist'])
                                writer.add_scalar('metrics/dist_all', ret['lm_dist'], step)

                        # track best epoch: LM network
                        if ret['lm_dist'] != {} and ret['category_accuracy_topk'] == {}:
                            if best_lm > ret['lm_dist']:
                                best_lm = ret['lm_dist']
                                best_epoch = epoch
                                torch.save(net.state_dict(), const.TRAIN_DIR+'/best_model')
                        # track best epoch: CTU network
                        elif ret['category_accuracy_topk'] != {} and ret['lm_dist'] == {}:
                             if best_cat == {} or (best_cat[1] < ret['category_accuracy_topk'][1]) or \
                              (best_cat[1] == ret['category_accuracy_topk'][1] and \
                               best_attr[1] <  ret['attr_accuracy_topk'][1]) or \
                              (best_cat[1] == ret['category_accuracy_topk'][1] and \
                               best_attr[1] == ret['attr_accuracy_topk'][1]  and epoch_loss < min_epoch_loss):

                                min_epoch_loss = epoch_loss
                                best_cat = ret['category_accuracy_topk']
                                best_attr = ret['attr_accuracy_topk']
                                best_epoch = epoch
                                torch.save(net.state_dict(), const.TRAIN_DIR+'/best_model')

                        # track best epoch: whole network
                        else:
                            if best_cat == {} or (best_cat[1] < ret['category_accuracy_topk'][1]) or \
                              (best_cat[1] == ret['category_accuracy_topk'][1] and best_lm > ret['lm_dist']):

                                best_lm = ret['lm_dist']
                                best_cat = ret['category_accuracy_topk']
                                best_epoch = epoch
                                torch.save(net.state_dict(), const.TRAIN_DIR+'/best_model')
    #                        elif best_lm > ret['lm_dist']:
    #                            best_lm = ret['lm_dist']


                        print('best epoch: {}, best category acc: {}, best attr acc: {}, best lm dist: {}'.format(best_epoch+1, best_cat[1], best_attr[1],  best_lm))
                        net.train()

            # early stopping
            if epoch > best_epoch + const.EARLYSTOPPING_THRESHOLD:
                print('Early stopping after epoch {}'.format(epoch+1))
                break;

            # learning rate decay
            if (epoch + 1) % const.LEARNING_RATE_STEP == 0:
                learning_rate *= const.LEARNING_RATE_DECAY
                optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()),
                                             lr=learning_rate)

        # log results over 5 different cross validation cycles
        writer.add_scalar('val/best_epoch', best_epoch, validation_cycle)
        writer.add_scalar('best_category_top1', best_cat[1], validation_cycle)
        writer.add_scalar('best_category_top2', best_cat[2], validation_cycle)
        writer.add_scalar('best_category_top3', best_cat[3], validation_cycle)
        writer.add_scalar('best_attr_top1', best_attr[1], validation_cycle)
        writer.add_scalar('best_attr_top2', best_attr[2], validation_cycle)
        writer.add_scalar('best_attr_top3', best_attr[3], validation_cycle)


        total_results['best_epoch_nr'][validation_cycle] = best_epoch
        total_results['best_category_top1'][validation_cycle] = best_cat[1]
        total_results['best_category_top2'][validation_cycle] = best_cat[2]
        total_results['best_category_top3'][validation_cycle] = best_cat[3]
        total_results['best_attr_top1'][validation_cycle] = best_attr[1]
        total_results['best_attr_top2'][validation_cycle] = best_attr[2]
        total_results['best_attr_top3'][validation_cycle] = best_attr[3]


    for validation_cycle in range(5):
        # print summary
        print('----------------------------------------------')
        print('validation cycle {}:'.format(validation_cycle))
        print('Best category accuracy:')
        print('top 1: {}, top 2: {} top 3: {}'.format(
            total_results['best_category_top1'][validation_cacle],
            total_results['best_category_top2'][validation_cacle],
            total_results['best_category_top3'][validation_cacle]))
        print('Best attribute accuracy:')
        print('top 1: {}, top 2: {} top 3: {}'.format(
            total_results['best_attr_top1'][validation_cacle],
            total_results['best_attr_top2'][validation_cacle],
            total_results['best_attr_top3'][validation_cacle]))


