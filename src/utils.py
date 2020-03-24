import os
import torch
import pandas as pd
import numpy as np
from src import const
from torchvision import transforms
import importlib
import argparse
import json
from joblib import Parallel, delayed


class Evaluator(object):

    def __init__(self, category_topk=(1, 3, 5), attr_topk=(3, 5)):
        self.category_topk = category_topk
        self.attr_topk = attr_topk
        self.reset()
        with open(const.base_path + 'Anno/list_attr_cloth.txt') as f:
            ret = []
            f.readline()
            f.readline()
            for line in f:
                line = line.split(' ')
                while line[-1].strip().isdigit() is False:
                    line = line[:-1]
                ret.append([
                    ' '.join(line[0:-1]).strip(),
                    int(line[-1])
                ])
        attr_type = pd.DataFrame(ret, columns=['attr_name', 'type'])
        attr_type['attr_index'] = ['attr_' + str(i) for i in range(1000)]
        attr_type.set_index('attr_index', inplace=True)
        self.attr_type = attr_type

    def reset(self):
        self.category_accuracy = []
        self.attr_group_gt = np.zeros((5, len(self.attr_topk)))
        self.attr_group_tp = np.zeros((5, len(self.attr_topk)))
        self.attr_all_gt = np.zeros((len(self.attr_topk),))
        self.attr_all_tp = np.zeros((len(self.attr_topk),))
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in self.category_topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100 / batch_size))
            for i in range(len(res)):
                res[i] = res[i].cpu().numpy()[0] / 100

            self.category_accuracy.append(res)

    def attr_count(self, output, sample):
        target = sample['attr'].cpu().numpy()
        target = np.split(target, target.shape[0])
        target = [x[0, :] for x in target]

        pred = output['attr_output'].cpu().detach().numpy()
        pred = np.split(pred, pred.shape[0])
        pred = [x[0, 1, :] for x in pred]

        # process each batch sample individuell in 8 parallel jobs
        attr_group_gt, attr_group_tp, attr_all_gt, attr_all_tp,= zip(*Parallel(n_jobs=8)
                                (delayed(self._parallel_attr)(target[batch_idx],
                                                              pred[batch_idx])
                                 for batch_idx in range(len(target))))

        # combine results from parallel processing
        attr_group_gt = sum(attr_group_gt)
        attr_group_tp = sum(attr_group_tp)
        attr_all_gt = sum(attr_all_gt)
        attr_all_tp = sum(attr_all_tp)

        # add current batch to total
        self.attr_group_gt += attr_group_gt
        self.attr_group_tp += attr_group_tp
        self.attr_all_gt += attr_all_gt
        self.attr_all_tp += attr_all_tp


    def _parallel_attr(self, target, pred):
        result_df = pd.DataFrame([target, pred],
                                 index=['target', 'pred'], columns=['attr_' + str(i) for i in range(1000)])
        result_df = result_df.transpose()
        result_df = result_df.join(self.attr_type[['type']])
        ret = []
        for i in range(1, 6):
            ret.append([np.clip(result_df[result_df['type'] == i]['target'].sum(), 0, k) for k in
            self.attr_topk])
        attr_group_gt = np.array(ret)

        ret = []
        result_df = result_df.sort_values('pred', ascending=False)
        for i in range(1, 6):
            sort_df = result_df[result_df['type'] == i]
            ret.append([
                sort_df.head(k)['target'].sum() for k in self.attr_topk
            ])
        attr_group_tp = np.array(ret)

        attr_all_tp = np.array([
            result_df.head(k)['target'].sum() for k in self.attr_topk
        ])

        ret = [np.clip(attr_group_gt.sum(axis=0)[i], 0, k) for i, k in
                                                         enumerate(self.attr_topk)]
        attr_all_gt = np.array(ret)

        return attr_group_gt, attr_group_tp, attr_all_gt, attr_all_tp


    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        if hasattr(const, 'SWITCH_LEFT_RIGHT') and const.SWITCH_LEFT_RIGHT == True:
            batch_size = output['lm_pos_output'].shape[0]
            landmark_dist = 0
            for k in range(batch_size):
                lm_pos_pred = landmark_vis_float[k] * output['lm_pos_output'][k,:,:].cpu().numpy()
                lm_pos_gt = landmark_vis_float[k] * sample['landmark_pos_normalized'][k,:,:].cpu().numpy()
                lm_pos_gt_switched = np.zeros_like(lm_pos_gt)
                # switch even and odd rows
                lm_pos_gt_switched[0::2] = lm_pos_gt[1::2]
                lm_pos_gt_switched[1::2] = lm_pos_gt[0::2]

                dist_normal = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt), axis=1))
                dist_switched = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt_switched), axis=1))

                if np.sum(dist_normal) < np.sum(dist_switched):
                    landmark_dist += dist_normal
                else:
                    landmark_dist += dist_switched

        else:
            landmark_dist = np.sum(np.sqrt(np.sum(np.square(
                landmark_vis_float * output['lm_pos_output'].cpu().numpy() - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
            ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        self.category_topk_accuracy(output['category_output'], sample['category_label'])
        self.attr_count(output, sample)
        self.landmark_count(output, sample)

    def evaluate(self):
        category_accuracy = np.array(self.category_accuracy).mean(axis=0)
        category_accuracy_topk = {}
        for i, top_n in enumerate(self.category_topk):
            category_accuracy_topk[top_n] = category_accuracy[i]

        attr_group_recall = {}
        attr_recall = {}
        for i, top_n in enumerate(self.attr_topk):
            attr_group_recall[top_n] = self.attr_group_tp[..., i] / self.attr_group_gt[..., i]
            attr_recall[top_n] = self.attr_all_tp[i] / self.attr_all_gt[i]

        lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {
            'category_accuracy_topk': category_accuracy_topk,
            'attr_group_recall': attr_group_recall,
            'attr_recall': attr_recall,
            'lm_individual_dist': lm_individual_dist,
            'lm_dist': lm_dist,
        }


class CTUInferenceEvaluator(object):
    def __init__(self, category_topk=(1, 2, 3, 5, 10)):
        self.category_topk = category_topk
        self.valid_categories = [9,10,15,16,17,25]
        self.reset()

        self.dic_category_label =  {0: 'Anorak',
                                    1: 'Blazer',
                                    2: 'Blouse',
                                    3: 'Bomber',
                                    4: 'Button-Down',
                                    5: 'Cardigan',
                                    6: 'Flannel',
                                    7: 'Halter',
                                    8: 'Henley',
                                    9: 'Hoodie',
                                    10: 'Jacket',
                                    11: 'Jersey',
                                    12: 'Parka',
                                    13: 'Peacoat',
                                    14: 'Poncho',
                                    15: 'Sweater',
                                    16: 'Tank',
                                    17: 'Tee',
                                    18: 'Top',
                                    19: 'Turtleneck',
                                    20: 'Capris',
                                    21: 'Chinos',
                                    22: 'Culottes',
                                    23: 'Cutoffs',
                                    24: 'Gauchos',
                                    25: 'Jeans',
                                    26: 'Jeggings',
                                    27: 'Jodhpurs',
                                    28: 'Joggers',
                                    29: 'Leggings',
                                    30: 'Sarong',
                                    31: 'Shorts',
                                    32: 'Skirt',
                                    33: 'Sweatpants',
                                    34: 'Sweatshorts',
                                    35: 'Trunks',
                                    36: 'Caftan',
                                    37: 'Cape',
                                    38: 'Coat',
                                    39: 'Coverup',
                                    40: 'Dress',
                                    41: 'Jumpsuit',
                                    42: 'Kaftan',
                                    43: 'Kimono',
                                    44: 'Nightdress',
                                    45: 'Onesie',
                                    46: 'Robe',
                                    47: 'Romper',
                                    48: 'Shirtdress',
                                    49: 'Sundress'}

    def reset(self):
        self.category_accuracy_sum = np.zeros((len(self.valid_categories), len(self.category_topk)), dtype=np.float32)
        self.category_samples_sum = np.zeros((len(self.valid_categories), len(self.category_topk)), dtype=np.float32)
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            # mask out categories that are not present (set prob to -Inf)
            mask = -float('Inf')* torch.ones_like(output)
            for i in self.valid_categories:
                mask[:,i] = 0
            output = output + mask

            val, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            pred = pred.float()
            target = target.float()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            # get correct and prediction category name
            all_lines = []
            for j in range(pred.shape[1]):
                line = [self.dic_category_label[int(target[j].cpu())]]
                line.extend(['-' for i in range(self.category_topk[-1])])
                line.extend([0])

                if correct[0,j]==1:
                    line[-1] = 1

                for k in range(pred.shape[0]):
                    line[k+1] = self.dic_category_label[int(pred[k,j].cpu())]
                all_lines.append(line)


            # get accuracy per category
            for i_k, k in enumerate(self.category_topk):
                for j, cat in enumerate(self.valid_categories):
                    # mask all samples of i-th category
                    cloth_mask = target.eq(cat)
                    # total of samples of i-th category
                    self.category_samples_sum[j, i_k] += cloth_mask.sum(0).float()
                    # correctly predicted samples of i-thcategory
                    self.category_accuracy_sum[j, i_k] += (cloth_mask.float()*correct[:k].sum(0).float()).sum()

            return all_lines

    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        if hasattr(const, 'SWITCH_LEFT_RIGHT') and const.SWITCH_LEFT_RIGHT == True:
            batch_size = output['lm_pos_output'].shape[0]
            landmark_dist = 0
            for k in range(batch_size):
                lm_pos_pred = landmark_vis_float[k] * output['lm_pos_output'][k,:,:].cpu().numpy()
                lm_pos_gt = landmark_vis_float[k] * sample['landmark_pos_normalized'][k,:,:].cpu().numpy()
                lm_pos_gt_switched = np.zeros_like(lm_pos_gt)
                # switch even and odd rows
                lm_pos_gt_switched[0::2] = lm_pos_gt[1::2]
                lm_pos_gt_switched[1::2] = lm_pos_gt[0::2]

                dist_normal = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt), axis=1))
                dist_switched = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt_switched), axis=1))

                if np.sum(dist_normal) < np.sum(dist_switched):
                    landmark_dist += dist_normal
                else:
                    landmark_dist += dist_switched

        else:
            landmark_dist = np.sum(np.sqrt(np.sum(np.square(
                landmark_vis_float * output['lm_pos_output'].cpu().numpy() - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
            ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist




    def add(self, output, sample):
        all_lines = self.category_topk_accuracy(output['category_output'], sample['category_label'])
        self.landmark_count(output, sample)
        return all_lines

    def evaluate(self):

        category_accuracy_group_topk = {}
        category_accuracy_topk = {}
        for i, top_k in enumerate(self.category_topk):
            category_accuracy_group_topk[top_k] = self.category_accuracy_sum[:,i] / self.category_samples_sum[:,i]
            category_accuracy_topk[top_k] = np.nansum(self.category_accuracy_sum[:,i])/np.nansum(self.category_samples_sum[:,i])


        for dist, count in zip(self.lm_dist_all,self.lm_vis_count_all):
            print(dist,count)

        lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {
            'category_accuracy_topk': category_accuracy_topk,
            'category_accuracy_group_topk': category_accuracy_group_topk,
            'lm_individual_dist': lm_individual_dist,
            'lm_dist': lm_dist,
        }



class RobotInferenceEvaluator(object):
    def __init__(self, category_topk=(1, 2, 3, 5, 10)):
        self.category_topk = category_topk
        self.valid_categories = [9,10,15,16,17,25]
        self.reset()

        self.dic_category_label =  {0: 'Anorak',
                                    1: 'Blazer',
                                    2: 'Blouse',
                                    3: 'Bomber',
                                    4: 'Button-Down',
                                    5: 'Cardigan',
                                    6: 'Flannel',
                                    7: 'Halter',
                                    8: 'Henley',
                                    9: 'Hoodie',
                                    10: 'Jacket',
                                    11: 'Jersey',
                                    12: 'Parka',
                                    13: 'Peacoat',
                                    14: 'Poncho',
                                    15: 'Sweater',
                                    16: 'Tank',
                                    17: 'Tee',
                                    18: 'Top',
                                    19: 'Turtleneck',
                                    20: 'Capris',
                                    21: 'Chinos',
                                    22: 'Culottes',
                                    23: 'Cutoffs',
                                    24: 'Gauchos',
                                    25: 'Jeans',
                                    26: 'Jeggings',
                                    27: 'Jodhpurs',
                                    28: 'Joggers',
                                    29: 'Leggings',
                                    30: 'Sarong',
                                    31: 'Shorts',
                                    32: 'Skirt',
                                    33: 'Sweatpants',
                                    34: 'Sweatshorts',
                                    35: 'Trunks',
                                    36: 'Caftan',
                                    37: 'Cape',
                                    38: 'Coat',
                                    39: 'Coverup',
                                    40: 'Dress',
                                    41: 'Jumpsuit',
                                    42: 'Kaftan',
                                    43: 'Kimono',
                                    44: 'Nightdress',
                                    45: 'Onesie',
                                    46: 'Robe',
                                    47: 'Romper',
                                    48: 'Shirtdress',
                                    49: 'Sundress'}

    def reset(self):
        self.category_accuracy_sum = np.zeros((len(self.valid_categories), len(self.category_topk)), dtype=np.float32)
        self.category_samples_sum = np.zeros((len(self.valid_categories), len(self.category_topk)), dtype=np.float32)
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            # mask out categories that are not present (set prob to -Inf)
            mask = -float('Inf')* torch.ones_like(output)
            for i in self.valid_categories:
                mask[:,i] = 0
            output = output + mask

            val, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            pred = pred.float()
            target = target.float()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            # get correct and prediction category name
            all_lines = []
            for j in range(pred.shape[1]):
                line = [self.dic_category_label[int(target[j].cpu())]]
                line.extend(['-' for i in range(self.category_topk[-1])])
                line.extend([0])

                if correct[0,j]==1:
                    line[-1] = 1

                for k in range(pred.shape[0]):
                    line[k+1] = self.dic_category_label[int(pred[k,j].cpu())]
                all_lines.append(line)


            # get accuracy per category
            for i_k, k in enumerate(self.category_topk):
                for j, cat in enumerate(self.valid_categories):
                    # mask all samples of i-th category
                    cloth_mask = target.eq(cat)
                    # total of samples of i-th category
                    self.category_samples_sum[j, i_k] += cloth_mask.sum(0).float()
                    # correctly predicted samples of i-thcategory
                    self.category_accuracy_sum[j, i_k] += (cloth_mask.float()*correct[:k].sum(0).float()).sum()

            return all_lines

    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        if hasattr(const, 'SWITCH_LEFT_RIGHT') and const.SWITCH_LEFT_RIGHT == True:
            batch_size = output['lm_pos_output'].shape[0]
            landmark_dist = 0
            for k in range(batch_size):
                lm_pos_pred = landmark_vis_float[k] * output['lm_pos_output'][k,:,:].cpu().numpy()
                lm_pos_gt = landmark_vis_float[k] * sample['landmark_pos_normalized'][k,:,:].cpu().numpy()
                lm_pos_gt_switched = np.zeros_like(lm_pos_gt)
                # switch even and odd rows
                lm_pos_gt_switched[0::2] = lm_pos_gt[1::2]
                lm_pos_gt_switched[1::2] = lm_pos_gt[0::2]

                dist_normal = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt), axis=1))
                dist_switched = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt_switched), axis=1))

                if np.sum(dist_normal) < np.sum(dist_switched):
                    landmark_dist += dist_normal
                else:
                    landmark_dist += dist_switched

        else:
            landmark_dist = np.sum(np.sqrt(np.sum(np.square(
                landmark_vis_float * output['lm_pos_output'].cpu().numpy() - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
            ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist




    def add(self, output, sample):
        all_lines = self.category_topk_accuracy(output['category_output'], sample['category_label'])
        self.landmark_count(output, sample)
        return all_lines

    def evaluate(self):

        category_accuracy_group_topk = {}
        category_accuracy_topk = {}
        for i, top_k in enumerate(self.category_topk):
            category_accuracy_group_topk[top_k] = self.category_accuracy_sum[:,i] / self.category_samples_sum[:,i]
            category_accuracy_topk[top_k] = np.nansum(self.category_accuracy_sum[:,i])/np.nansum(self.category_samples_sum[:,i])


        lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {
            'category_accuracy_topk': category_accuracy_topk,
            'category_accuracy_group_topk': category_accuracy_group_topk,
            'lm_individual_dist': lm_individual_dist,
            'lm_dist': lm_dist,
        }


class InferenceEvaluator(object):

    def __init__(self, category_topk=(1, 2, 3, 5, 10)):
        self.category_topk = category_topk
        self.reset()
        self.category_names = ['Anorak',
                               'Blazer',
                               'Blouse',
                               'Bomber',
                               'Button-Down',
                               'Cardigan',
                               'Flannel',
                               'Halter',
                               'Henley',
                               'Hoodie',
                               'Jacket',
                               'Jersey',
                               'Parka',
                               'Peacoat',
                               'Poncho',
                               'Sweater',
                               'Tank',
                               'Tee',
                               'Top',
                               'Turtleneck',
                               'Capris',
                               'Chinos',
                               'Culottes',
                               'Cutoffs',
                               'Gauchos',
                               'Jeans',
                               'Jeggings',
                               'Jodhpurs',
                               'Joggers',
                               'Leggings',
                               'Sarong',
                               'Shorts',
                               'Skirt',
                               'Sweatpants',
                               'Sweatshorts',
                               'Trunks',
                               'Caftan',
                               'Cape',
                               'Coat',
                               'Coverup',
                               'Dress',
                               'Jumpsuit',
                               'Kaftan',
                               'Kimono',
                               'Nightdress',
                               'Onesie',
                               'Robe',
                               'Romper',
                               'Shirtdress',
                               'Sundress']

        self.dic_category_label = {100: 'bluse',
                                   101: 'hoody',
                                   102: 'pants',
                                   103: 'polo',
                                   104: 'polo-long',
                                   105: 'skirt',
                                   106: 'tshirt',
                                   107: 'tshirt-long'}

    def reset(self):
        self.category_accuracy = []
        self.category_prediction = []


    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            valid_categories = [2,4,5,8,9,15,17,25,26,28,29,32]
            mask = torch.zeros_like(output)
            for i in valid_categories:
                mask[:,i] = 1

#            print(mask[1,:])
#            print('===========')
#            print(output)
#            print('===========')
#            print((output*mask))
#            print('===========')

            output = output+100.0
            val, pred = (output*mask).topk(maxk, 1, True, True)
#            print(val)
##            print(pred)
#            print('===========')

            pred = pred.t()
            correct = torch.zeros_like(pred)

            all_lines = []
            for j in range(pred.shape[1]):
                print(j)
                print(int(target[j].cpu()))
                line = [self.dic_category_label[int(target[j].cpu())]]
                line.extend(['-' for i in range(self.category_topk[-1])])
                line.extend([0])

                for k in range(pred.shape[0]):
                    line[k+1] = self.category_names[int(pred[k,j].cpu())]
                    if self.equal(pred[k,j], target[j]):
                        correct[k,j] = 1
                        if k==0:
                            line[-1] = 1
                        break
#                    print(pred[:,j], target[j])

                all_lines.append(line)
                self.category_prediction.append(line)

            res = []
            for k in self.category_topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100 / batch_size))
            for i in range(len(res)):
                res[i] = res[i].cpu().numpy()[0] / 100

            self.category_accuracy.append(res)
            return all_lines

    def equal(self, prediction, groundtruth):
        equal=False
        if groundtruth==100:
            equal = prediction in [2]
        elif groundtruth==101:
            equal = prediction in [9,15]
        elif groundtruth==102:
            equal = prediction in [25,26,28,29]
        elif groundtruth==103:
            equal = prediction in [2,4,5,8,15,17]#[2,4,17]
        elif groundtruth==104:
            equal = prediction in [2,4,5,8,15,17]#[4,5,8]
        elif groundtruth==105:
            equal = prediction in [32]
        elif groundtruth==106:
            equal = prediction in [2,4,5,8,15,17]#[2,17]
        elif groundtruth==107:
            equal = prediction in [2,4,5,8,15,17]#[2,5,15,17]

        return equal




    def add(self, output, sample):
        all_lines = self.category_topk_accuracy(output['category_output'], sample['category_label'])
        return all_lines

    def evaluate(self):
        category_accuracy = np.array(self.category_accuracy).mean(axis=0)
        category_accuracy_topk = {}
        for i, top_n in enumerate(self.category_topk):
            category_accuracy_topk[top_n] = category_accuracy[i]

        return {
            'category_accuracy_topk': category_accuracy_topk,
        }


class CTUEvaluator(object):

    def __init__(self, category_topk=(1, 3, 5)):
        self.category_topk = category_topk
        self.reset()

    def reset(self):
        self.category_accuracy = []
        self.category_accuracy_sum = np.zeros((4, len(self.category_topk)), dtype=np.float32)
        self.category_samples_sum = np.zeros((4, len(self.category_topk)), dtype=np.float32)
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

#            res = []
#            for k in self.category_topk:
#                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#                res.append(correct_k.mul_(100 / batch_size))
#            for i in range(len(res)):
#                res[i] = res[i].cpu().numpy()[0] / 100
#
#            self.category_accuracy.append(res)
#
            for i_k, k in enumerate(self.category_topk):
                for j in range(4):
                    # mask all samples of i-th category
                    cloth_mask = target.eq(j)
                    # total of samples of i-th category
                    self.category_samples_sum[j, i_k] += cloth_mask.sum(0).float()
                    # correctly predicted samples of i-thcategory
                    self.category_accuracy_sum[j, i_k] += (cloth_mask.float()*correct[:k].sum(0).float()).sum()


    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        if hasattr(const, 'SWITCH_LEFT_RIGHT') and const.SWITCH_LEFT_RIGHT == True:
            batch_size = output['lm_pos_output'].shape[0]
            landmark_dist = 0
            for k in range(batch_size):
                lm_pos_pred = landmark_vis_float[k] * output['lm_pos_output'][k,:,:].cpu().numpy()
                lm_pos_gt = landmark_vis_float[k] * sample['landmark_pos_normalized'][k,:,:].cpu().numpy()
                lm_pos_gt_switched = np.zeros_like(lm_pos_gt)
                # switch even and odd rows
                lm_pos_gt_switched[0::2] = lm_pos_gt[1::2]
                lm_pos_gt_switched[1::2] = lm_pos_gt[0::2]

                dist_normal = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt), axis=1))
                dist_switched = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt_switched), axis=1))

                if np.sum(dist_normal) < np.sum(dist_switched):
                    landmark_dist += dist_normal
                else:
                    landmark_dist += dist_switched

        else:
            landmark_dist = np.sum(np.sqrt(np.sum(np.square(
                landmark_vis_float * output['lm_pos_output'].cpu().numpy() - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
            ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        self.category_topk_accuracy(output['category_output'], sample['category_label'])
        self.landmark_count(output, sample)

    def evaluate(self):
#        category_accuracy = np.array(self.category_accuracy).mean(axis=0)
#        category_accuracy_topk = {}
#        for i, top_n in enumerate(self.category_topk):
#            category_accuracy_topk[top_n] = category_accuracy[i]

        category_accuracy_group_topk = {}
        category_accuracy_topk = {}
        for i, top_k in enumerate(self.category_topk):
            category_accuracy_group_topk[top_k] = self.category_accuracy_sum[:,i] / self.category_samples_sum[:,i]
            category_accuracy_topk[top_k] = np.nansum(self.category_accuracy_sum[:,i])/np.nansum(self.category_samples_sum[:,i])


        lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()

        return {
            'category_accuracy_topk': category_accuracy_topk,
            'category_accuracy_group_topk': category_accuracy_group_topk,
            'attr_group_recall': {},
            'attr_recall': {},
            'lm_individual_dist': lm_individual_dist,
            'lm_dist': lm_dist,
        }

class CloPeMaEvaluator(object):

    def __init__(self, category_topk=(1, 2, 3), attr_topk=(1, 2, 3)):
        self.category_topk = category_topk
        self.attr_topk = attr_topk
        self.reset()

    def reset(self):
        self.category_accuracy_sum = np.zeros((4, len(self.category_topk)), dtype=np.float32)
        self.category_samples_sum = np.zeros((4, len(self.category_topk)), dtype=np.float32)
        self.attr_accuracy_sum = np.zeros((4, len(self.attr_topk)), dtype=np.float32)
        self.attr_samples_sum = np.zeros((4, len(self.attr_topk)), dtype=np.float32)
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def category_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            # get top k prediction results
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i_k, k in enumerate(self.category_topk):
                for j in range(4):
                    # mask all samples of i-th category
                    cloth_mask = target.eq(j)
                    # total of samples of i-th category
                    self.category_samples_sum[j, i_k] += cloth_mask.sum(0).float()
                    # correctly predicted samples of i-thcategory
                    self.category_accuracy_sum[j, i_k] += (cloth_mask.float()*correct[:k].sum(0).float()).sum()

    def attr_topk_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.attr_topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for i_k, k in enumerate(self.attr_topk):
                for j in range(4):
                    # mask all samples of i-th category
                    cloth_mask = target.eq(j)
                    # total of samples of i-th category
                    self.attr_samples_sum[j, i_k] += cloth_mask.sum(0).float()
                    # correctly predicted samples of i-thcategory
                    self.attr_accuracy_sum[j, i_k] += (cloth_mask.float()*correct[:k].sum(0).float()).sum()

    def add(self, output, sample):
        self.category_topk_accuracy(output['category_output'], sample['category_label'])
        self.attr_topk_accuracy(output['attr_output'], sample['attr'][:,0])
#        self.landmark_count(output, sample)

    def evaluate(self):
        category_accuracy_group_topk = {}
        category_accuracy_topk = {}
        for i, top_k in enumerate(self.category_topk):
            category_accuracy_group_topk[top_k] = self.category_accuracy_sum[:,i] / self.category_samples_sum[:,i]
            category_accuracy_topk[top_k] = np.nansum(self.category_accuracy_sum[:,i])/np.nansum(self.category_samples_sum[:,i])

        attr_accuracy_group_topk = {}
        attr_accuracy_topk = {}
        for i, top_k in enumerate(self.category_topk):
            attr_accuracy_group_topk[top_k] = self.attr_accuracy_sum[:,i]/ self.attr_samples_sum[:,i]
            attr_accuracy_topk[top_k] = np.nansum(self.attr_accuracy_sum[:,i])/np.nansum(self.attr_samples_sum[:,i])


        return {
            'category_accuracy_topk': category_accuracy_topk,
            'category_accuracy_group_topk': category_accuracy_group_topk,
            'attr_accuracy_topk': attr_accuracy_topk,
            'attr_accuracy_group_topk': attr_accuracy_group_topk,
            'attr_group_recall': {},
            'attr_recall': {},
            'lm_individual_dist': {},
            'lm_dist': {},
        }



class LandmarkEvaluator(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_dist_all = np.array([0.] * 8)

    def landmark_count(self, output, sample):
        if hasattr(const, 'LM_EVAL_USE') and const.LM_EVAL_USE == 'in_pic':
            mask_key = 'landmark_in_pic'
        else:
            mask_key = 'landmark_vis'
        landmark_vis_count = sample[mask_key].cpu().numpy().sum(axis=0)
        landmark_vis_float = torch.unsqueeze(sample[mask_key].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        if hasattr(const, 'SWITCH_LEFT_RIGHT') and const.SWITCH_LEFT_RIGHT == True:
            batch_size = output['lm_pos_output'].shape[0]
            landmark_dist = 0
            for k in range(batch_size):
                lm_pos_pred = landmark_vis_float[k] * output['lm_pos_output'][k,:,:].cpu().numpy()
                lm_pos_gt = landmark_vis_float[k] * sample['landmark_pos_normalized'][k,:,:].cpu().numpy()
                lm_pos_gt_switched = np.zeros_like(lm_pos_gt)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        if hasattr(const, 'SWITCH_LEFT_RIGHT') and const.SWITCH_LEFT_RIGHT == True:
            batch_size = output['lm_pos_output'].shape[0]
            landmark_dist = 0
            for k in range(batch_size):
                lm_pos_pred = landmark_vis_float[k] * output['lm_pos_output'][k,:,:].cpu().numpy()
                lm_pos_gt = landmark_vis_float[k] * sample['landmark_pos_normalized'][k,:,:].cpu().numpy()
                lm_pos_gt_switched = np.zeros_like(lm_pos_gt)
                # switch even and odd rows
                lm_pos_gt_switched[0::2] = lm_pos_gt[1::2]
                lm_pos_gt_switched[1::2] = lm_pos_gt[0::2]

                dist_normal = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt), axis=1))
                dist_switched = np.sqrt(np.sum(np.square(lm_pos_pred - lm_pos_gt_switched), axis=1))

                if np.sum(dist_normal) < np.sum(dist_switched):
                    landmark_dist += dist_normal
                else:
                    landmark_dist += dist_switched

        else:
            landmark_dist = np.sum(np.sqrt(np.sum(np.square(
                landmark_vis_float * output['lm_pos_output'].cpu().numpy() - landmark_vis_float * sample['landmark_pos_normalized'].cpu().numpy(),
            ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_dist_all += landmark_dist

    def add(self, output, sample):
        self.landmark_count(output, sample)

    def evaluate(self):
        lm_individual_dist = self.lm_dist_all / self.lm_vis_count_all
        lm_dist = (self.lm_dist_all / self.lm_vis_count_all).mean()
        return {
            'category_accuracy_topk': {},
            'attr_group_recall': {},
            'attr_recall': {},
            'lm_individual_dist': lm_individual_dist,
            'lm_dist': lm_dist,
        }


def merge_const(module_name):
    new_conf = importlib.import_module(module_name)
    for key, value in new_conf.__dict__.items():
        if not(key.startswith('_')):
            setattr(const, key, value)
            print('override', key, value)

def log_parameters():
    vars_to_store = []
    for key, value in const.__dict__.items():
        if not(key.startswith('_')):
            vars_to_store.append([str(key), str(value)])

    filename = const.TRAIN_DIR+'/parameters.json'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w') as fp:
        json.dump(vars_to_store, fp, indent=4)



def parse_args_and_merge_const():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='', type=str)
    args = parser.parse_args()
    if args.conf != '':
        merge_const(args.conf)
    log_parameters()

def unnormalize_image(image):
    unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    return unnormalize(image)
