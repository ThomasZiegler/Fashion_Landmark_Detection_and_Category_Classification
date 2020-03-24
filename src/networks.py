import torch
from torch import nn
from torch.nn import functional as F
from src import const
from src.base_networks import CustomUnetGenerator, ModuleWithAttr,  VGG16Extractor, SELayer, SELayerReLU

class ReLUWholeNetwork(ModuleWithAttr):

    def __init__(self, use_iorn=False):
        super(ReLUWholeNetwork, self).__init__()
        self.vgg16_extractor = VGG16Extractor(use_iorn=use_iorn)
        self.lm_branch = const.LM_BRANCH(const.LM_SELECT_VGG_CHANNEL)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        self.spatial_attention_net = CustomUnetGenerator(512, 512, num_downs=2, ngf=32, last_act='relu')
        self.channel_attention_net = SELayerReLU(channel=512, reduction=const.SE_REDUCTION)
        self.combined_attention_net = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0),
                                                    nn.Tanh())
        self.pooled_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        conv5_para_vgg16 = [
            self.vgg16_extractor.vgg[-7].state_dict(),
            self.vgg16_extractor.vgg[-5].state_dict(),
            self.vgg16_extractor.vgg[-3].state_dict(),
        ]
        self.conv5_1.load_state_dict(conv5_para_vgg16[0])
        self.conv5_2.load_state_dict(conv5_para_vgg16[1])
        self.conv5_3.load_state_dict(conv5_para_vgg16[2])
        self.pooled_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.category_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.category_fc2 = nn.Linear(1024, 48)
        self.attr_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.attr_fc2 = nn.Linear(1024, 1000 * 2)
        self.softmax = nn.Softmax(dim=1)

        self.category_loss_func = torch.nn.CrossEntropyLoss()
        self.attr_loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([const.WEIGHT_ATTR_NEG, const.WEIGHT_ATTR_POS]).to(const.device))

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg16_for_lm = vgg16_output[const.LM_SELECT_VGG]
        lm_pos_map, lm_pos_output = self.lm_branch(vgg16_for_lm)

        lm_merge_map, _ = lm_pos_map.max(dim=1, keepdim=True)
        lm_merge_map = self.downsample(lm_merge_map)

        conv_feature = vgg16_output['conv4_3']

#        spatial_attention = torch.cat([lm_merge_map, conv_feature], dim=1)
#        spatial_attention = self.spatial_attention_net(spatial_attention)
        spatial_attention = self.spatial_attention_net(conv_feature)
        channel_attention = self.channel_attention_net(conv_feature)
        attention_map = self.combined_attention_net((spatial_attention+lm_merge_map)*channel_attention)

        new_conv_feature = (1 + attention_map) * conv_feature

        new_conv_feature = self.pooled_4(new_conv_feature)
        new_conv_feature = F.relu(self.conv5_1(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_2(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_3(new_conv_feature))
        new_conv_feature = self.pooled_5(new_conv_feature)
        feature = new_conv_feature.reshape(batch_size, -1)

        category_output = self.category_fc1(feature)
        category_output = F.relu(category_output)
        category_output = self.category_fc2(category_output)  # [batch_size, 48]
        category_pred = self.softmax(category_output)

        attr_output = self.attr_fc1(feature)
        attr_output = F.relu(attr_output)
        attr_output = self.attr_fc2(attr_output)
        attr_output = attr_output.reshape(attr_output.size()[0], 2,
                                          int(attr_output.size()[1]/2))  # [batch_size, 2, 1000]
        output = {}
        output['category_output'] = category_output
        output['category_pred'] = category_pred
        output['attr_output'] = attr_output
        output['lm_pos_output'] = lm_pos_output
        output['lm_pos_map'] = lm_pos_map
        output['attention_map'] = attention_map
        output['image'] = sample['image']

        return output

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()

        lm_size = int(output['lm_pos_map'].shape[2])
        vis_sample = sample['landmark_vis'].reshape(batch_size * 8, -1)
        vis_mask = torch.cat([vis_sample] * lm_size * lm_size, dim=1).float()
        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
        lm_pos_loss = torch.pow(vis_mask * (map_output - map_sample), 2).mean()

        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])


        attr_loss = self.attr_loss_func(output['attr_output'], sample['attr'])

        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
            const.WEIGHT_LOSS_ATTR * attr_loss + \
            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'attr_loss': attr_loss.item(),
            'lm_pos_loss': lm_pos_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
            'weighted_attr_loss': const.WEIGHT_LOSS_ATTR * attr_loss.item(),
            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss



class OldWholeNetwork(ModuleWithAttr):
    def __init__(self, use_iorn=False):
        super(OldWholeNetwork, self).__init__()
        self.vgg16_extractor = VGG16Extractor(use_iorn=use_iorn)
        self.lm_branch = const.LM_BRANCH(const.LM_SELECT_VGG_CHANNEL)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        self.spatial_attention_net = CustomUnetGenerator(512, 512, num_downs=2, ngf=32, last_act='relu')
        self.pooled_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        conv5_para_vgg16 = [
            self.vgg16_extractor.vgg[-7].state_dict(),
            self.vgg16_extractor.vgg[-5].state_dict(),
            self.vgg16_extractor.vgg[-3].state_dict(),
        ]
        self.conv5_1.load_state_dict(conv5_para_vgg16[0])
        self.conv5_2.load_state_dict(conv5_para_vgg16[1])
        self.conv5_3.load_state_dict(conv5_para_vgg16[2])
        self.pooled_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.category_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.category_fc2 = nn.Linear(1024, 48)
        self.attr_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.attr_fc2 = nn.Linear(1024, 1000 * 2)
        self.softmax = nn.Softmax(dim=1)

        self.category_loss_func = torch.nn.CrossEntropyLoss()
        self.attr_loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([const.WEIGHT_ATTR_NEG, const.WEIGHT_ATTR_POS]).to(const.device))

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg16_for_lm = vgg16_output[const.LM_SELECT_VGG]
        lm_pos_map, lm_pos_output = self.lm_branch(vgg16_for_lm)

        lm_merge_map, _ = lm_pos_map.max(dim=1, keepdim=True)
        lm_merge_map = self.downsample(lm_merge_map)

        conv_feature = vgg16_output['conv4_3']

#        spatial_attention = torch.cat([lm_merge_map, conv_feature], dim=1)
#        attention_map = self.spatial_attention_net(spatial_attention)
        spatial_attention = self.spatial_attention_net(conv_feature)
        channel_attention = self.channel_attention_net(conv_feature)
        attention_map = self.combined_attention_net((spatial_attention+lm_merge_map)*channel_attention)

        new_conv_feature = (1 + attention_map) * conv_feature

        new_conv_feature = self.pooled_4(new_conv_feature)
        new_conv_feature = F.relu(self.conv5_1(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_2(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_3(new_conv_feature))
        new_conv_feature = self.pooled_5(new_conv_feature)
        feature = new_conv_feature.reshape(batch_size, -1)

        category_output = self.category_fc1(feature)
        category_output = F.relu(category_output)
        category_output = self.category_fc2(category_output)  # [batch_size, 48]
        category_pred = self.softmax(category_output)

        attr_output = self.attr_fc1(feature)
        attr_output = F.relu(attr_output)
        attr_output = self.attr_fc2(attr_output)
        attr_output = attr_output.reshape(attr_output.size()[0], 2,
                                          int(attr_output.size()[1]/2))  # [batch_size, 2, 1000]
        output = {}
        output['category_output'] = category_output
        output['category_pred'] = category_pred
        output['attr_output'] = attr_output
        output['lm_pos_output'] = lm_pos_output
        output['lm_pos_map'] = lm_pos_map
        output['attention_map'] = attention_map
        output['image'] = sample['image']

        return output

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()

        lm_size = int(output['lm_pos_map'].shape[2])
        vis_sample = sample['landmark_vis'].reshape(batch_size * 8, -1)
        vis_mask = torch.cat([vis_sample] * lm_size * lm_size, dim=1).float()
        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
        lm_pos_loss = torch.pow(vis_mask * (map_output - map_sample), 2).mean()

        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])


        attr_loss = self.attr_loss_func(output['attr_output'], sample['attr'])

        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
            const.WEIGHT_LOSS_ATTR * attr_loss + \
            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'attr_loss': attr_loss.item(),
            'lm_pos_loss': lm_pos_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
            'weighted_attr_loss': const.WEIGHT_LOSS_ATTR * attr_loss.item(),
            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss



class WholeNetwork(ModuleWithAttr):

    def __init__(self, use_iorn=False):
        super(WholeNetwork, self).__init__()
        self.vgg16_extractor = VGG16Extractor(use_iorn=use_iorn)
        self.lm_branch = const.LM_BRANCH(const.LM_SELECT_VGG_CHANNEL)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        self.spatial_attention_net = CustomUnetGenerator(512, 512, num_downs=2, ngf=32, last_act='relu')
        self.channel_attention_net = SELayer(channel=512, reduction=const.SE_REDUCTION)
        self.combined_attention_net = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0),
                                                    nn.Tanh())
        self.pooled_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        conv5_para_vgg16 = [
            self.vgg16_extractor.vgg[-7].state_dict(),
            self.vgg16_extractor.vgg[-5].state_dict(),
            self.vgg16_extractor.vgg[-3].state_dict(),
        ]
        self.conv5_1.load_state_dict(conv5_para_vgg16[0])
        self.conv5_2.load_state_dict(conv5_para_vgg16[1])
        self.conv5_3.load_state_dict(conv5_para_vgg16[2])
        self.pooled_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.category_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.category_fc2 = nn.Linear(1024, 48)
        self.attr_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.attr_fc2 = nn.Linear(1024, 1000 * 2)
        self.softmax = nn.Softmax(dim=1)

        self.category_loss_func = torch.nn.CrossEntropyLoss()
        self.attr_loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([const.WEIGHT_ATTR_NEG, const.WEIGHT_ATTR_POS]).to(const.device))

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg16_for_lm = vgg16_output[const.LM_SELECT_VGG]
        lm_pos_map, lm_pos_output = self.lm_branch(vgg16_for_lm)

        lm_merge_map, _ = lm_pos_map.max(dim=1, keepdim=True)
        lm_merge_map = self.downsample(lm_merge_map)

        conv_feature = vgg16_output['conv4_3']

#        spatial_attention = torch.cat([lm_merge_map, conv_feature], dim=1)
#        spatial_attention = self.spatial_attention_net(spatial_attention)
        spatial_attention = self.spatial_attention_net(conv_feature)
        channel_attention = self.channel_attention_net(conv_feature)
        attention_map = self.combined_attention_net((spatial_attention+lm_merge_map)*channel_attention)

        new_conv_feature = (1 + attention_map) * conv_feature

        new_conv_feature = self.pooled_4(new_conv_feature)
        new_conv_feature = F.relu(self.conv5_1(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_2(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_3(new_conv_feature))
        new_conv_feature = self.pooled_5(new_conv_feature)
        feature = new_conv_feature.reshape(batch_size, -1)

        category_output = self.category_fc1(feature)
        category_output = F.relu(category_output)
        category_output = self.category_fc2(category_output)  # [batch_size, 48]
        category_pred = self.softmax(category_output)

        attr_output = self.attr_fc1(feature)
        attr_output = F.relu(attr_output)
        attr_output = self.attr_fc2(attr_output)
        attr_output = attr_output.reshape(attr_output.size()[0], 2,
                                          int(attr_output.size()[1]/2))  # [batch_size, 2, 1000]
        output = {}
        output['category_output'] = category_output
        output['category_pred'] = category_pred
        output['attr_output'] = attr_output
        output['lm_pos_output'] = lm_pos_output
        output['lm_pos_map'] = lm_pos_map
        output['attention_map'] = attention_map
        output['image'] = sample['image']

        return output

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()

        lm_size = int(output['lm_pos_map'].shape[2])
        vis_sample = sample['landmark_vis'].reshape(batch_size * 8, -1)
        vis_mask = torch.cat([vis_sample] * lm_size * lm_size, dim=1).float()
        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
        lm_pos_loss = torch.pow(vis_mask * (map_output - map_sample), 2).mean()

        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])


        attr_loss = self.attr_loss_func(output['attr_output'], sample['attr'])

        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
            const.WEIGHT_LOSS_ATTR * attr_loss + \
            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'attr_loss': attr_loss.item(),
            'lm_pos_loss': lm_pos_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
            'weighted_attr_loss': const.WEIGHT_LOSS_ATTR * attr_loss.item(),
            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss


class CTUNetwork(ModuleWithAttr):

    def __init__(self, use_iorn=False):
        super(CTUNetwork, self).__init__()
        self.vgg16_extractor = VGG16Extractor(use_iorn=use_iorn)
        self.lm_branch = const.LM_BRANCH(const.LM_SELECT_VGG_CHANNEL)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        self.attention_pred_net = CustomUnetGenerator(512 + 1, 512, num_downs=2, ngf=32, last_act='tanh')
        self.pooled_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        conv5_para_vgg16 = [
            self.vgg16_extractor.vgg[-7].state_dict(),
            self.vgg16_extractor.vgg[-5].state_dict(),
            self.vgg16_extractor.vgg[-3].state_dict(),
        ]
        self.conv5_1.load_state_dict(conv5_para_vgg16[0])
        self.conv5_2.load_state_dict(conv5_para_vgg16[1])
        self.conv5_3.load_state_dict(conv5_para_vgg16[2])
        self.pooled_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ctu_category_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.ctu_category_fc2 = nn.Linear(1024, 8)

        self.category_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg16_for_lm = vgg16_output[const.LM_SELECT_VGG]
        lm_pos_map, lm_pos_output = self.lm_branch(vgg16_for_lm)

        lm_merge_map, _ = lm_pos_map.max(dim=1, keepdim=True)
        lm_merge_map = self.downsample(lm_merge_map)

        conv_feature = vgg16_output['conv4_3']

        attention_map = torch.cat([lm_merge_map, conv_feature], dim=1)
        attention_map = self.attention_pred_net(attention_map)

        new_conv_feature = (1 + attention_map) * conv_feature

        new_conv_feature = self.pooled_4(new_conv_feature)
        new_conv_feature = F.relu(self.conv5_1(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_2(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_3(new_conv_feature))
        new_conv_feature = self.pooled_5(new_conv_feature)
        feature = new_conv_feature.reshape(batch_size, -1)

        category_output = self.ctu_category_fc1(feature)
        category_output = F.relu(category_output)
        category_output = self.ctu_category_fc2(category_output)  # [batch_size, 8]

        output = {}
        output['category_output'] = category_output
        output['lm_pos_output'] = lm_pos_output
        output['lm_pos_map'] = lm_pos_map
        output['attention_map'] = attention_map
        output['image'] = sample['image']

        return output

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()


        lm_size = int(output['lm_pos_map'].shape[2])
        vis_sample = sample['landmark_vis'].reshape(batch_size * 8, -1)
        vis_mask = torch.cat([vis_sample] * lm_size * lm_size, dim=1).float()
        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
        lm_pos_loss = torch.pow(vis_mask * (map_output - map_sample), 2).mean()

        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])

        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
            'lm_pos_loss': lm_pos_loss.item(),
            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss

class CloPeMaNetworkNoLM(ModuleWithAttr):

    def __init__(self, use_iorn=False):
        super(CloPeMaNetworkNoLM, self).__init__()
        self.vgg16_extractor = VGG16Extractor(use_iorn=use_iorn)
        self.lm_branch = const.LM_BRANCH(const.LM_SELECT_VGG_CHANNEL)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        self.spatial_attention_net = CustomUnetGenerator(512, 512, num_downs=2, ngf=32, last_act='relu')
        self.channel_attention_net = SELayer(channel=512, reduction=const.SE_REDUCTION)
        self.combined_attention_net = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0),
                                                    nn.Tanh())

        self.pooled_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        conv5_para_vgg16 = [
            self.vgg16_extractor.vgg[-7].state_dict(),
            self.vgg16_extractor.vgg[-5].state_dict(),
            self.vgg16_extractor.vgg[-3].state_dict(),
        ]
        self.conv5_1.load_state_dict(conv5_para_vgg16[0])
        self.conv5_2.load_state_dict(conv5_para_vgg16[1])
        self.conv5_3.load_state_dict(conv5_para_vgg16[2])
        self.pooled_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ctu_dropout_1 = nn.Dropout(const.DROPOUT_RATE)
        self.ctu_category_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.ctu_dropout_2 = nn.Dropout(const.DROPOUT_RATE)
        self.ctu_category_fc2 = nn.Linear(1024, 8)

        self.attr_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.attr_fc2 = nn.Linear(1024, 4)

        self.category_loss_func = torch.nn.CrossEntropyLoss()
        self.attr_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg16_for_lm = vgg16_output[const.LM_SELECT_VGG]
        lm_pos_map, lm_pos_output = self.lm_branch(vgg16_for_lm)

        lm_merge_map, _ = lm_pos_map.max(dim=1, keepdim=True)
        lm_merge_map = self.downsample(lm_merge_map)

        conv_feature = vgg16_output['conv4_3']

#        attention_map = torch.cat([lm_merge_map, conv_feature], dim=1)
#        attention_map = self.attention_pred_net(attention_map)
#
#        new_conv_feature = (1 + attention_map) * conv_feature

        spatial_attention = self.spatial_attention_net(conv_feature)
        channel_attention = self.channel_attention_net(conv_feature)
        attention_map = self.combined_attention_net((spatial_attention)*channel_attention)

        new_conv_feature = (1 + attention_map) * conv_feature



        new_conv_feature = self.pooled_4(new_conv_feature)
        new_conv_feature = F.relu(self.conv5_1(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_2(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_3(new_conv_feature))
        new_conv_feature = self.pooled_5(new_conv_feature)
        feature = new_conv_feature.reshape(batch_size, -1)

        feature = self.ctu_dropout_1(feature)
        category_output = self.ctu_category_fc1(feature)
        category_output = self.ctu_dropout_2(category_output)
        category_output = F.relu(category_output)
        category_output = self.ctu_category_fc2(category_output)  # [batch_size, 8]

        attr_output = self.attr_fc1(feature)
        attr_output = F.relu(attr_output)
        attr_output = self.attr_fc2(attr_output)  # [batch_size, 4]


        output = {}
        output['category_output'] = category_output
        output['attr_output'] = attr_output
        output['lm_pos_output'] = lm_pos_output
        output['lm_pos_map'] = lm_pos_map
        output['attention_map'] = attention_map
        output['image'] = sample['image']

        return output

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()

#        lm_size = int(output['lm_pos_map'].shape[2])
#        vis_sample = sample['landmark_vis'].reshape(batch_size * 8, -1)
#        vis_mask = torch.cat([vis_sample] * lm_size * lm_size, dim=1).float()
#        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
#        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
#        lm_pos_loss = torch.pow(vis_mask * (map_output - map_sample), 2).mean()
#        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
#            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])

        attr_loss = self.attr_loss_func(output['attr_output'], sample['attr'][:,0])
        all_loss =  const.WEIGHT_LOSS_CATEGORY * category_loss + const.WEIGHT_LOSS_ATTR * attr_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'attr_loss': attr_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
            'weighted_attr_loss': const.WEIGHT_LOSS_ATTR * attr_loss.item(),
#            'lm_pos_loss': lm_pos_loss.item(),
#            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss



class CloPeMaNetwork(ModuleWithAttr):

    def __init__(self, use_iorn=False):
        super(CloPeMaNetwork, self).__init__()
        self.vgg16_extractor = VGG16Extractor(use_iorn=use_iorn)
        self.lm_branch = const.LM_BRANCH(const.LM_SELECT_VGG_CHANNEL)
        self.downsample = nn.Upsample((28, 28), mode='bilinear', align_corners=False)
        self.attention_pred_net = CustomUnetGenerator(512 + 1, 512, num_downs=2, ngf=32, last_act='tanh')
        self.pooled_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        conv5_para_vgg16 = [
            self.vgg16_extractor.vgg[-7].state_dict(),
            self.vgg16_extractor.vgg[-5].state_dict(),
            self.vgg16_extractor.vgg[-3].state_dict(),
        ]
        self.conv5_1.load_state_dict(conv5_para_vgg16[0])
        self.conv5_2.load_state_dict(conv5_para_vgg16[1])
        self.conv5_3.load_state_dict(conv5_para_vgg16[2])
        self.pooled_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ctu_category_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.ctu_category_fc2 = nn.Linear(1024, 8)

        self.attr_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.attr_fc2 = nn.Linear(1024, 4)

        self.category_loss_func = torch.nn.CrossEntropyLoss()
        self.attr_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg16_for_lm = vgg16_output[const.LM_SELECT_VGG]
        lm_pos_map, lm_pos_output = self.lm_branch(vgg16_for_lm)

        lm_merge_map, _ = lm_pos_map.max(dim=1, keepdim=True)
        lm_merge_map = self.downsample(lm_merge_map)

        conv_feature = vgg16_output['conv4_3']

        attention_map = torch.cat([lm_merge_map, conv_feature], dim=1)
        attention_map = self.attention_pred_net(attention_map)

        new_conv_feature = (1 + attention_map) * conv_feature

        new_conv_feature = self.pooled_4(new_conv_feature)
        new_conv_feature = F.relu(self.conv5_1(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_2(new_conv_feature))
        new_conv_feature = F.relu(self.conv5_3(new_conv_feature))
        new_conv_feature = self.pooled_5(new_conv_feature)
        feature = new_conv_feature.reshape(batch_size, -1)

        category_output = self.ctu_category_fc1(feature)
        category_output = F.relu(category_output)
        category_output = self.ctu_category_fc2(category_output)  # [batch_size, 8]

        attr_output = self.attr_fc1(feature)
        attr_output = F.relu(attr_output)
        attr_output = self.attr_fc2(attr_output)  # [batch_size, 4]


        output = {}
        output['category_output'] = category_output
        output['attr_output'] = attr_output
        output['lm_pos_output'] = lm_pos_output
        output['lm_pos_map'] = lm_pos_map
        output['attention_map'] = attention_map
        output['image'] = sample['image']

        return output

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()

#        lm_size = int(output['lm_pos_map'].shape[2])
#        vis_sample = sample['landmark_vis'].reshape(batch_size * 8, -1)
#        vis_mask = torch.cat([vis_sample] * lm_size * lm_size, dim=1).float()
#        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
#        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
#        lm_pos_loss = torch.pow(vis_mask * (map_output - map_sample), 2).mean()
#        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
#            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])

        attr_loss = self.attr_loss_func(output['attr_output'], sample['attr'][:,0])
        all_loss =  const.WEIGHT_LOSS_CATEGORY * category_loss + const.WEIGHT_LOSS_ATTR * attr_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'attr_loss': attr_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
            'weighted_attr_loss': const.WEIGHT_LOSS_ATTR * attr_loss.item(),
#            'lm_pos_loss': lm_pos_loss.item(),
#            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss

class CloPeMaVGGNetwork(ModuleWithAttr):

    def __init__(self, use_iorn=False):
        super(CloPeMaVGGNetwork, self).__init__()
        self.vgg16_extractor = VGG16Extractor(use_iorn=use_iorn)
        self.ctu_category_fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.ctu_category_fc2 = nn.Linear(1024, 8)

        self.category_loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg_conv_feature = vgg16_output['pooled_5']
        feature = vgg_conv_feature.reshape(batch_size, -1)

        category_output = self.ctu_category_fc1(feature)
        category_output = F.relu(category_output)
        category_output = self.ctu_category_fc2(category_output)  # [batch_size, 8]

        output = {}
        output['category_output'] = category_output
        output['image'] = sample['image']

        return output

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()

#        lm_size = int(output['lm_pos_map'].shape[2])
#        vis_sample = sample['landmark_vis'].reshape(batch_size * 8, -1)
#        vis_mask = torch.cat([vis_sample] * lm_size * lm_size, dim=1).float()
#        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
#        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
#        lm_pos_loss = torch.pow(vis_mask * (map_output - map_sample), 2).mean()
#        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
#            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])
        all_loss =  const.WEIGHT_LOSS_CATEGORY * category_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
#            'lm_pos_loss': lm_pos_loss.item(),
#            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss




if __name__ == '__main__':
    pass
