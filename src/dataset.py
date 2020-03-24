import torch
import torch.utils.data
import math
import numpy as np
import numpy.matlib
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
#from skimage import io, transform
from src import const

import cv2
import random

def read_image(path, dtype=np.float32):
    image = cv2.imread(path, 1)
    # BGR to RGB
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#    # transpose (H, W, C) -> (C, H, W)
#    image = image.transpose((2, 0, 1))

    return image

def gaussian_map(image_w, image_h, center_x, center_y, R):
    Gauss_map = np.zeros((image_h, image_w))

    mask_x = np.matlib.repmat(center_x, image_h, image_w)
    mask_y = np.matlib.repmat(center_y, image_h, image_w)
    x1 = np.arange(image_w)
    x_map = np.matlib.repmat(x1, image_h, 1)
    y1 = np.arange(image_h)
    y_map = np.matlib.repmat(y1, image_w, 1)
    y_map = np.transpose(y_map)
    Gauss_map = np.sqrt((x_map - mask_x)**2 + (y_map - mask_y)**2)
    Gauss_map = np.exp(-0.5 * Gauss_map / R)
    return Gauss_map


def gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R):
    ret = []
    # 修改为不在图片里的时候才为0，其他都给出
    for i in range(landmark_in_pic.shape[0]):
        if landmark_in_pic[i] == 0:
            ret.append(np.zeros((image_w, image_h)))
        else:
            channel_map = gaussian_map(image_w, image_h, landmark_pos[i][0], landmark_pos[i][1], R)
            ret.append(channel_map.reshape((image_w, image_h)))
    return np.stack(ret, axis=0).astype(np.float32)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

#        img = transform.resize(image, (new_h, new_w), mode='constant', anti_aliasing=True)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return img, landmarks


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks


class CenterCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks


class RandomFlip(object):

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            landmarks[:, 0] = w - landmarks[:, 0]

        return image, landmarks


class BBoxCrop(object):

    def __call__(self, image, landmarks, bbox):
        h, w = image.shape[:2]

        top = int(bbox[0,1])
        left = int(bbox[0,0])
        new_h = int(bbox[3,1])-top
        new_w = int(bbox[3,0])-left

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks


class CheckLandmarks(object):

    def __call__(self, image, landmark_vis, landmark_in_pic, landmark_pos):
        h, w = image.shape[:2]
        landmark_vis = landmark_vis.copy()
        landmark_in_pic = landmark_in_pic.copy()
        landmark_pos = landmark_pos.copy()
        for i, vis in enumerate(landmark_vis):
            if (landmark_pos[i, 0] <= 0) or (landmark_pos[i, 0] >= w) or (landmark_pos[i, 1] <= 0) or (landmark_pos[i, 1] >= h):
                landmark_vis[i] = 0
                landmark_in_pic[i] = 0
        for i, in_pic in enumerate(landmark_in_pic):
            if in_pic == 0:
                landmark_pos[i, :] = 0
        return landmark_vis, landmark_in_pic, landmark_pos


class LandmarksNormalize(object):

    def __call__(self, image, landmark_pos):
        h, w = image.shape[:2]
        landmark_pos = landmark_pos / [float(w), float(h)]
        return landmark_pos


class LandmarksUnNormalize(object):

    def __call__(self, image, landmark_pos):
        h, w = image.shape[:2]
        landmark_pos = landmark_pos * [float(w), float(h)]
        return landmark_pos

class Rotate(object):
    def __init__(self, random_state, max_degree, prob=1.0, border_mode='border', center_mode='bbox'):

        _pad_mode_convert = {'reflection':cv2.BORDER_REFLECT,
                             'zeros':cv2.BORDER_CONSTANT,
                             'border':cv2.BORDER_REPLICATE}
        self.random_state = random_state
        self.border_mode = _pad_mode_convert[border_mode]
        self.max_degree = max_degree
        self.prob = prob
        self.center_mode = center_mode
        self.rotation_matrix = None



    def __call__(self, image, landmarks, bbox):
        # perform rotation with probability self.prob
        p = self.random_state.uniform(0,1)
        if p <= self.prob:
            angle = self.random_state.uniform(-self.max_degree, self.max_degree)

            height, width = image.shape[:2]
            if self.center_mode is 'middle':
                center = (width//2, height//2) # (x,y)
            else:
                center = ((bbox[0,0]+bbox[3,0])//2, (bbox[0,1]+bbox[3,1])//2)


            self._create_rotation_matrix(center, angle)
            image = self._rotate_image(image, center)
            landmarks = self._rotate_points(landmarks)
            bbox = self._rotate_points(bbox)
            bbox = self._update_bbox(bbox, height, width)


        return image, landmarks, bbox

    def _update_bbox(self, bbox, height, width):
        p_min = np.amin(bbox, axis=0)
        p_max = np.amax(bbox, axis=0)

        p_min = p_min.clip(min=0)
        p_max[0].clip(max=width)
        p_max[1].clip(max=height)

        new_bbox =np.array([[p_min[0], p_min[1]],
                            [p_max[0], p_min[1]],
                            [p_min[0], p_max[1]],
                            [p_max[0], p_max[1]]])

        return new_bbox

    def _rotate_points(self, points):
        nr_points = points.shape[0]
        affine_points = np.ones((nr_points,3))
        affine_points[:,:2] = points

        rotated_points = np.matmul(self.rotation_matrix, affine_points.transpose()).transpose()
        # set points back to zero
        rotated_points[points==[0, 0]] = points[points==[0, 0]]

        return rotated_points

    def _create_rotation_matrix(self, center, angle, scale=1.0):
        self.rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=scale)

    def _rotate_image(self, image, center):
        assert self.rotation_matrix is not None
        height, width = image.shape[:2]

        rotation_matrix = self.rotation_matrix
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        # compute the new bounding dimensions of the image
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # adjust the rotation matrix to take into account translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        return cv2.warpAffine(image, rotation_matrix,
                              (new_width, new_height), borderMode=self.border_mode,
                              flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_AREA)


class ElasticTransformation(object):
    def __init__(self, random_state, mode, prob=1.0, sigma=4.0, alpha=100.0, border_mode='border', interpolation_mode='linear', nr_points=3):

        _pad_mode_convert = {'reflection': cv2.BORDER_REFLECT,
                             'zeros': cv2.BORDER_CONSTANT,
                             'border': cv2.BORDER_REPLICATE}

        _interpolation_mode_convert = {'linear': cv2.INTER_LINEAR,
                                       'cubic': cv2.INTER_CUBIC}


        self.random_state = random_state
        self.mode = mode
        self.border_mode = _pad_mode_convert[border_mode]
        self.interpolation_mode = _interpolation_mode_convert[interpolation_mode]
        self.prob = prob
        self.sigma = sigma
        self.alpha = alpha
        self.nr_points = nr_points

        self.img_height = None
        self.img_width = None
        self.kernel_size = 2*int(4*self.sigma)+1
        self.map_x = []
        self.map_y = []

        self.mode in ['random', 'points']

    def __call__(self, image, landmarks):
        # perform rotation with probability self.prob
        p = self.random_state.uniform(0,1)
        if p <= self.prob:
            self.img_height, self.img_width = image.shape[:2]

            if self.mode == 'random':
                # create delta_coordinate maps
                dx = (self.random_state.rand(self.img_height, self.img_width)*2-1)
                dy = (self.random_state.rand(self.img_height, self.img_width)*2-1)
            elif self.mode == 'points':
                dx = np.zeros((self.img_height, self.img_width))
                dy = np.zeros((self.img_height, self.img_width))

                for i in range(self.nr_points):
                    x_idx = int(self.random_state.rand(1)*self.img_width)
                    y_idx = int(self.random_state.rand(1)*self.img_height)
                    x_value = self.alpha*(self.random_state.rand(1)*2-1)
                    y_value = self.alpha*(self.random_state.rand(1)*2-1)

                    dx[y_idx, x_idx] = x_value
                    dy[y_idx, x_idx] = y_value
            else:
                raise NotImplementedError

            # Smooth delta_coordinates
            dx_smooth = self.alpha * cv2.GaussianBlur(dx,
                                                      (self.kernel_size,
                                                       self.kernel_size),
                                                      sigmaX=self.sigma,
                                                      sigmaY=self.sigma,
                                                      borderType=self.border_mode)
            dy_smooth = self.alpha * cv2.GaussianBlur(dy,
                                                      (self.kernel_size,
                                                       self.kernel_size),
                                                      sigmaX=self.sigma,
                                                      sigmaY=self.sigma,
                                                      borderType=self.border_mode)


            # create coordinate maps src_coordinate + delta_coordinate
            x, y = np.meshgrid(np.arange(self.img_width), np.arange(self.img_height))
            self.map_x = np.float32(x+dx_smooth)
            self.map_y = np.float32(y+dy_smooth)

            # remap image
            image = cv2.remap(image, self.map_x, self.map_y, self.interpolation_mode, self.border_mode)

            # remap landmarks
            self.map_x = self.map_x.flatten()
            self.map_y = self.map_y.flatten()
            landmarks = self._map_points(landmarks)

        return image, landmarks

    def _map_points(self, points):

        def _intersect(arrlist_1, arrlist_2):
            """
                Find intersection via Hashmap
            """
            hash_table = {}
            for i, point in enumerate(arrlist_1):
                hash_table[point] = i
            for i, point in enumerate(arrlist_2):
                if point in hash_table:
                    return point
            return None

        def _nearest(arrlist_1, arrlist_2):
            """
                Find nearest coordinate using KDTree
            """
            tree = KDTree(arrlist_1);
            pts = tree.query(arrlist_2)

            return tree.data[pts[1][pts[0].argmin()]]



        new_points = np.zeros_like(points, dtype=np.float32)
        k = 500
        for i,point in enumerate(points):
            if point[0] !=0 and point[1] !=0:
                x_map = np.vstack(
                             np.unravel_index(
                                 np.argpartition(
                                     np.abs(point[0]-self.map_x), k)[:k],
                                     (self.img_height, self.img_width))).transpose()
                x_list = list(tuple(x) for x in list(x_map))

                y_map = np.vstack(
                             np.unravel_index(
                                 np.argpartition(
                                     np.abs(point[1]-self.map_y), k)[:k],
                                     (self.img_height, self.img_width))).transpose()
                y_list = list(tuple(y) for y in list(y_map))

                intersect_point = _intersect(x_list, y_list)

                if intersect_point is None:
                    nearest_point = _nearest(x_map, y_map)
                    nearest_point = [nearest_point[1], nearest_point[0]]
                else:
                    new_points[i,:] = [intersect_point[1], intersect_point[0]]


        return new_points


class DeepFashionCAPDataset(torch.utils.data.Dataset):

    def __init__(self, df, mode, random_state, base_path):
        self.df = df
        self.base_path = base_path
        self.to_tensor = transforms.ToTensor()  # pytorch使用c x h x w的格式转换
        self.rescale = Rescale(256)
        self.rescale_largest_center = Rescale(224)
        self.rescale224square = Rescale((224, 224))
        self.bbox_crop = BBoxCrop()
        self.center_crop = CenterCrop(224)
        self.random_crop = RandomCrop(224)
        # self.random_flip = RandomFlip()
        self.rotate = Rotate(random_state, 180, 0.9)
        self.elastic_transformation = ElasticTransformation(random_state, mode='points', prob=0.6, sigma=40.0, alpha=500, nr_points=3)
        self.check_landmarks = CheckLandmarks()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.landmarks_normalize = LandmarksNormalize()
        self.landmarks_unnormalize = LandmarksUnNormalize()
        self.mode = mode
        assert self.mode in ['ELASTIC_ROTATION_BBOXRESIZE', 'ROTATION_BBOXRESIZE', 'BBOXRESIZE',
                             'MASK_ELASTIC_ROTATION_BBOXRESIZE', 'MASK_ROTATION_BBOXRESIZE', 'MASK_BBOXRESIZE']

        # for vis
        self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.to_pil = transforms.ToPILImage()

    def plot_img_and_lm(self, image, lm):
        nr_points = lm.shape[0]
        plt.figure()
        plt.imshow(image)
        for i in range(nr_points):
            plt.scatter(lm[i, 0], lm[i, 1], s=20, marker='x', c='r')


    def plot_sample(self, i):
        sample = self[i]
        if isinstance(sample['image'], torch.Tensor):
            image = self.unnormalize(sample['image'])
            image = self.to_pil(image.float())
            image = np.array(image)
        plt.figure(dpi=200)
        plt.imshow(image)
        for i, in_pic in enumerate(sample['landmark_in_pic']):
            if (in_pic == 1):
                plt.scatter([sample['landmark_pos'][i, 0]], [sample['landmark_pos'][i, 1]], s=20, marker='.', c='g')
            else:
                plt.scatter([sample['landmark_pos'][i, 0]], [sample['landmark_pos'][i, 1]], s=20, marker='x', c='r')
        plt.show()

    def plot_landmark_map(self, i):
        sample = self[i]
        landmark_map = sample['landmark_map']
        print(landmark_map.shape)
        landmark_map = np.max(landmark_map, axis=0)
        print(landmark_map.shape)
        plt.imshow(landmark_map)

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = read_image(self.base_path + sample['image_name'])
        category_label = sample['category_label']
        landmark_vis = sample.filter(regex='lm.*vis').astype(np.int64).values
        landmark_in_pic = sample.filter(regex='lm.*in_pic').astype(np.int64).values
        landmark_pos_x = sample.filter(regex='lm.*x').astype(np.int64).values.reshape(-1, 1)
        landmark_pos_y = sample.filter(regex='lm.*y').astype(np.int64).values.reshape(-1, 1)
        landmark_pos = np.concatenate([landmark_pos_x, landmark_pos_y], axis=1)
        attr = sample.filter(regex='attr.*').astype(np.int64).values
        category_type = sample['category_type']

        bbox = np.array([[sample.x_1, sample.y_1],
                         [sample.x_2, sample.y_1],
                         [sample.x_1, sample.y_2],
                         [sample.x_2, sample.y_2]])

        first_action = self.mode.split('_')[0]
        if first_action == 'MASK':
            # get background mask
            background_path = "/".join((self.base_path + sample['image_name']).split("/")[:-1])+'/background.png'
            background = read_image(background_path)

            # get roi pixels
            locs = np.where(background >= 200)
            # copy roi into background image
            background[locs[0], locs[1]] = image[locs[0], locs[1]]
            image = background

        if self.mode == 'ROTATION_BBOXRESIZE' or self.mode == 'MASK_ROTATION_BBOXRESIZE':
            image, landmark_pos, bbox = self.rotate(image, landmark_pos, bbox)
            image, landmark_pos = self.bbox_crop(image, landmark_pos, bbox)
            image, landmark_pos = self.rescale224square(image, landmark_pos)
        elif self.mode == 'BBOXRESIZE' or self.mode == 'MASK_BBOXRESIZE':
            image, landmark_pos = self.bbox_crop(image, landmark_pos, bbox)
            image, landmark_pos = self.rescale224square(image, landmark_pos)
        elif self.mode == 'ELASTIC_ROTATION_BBOXRESIZE' or self.mode == 'MASK_ELASTIC_ROTATION_BBOXRESIZE':
            image, landmark_pos, bbox = self.rotate(image, landmark_pos, bbox)
            image, landmark_pos = self.bbox_crop(image, landmark_pos, bbox)
            image, landmark_pos = self.rescale224square(image, landmark_pos)
            image, landmark_pos = self.elastic_transformation(image, landmark_pos)
        else:
            raise NotImplementedError
        landmark_vis, landmark_in_pic, landmark_pos = self.check_landmarks(image, landmark_vis, landmark_in_pic, landmark_pos)

        landmark_pos = landmark_pos.astype(np.float32)
        landmark_pos_normalized = self.landmarks_normalize(image, landmark_pos).astype(np.float32)


        image = image.copy()

        image = self.to_tensor(image)
        image = self.normalize(image)
        image = image.float()

        ret = {}
        ret['image'] = image
        ret['category_type'] = category_type
        ret['category_label'] = category_label
        ret['landmark_vis'] = landmark_vis
        ret['landmark_in_pic'] = landmark_in_pic
        ret['landmark_pos'] = landmark_pos
        ret['landmark_pos_normalized'] = landmark_pos_normalized
        ret['attr'] = attr
        image_h, image_w = image.size()[1:]
        if hasattr(const, 'gaussian_R'):
            R = const.gaussian_R
        else:
            R = 16
        ret['landmark_map'] = gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R)
        ret['landmark_map28'] = gen_landmark_map(int(image_w / 8), int(image_h / 8), landmark_in_pic, landmark_pos / 8, R / 8)
        ret['landmark_map56'] = gen_landmark_map(int(image_w / 4), int(image_h / 4), landmark_in_pic, landmark_pos / 4, R / 4)
        ret['landmark_map112'] = gen_landmark_map(int(image_w / 2), int(image_h / 2), landmark_in_pic, landmark_pos / 2, R / 2)
        ret['landmark_map224'] = gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R)
        return ret


    def __len__(self):
        return len(self.df)
