import time as _time
import torch as _torch
import socket as _socket

_hostname = str(_socket.gethostname())

name = _time.strftime('%m-%d_%H:%M:%S', _time.localtime())


USE_NET = 'VGG16'

TRAIN_DIR = 'runs/' + name
VAL_DIR = 'runs/' + name

FASHIONET_LOAD_VGG16_GLOBAL = False

DATASET_PROC_METHOD_TRAIN = 'ROTATION_BBOXRESIZE'
DATASET_PROC_METHOD_VAL = 'ROTATION_BBOXRESIZE'

# 0: no sigmoid 1: sigmoid
VGG16_ACT_FUNC_IN_POSE = 0

MODEL_NAME = 'vgg16.pkl'

if _hostname == 'ThinkPad-X1-Yoga':
    base_path = '/home/zieglert/ETH/SA-FL/data/AttributePrediction/'
elif _hostname == 'mordor':
    base_path = '/home/thomasz/SA-FL/data/AttributePrediction/'
else:
#    base_path = '/cluster/scratch/zieglert/AttributePrediction/'
    base_path = '/cluster/scratch/zieglert/LandmarkDetection/'
    TRAIN_DIR = base_path + 'runs/' + name
    VAL_DIR = base_path + 'runs/' + name
    STDOUT_FILE = base_path + 'runs/' + name + '/std_out'


NUM_EPOCH = 50
LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY = 0.9
LEARNING_RATE_STEP = 5
BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
EARLYSTOPPING_THRESHOLD = 10
LOG_INTERVAL_SCALAR = 10
LOG_INTERVAL_IMAGE = 100
RANDOM_SEED = 17
SE_REDUCTION = 16

WEIGHT_ATTR_NEG = 0.1
WEIGHT_ATTR_POS = 1
WEIGHT_LANDMARK_VIS_NEG = 0.5
WEIGHT_LANDMARK_VIS_POS = 0.5


# LOSS WEIGHT
WEIGHT_LOSS_CATEGORY = 1
WEIGHT_LOSS_ATTR = 20
WEIGHT_LOSS_LM_POS = 100


# VAL
VAL_CATEGORY_TOP_N = (1, 3, 5)
VAL_ATTR_TOP_N = (3, 5)
VAL_LM_RELATIVE_DIS = 0.1

device = _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')

lm2name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']
attrtype2name = {1: 'texture', 2: 'fabric', 3: 'shape', 4: 'part', 5: 'style'}

VAL_WHILE_TRAIN = True

USE_CSV = 'info.csv'

LM_TRAIN_USE = 'vis'
LM_EVAL_USE = 'vis'

USE_IORN = True 
VGG_INIT_MODEL = "./model_best.pth.tar"


CATEGORY_NAMES = {0: 'Anorak',
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
