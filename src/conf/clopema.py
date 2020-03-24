import time as _time
import socket as _socket
from src.networks import CloPeMaNetwork as _net
from src.lm_networks import LandmarkBranchUpsample as _lm_branch
from src.utils import CloPeMaEvaluator as _evaluator
#from src.lm_networks import LandmarkExpNetwork as _net
#from src.lm_networks import LandmarkBranchUpsample as _lm_branch
#from src.utils import LandmarkEvaluator as _evaluator

_name = 'clopema'
_time = _time.strftime('%m-%d_%H:%M:%S', _time.localtime())
_hostname = str(_socket.gethostname())

# Dataset
gaussian_R = 8
DATASET_PROC_METHOD_TRAIN = 'ROTATION_BBOXRESIZE'
DATASET_PROC_METHOD_VAL = 'BBOXRESIZE'
########

# Network
USE_NET = _net
LM_SELECT_VGG = 'conv4_3'
LM_SELECT_VGG_SIZE = 28
LM_SELECT_VGG_CHANNEL = 512
LM_BRANCH = _lm_branch
LM_INIT_MODEL = None
FREEZE_LM_NETWORK = False 
EVALUATOR = _evaluator
#################

# Learning Scheme
LEARNING_RATE_DECAY = 0.8
WEIGHT_LOSS_LM_POS = 10
SWITCH_LEFT_RIGHT = True

NUM_EPOCH = 500
LEARNING_RATE_STEP = 25 
EARLYSTOPPING_THRESHOLD = 150
LOG_INTERVAL_SCALAR = 1
LOG_INTERVAL_IMAGE = 5
#################

# auto
TRAIN_DIR = 'runs/%s/' % _name + _time
VAL_DIR = 'runs/%s/' % _name + _time


if _hostname == 'ThinkPad-X1-Yoga':
    base_path = '/home/zieglert/ETH/SA-FL/data/CloPeMa/clothes_dataset_RH/'
elif _hostname == 'mordor':
    base_path = '/home/thomasz/SA-FL/data/CloPeMa/clothes_dataset_RH/'
else:
    base_path = '/cluster/scratch/zieglert/CloPeMa/clothes_dataset_RH/'
    TRAIN_DIR = base_path + 'runs/%s/' % _name + _time
    VAL_DIR = base_path + 'runs/%s/' % _name + _time
    STDOUT_FILE = base_path + 'runs/%s/' % _name + _time + '/stdout'


USE_CSV = 'info.csv'
USE_IORN = True 

MODEL_NAME = '%s.pkl' % _name
#############

# Category labels
#CATEGORY_NAMES = {0: 'bluse',
#                  1: 'hoody',
#                  2: 'pants',
#                  3: 'polo',
#                  4: 'polo-long',
#                  5: 'skirt',
#                  6: 'tshirt',
#                  7: 'tshirt-long'}
#

CATEGORY_NAMES = {0: 't-shirt',
                  1: 'shirt',
                  2: 'thick-sweater',
                  3: 'jean'}

SWEATER_LABELS = [10, 11, 13, 17, 18, 46, 47, 48, 49, 50]
JEAN_LABELS = [12, 14, 15, 16, 19, 20, 26, 31, 36, 37]
SHIRT_LABELS = [6, 7, 8, 9, 22, 23, 32, 33, 34, 35]
TSHIRT_LABELS = [1, 2, 3, 4, 5, 27, 28, 29, 30, 45]



