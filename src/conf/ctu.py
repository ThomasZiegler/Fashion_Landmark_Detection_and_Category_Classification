import time as _time
import socket as _socket
from src.networks import CTUNetwork as _net
from src.lm_networks import LandmarkBranchUpsample as _lm_branch
from src.utils import CTUEvaluator as _evaluator
#from src.lm_networks import LandmarkExpNetwork as _net
#from src.lm_networks import LandmarkBranchUpsample as _lm_branch
#from src.utils import LandmarkEvaluator as _evaluator

_name = 'ctu'
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
EVALUATOR = _evaluator
#################

# Learning Scheme
LEARNING_RATE_DECAY = 0.8
WEIGHT_LOSS_LM_POS = 10
SWITCH_LEFT_RIGHT = True

NUM_EPOCH = 500
LEARNING_RATE_STEP = 25
EARLYSTOPPING_THRESHOLD = 500
LOG_INTERVAL_SCALAR = 1
LOG_INTERVAL_IMAGE = 5
#################

if _hostname == 'ThinkPad-X1-Yoga':
    base_path = '/home/zieglert/ETH/SA-FL/data/CTU/'
elif _hostname == 'mordor':
    base_path = '/home/thomasz/SA-FL/data/CTU/'
else:
    base_path = '/cluster/scratch/zieglert/CTU/'

USE_CSV = 'new_info.csv'
USE_IORN = False

# auto
TRAIN_DIR = 'runs/%s/' % _name + _time
VAL_DIR = 'runs/%s/' % _name + _time

MODEL_NAME = '%s.pkl' % _name
#############

# Category labels
CATEGORY_NAMES = {0: 'bluse',
                  1: 'hoody',
                  2: 'pants',
                  3: 'polo',
                  4: 'polo-long',
                  5: 'skirt',
                  6: 'tshirt',
                  7: 'tshirt-long'}
