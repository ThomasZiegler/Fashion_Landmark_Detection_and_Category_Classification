import time as _time
import torch as _torch
import socket as _socket

from src.networks import WholeNetwork as _net
from src.lm_networks import LandmarkBranchUpsample as _lm_branch
from src.utils import InferenceEvaluator as _evaluator


_hostname = str(_socket.gethostname())

_name = 'inference'
_time = _time.strftime('%m-%d_%H:%M:%S', _time.localtime())


# Dataset
#gaussian_R = 8
DATASET_PROC_METHOD_INF = 'BBOXRESIZE'
########

# Network
USE_NET = _net
LM_SELECT_VGG = 'conv4_3'
LM_SELECT_VGG_SIZE = 28
LM_SELECT_VGG_CHANNEL = 512
LM_BRANCH = _lm_branch
EVALUATOR = _evaluator
INIT_MODEL = './whole.pkl_epoch_130780'
INF_BATCH_SIZE = 1
#################

INF_DIR = 'runs/%s/' % _name + _time
TRAIN_DIR = 'runs/%s/' % _name + _time
MODEL_NAME = 'vgg16.pkl'


if _hostname == 'ThinkPad-X1-Yoga':
    base_path = '/home/zieglert/ETH/SA-FL/data/CTU/'
elif _hostname == 'mordor':
    base_path = '/home/thomasz/SA-FL/data/CTU/'
else:
    base_path = '/cluster/scratch/zieglert/CTU/'

device = _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')

lm2name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']
attrtype2name = {1: 'texture', 2: 'fabric', 3: 'shape', 4: 'part', 5: 'style'}

USE_CSV = 'info.csv'

LM_TRAIN_USE = 'vis'
LM_EVAL_USE = 'vis'

USE_IORN = True
