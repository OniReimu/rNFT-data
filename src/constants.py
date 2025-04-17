"""
 @author suncj
 @email sun.cj@zhejianglab.com
 @create date 2023-09-20 19:57:33
 @modify date 2023-11-20 15:34:31
 @desc [description]
"""
import torch

TRANS_TYPE_NONE = 0
TRANS_TYPE_DATASET = 1
TRANS_TYPE_MODEL = 2

DATASET_DIM = 10
MODEL_DIM = 10

EPISODES = 100
EPOCHS = 150

BATCH_SIZE = 128
EXTENDED_DELAY_STEPS_FACTOR = 2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
