from . import train
import tensorflow as tf
from collections import namedtuple
import os
from pkg_resources import resource_filename as rf

my_struct = namedtuple("model_args","mode learning_rate decay_rate decay_staircase momentum decay_steps fix_amount max_len_text batch_size")
args= my_struct("test",0.001,0.9,True,0.9,2**16,-1,-1,1)
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
sess1, decoded, cost, image, label, width, lineOCR = train.load_model(args) 

def run(path):
    pred = train.predict_path(sess1,  args, decoded, cost, image, label, width, lineOCR,
                                path=path)
    return pred
