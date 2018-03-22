def abs_acc(predict_str, real_str):
    if predict_str==real_str:
        return 1
    return 0

def set_acc(predict_str, real_str):
    acc = len(set(predict_str).intersection(set(real_str))) / len(set(real_str))
    return acc

import editdistance as ed
def nedit_dist(predict_str, real_str):
    return ed.eval(predict_str, real_str) / max(len(predict_str), len(real_str))