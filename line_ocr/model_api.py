import train
from collections import namedtuple


my_struct = namedtuple("model_args","mode learning_rate decay_rate decay_staircase momentum decay_steps fix_amount max_len_text batch_size")
args= my_struct("test",0.001,0.9,True,0.9,2**16,-1,-1,1)
print(args.mode)


sess, decoded, cost, image, label, width, lineOCR = train.load_model(args)
# all_preds = train.predict_folder(sess,  args, decoded, cost, image, label, width, lineOCR,
#                                  folder='./data/japanese_name/real_data/name_packaged/')
# print(all_preds)
pred = train.predict_path(sess,  args, decoded, cost, image, label, width, lineOCR,
                                path='./data/japanese_name/real_data/name_packaged/アサユリカ.png')
print(pred)