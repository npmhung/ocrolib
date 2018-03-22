# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import tensorflow as tf
from tensorflow.contrib import learn
import sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../')
from image_loader import *
import model
from c_metrics import *
import time


def _get_training(args, rnn_logits,label,sequence_length):
    """Set up training ops"""

    with tf.name_scope("train"):

        scope="convnet|rnn"

        rnn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope)

        loss = model.ctc_loss_layer(rnn_logits,label,sequence_length) 

        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        with tf.control_dependencies(extra_update_ops):

            learning_rate = tf.train.exponential_decay(
                args.learning_rate,
                tf.train.get_global_step(),
                args.decay_steps,
                args.decay_rate,
                staircase=args.decay_staircase,
                name='learning_rate')

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=args.momentum)
            
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate, 
                optimizer=optimizer,
                variables=rnn_vars)

    return train_op, loss

def _get_session_config():
    """Setup session config to soften device placement"""

    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config




def build_model(args):

    #image,width,label = _get_input()
    image = tf.placeholder(tf.float32, [args.batch_size,32,426,4])
    x_tensor = image #tf.expand_dims(image,-1)
    print(x_tensor)
    width = tf.constant(value=426, shape=[args.batch_size])
    label = tf.sparse_placeholder(tf.int32)
    if args.mode == 'train':
        mode = learn.ModeKeys.TRAIN  # 'Configure' training mode for dropout layers
    else:
        mode = learn.ModeKeys.INFER
    features,seq_len = model.convnet_layers( x_tensor, width, mode)
    logits = model.rnn_layers( features, seq_len,num_classes=102)
    train_op, cost = _get_training(args, logits,label,seq_len)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len,
                                                    beam_width=5, top_paths=1, merge_repeated=False)
    print(features)
    print(logits)
    print(decoded)
    return train_op, cost, image, label, width, decoded, None

def build_model2(args):

    import gen_lineocr
    lineOCR = gen_lineocr.LineOCRGenerator(h_shape=64, limit_per_char=500)
    lineOCR.load_db(data_dir='../hwocr/data/kanji_hira.pkl')
    lineOCR.load_db(data_dir='../hwocr/data/full_katakana_quote.nice.pkl')
    lineOCR.finalize_all_db()
    lineOCR.equip_txt(max_len=1000, is_random=True, top_freq=False)
    num_classes=len(lineOCR.label2char)
    print(lineOCR.label2char)
    print(lineOCR.char2label)
    #image,width,label = _get_input()
    image = tf.placeholder(tf.float32, [args.batch_size,lineOCR.h_shape,None,1])
    x_tensor = image #tf.expand_dims(image,-1)
    print(x_tensor)
    width = tf.placeholder(tf.int32, shape=[args.batch_size])
    label = tf.sparse_placeholder(tf.int32)
    if args.mode == 'train':
        mode = learn.ModeKeys.TRAIN  # 'Configure' training mode for dropout layers
    else:
        mode = learn.ModeKeys.INFER
    features,seq_len = model.convnet_layers( x_tensor, width, mode)
    logits = model.rnn_layers( features, seq_len,num_classes=num_classes)
    train_op, cost = _get_training(args, logits,label,seq_len)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_len,
                                                    beam_width=5, top_paths=1, merge_repeated=False)
    print(features)
    print(logits)
    print(decoded)
    return train_op, cost, image, label, width, decoded, lineOCR


def train_name(args):
    model_path = args.model_path
    n_epochs = args.n_epochs
    n_train = args.n_train
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.Graph()
    with graph.as_default():
        tf.contrib.framework.get_or_create_global_step()
        train_op, cost, image, label, width, decoded, lineOCR = build_model2(args)
        with tf.Session(graph=graph) as sess:
            if not os.path.isdir(args.log_dir):
                os.mkdir(args.log_dir)
            train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            if args.is_restore:
                print("Restoring Checkpoint %s ... " % args.model_path)
                saver.restore(sess, args.model_path)
            min_cost = 100
            if args.mode!='train':
                n_epochs=1
            for epoch in range(n_epochs):
                total_cost = 0.0
                if args.mode == 'train':
                    for i_train in range(n_train):
                        # batch_imgs, batch_lbls = Next_Batch(batch_size)
                        # batch_imgs, batch_lbls = lineOCR.generate_lineocr_samples(batch_size=batch_size)
                        batch_imgs, batch_lbls, widths = lineOCR.generate_lineocr_samples(batch_size=args.batch_size,
                                                                                          is_dynamic=args.is_dynamic)
                        # print(batch_imgs.shape)
                        # print(batch_imgs[0])
                        # print(batch_lbls)
                        # raise False
                        # imgs_len = [30 for _ in range(batch_size)]
                        target_lbls = Labels_To_Sparse_Tuple(batch_lbls)
                        feeds = {image: batch_imgs, label: target_lbls, width:widths}
                        __, train_cost = sess.run([train_op, cost], feed_dict=feeds)
                        total_cost += train_cost
                    total_cost /= n_train

                print('Epoch %5d: %f' % (epoch + 1, total_cost))
                if epoch % args.eval_step == 0:
                    if args.mode=='train' and min_cost > total_cost:
                        min_cost = total_cost
                        saver.save(sess, model_path)
                        print('save model with cost {}'.format(min_cost))
                    # batch_imgs, batch_lbls = Next_Batch(batch_size)
                    # batch_imgs, batch_lbls = lineOCR.generate_lineocr_samples(batch_size=batch_size)
                    test_score=0
                    abs_acc_score=[]
                    set_acc_score=[]
                    ned=[]

                    if args.mode=='test_file':
                        start_time = time.time()
                        fo=open(args.test_file_out,'w')
                        batch_imgs, batch_lbls, widths = lineOCR.generate_samples_from_files(is_dynamic=args.is_dynamic)
                        ntb = len(batch_imgs) // args.batch_size + 1
                        for ii in range(ntb):
                            if ii * args.batch_size == len(batch_imgs):
                                break
                            bs = [ii * args.batch_size, min((ii + 1) * args.batch_size, len(batch_imgs))]
                            rs = bs[1] - bs[0]
                            if bs[1] >= len(batch_imgs):
                                bs = [len(batch_imgs) - args.batch_size, len(batch_imgs)]
                            # print(bs)
                            imgs=batch_imgs[bs[0]:bs[1]]
                            lbls= Labels_To_Sparse_Tuple(batch_lbls[bs[0]:bs[1]])

                            ws=widths[bs[0]:bs[1]]
                            feeds = {image: imgs, label: lbls, width: ws}
                            result, tcost = sess.run([decoded, cost], feed_dict=feeds)
                            test_score += tcost
                            predicted_lbls = Sparse_Tuple_To_Labels(result[0])
                            for seq, rseq in zip(predicted_lbls[:rs], batch_lbls[bs[0]:bs[1]][:rs]):
                                stri = ''
                                rstri = ''
                                for s in seq:
                                    stri += lineOCR.label2char[s]
                                for s in rseq:
                                    rstri += lineOCR.label2char[s]
                                if ii == 0:
                                    print('{} vs {}'.format(seq, rseq))
                                    print('predict {} vs real {}'.format(stri, rstri))
                                fo.write('{} vs {}\n'.format(stri, rstri))
                                abs_acc_score.append(abs_acc(stri, rstri))
                                set_acc_score.append(set_acc(stri, rstri))
                                ned.append(nedit_dist(stri, rstri))
                        test_score /= ntb
                        fo.close()
                        print('*Result:  {}'.format(test_score))
                        print('metric abs acc {} set acc {} nedit {}'.format(
                            np.mean(abs_acc_score), np.mean(set_acc_score), np.mean(ned)))

                        print("--- %s seconds ---" % (time.time() - start_time))
                    else:
                        for ii in range(args.n_test):
                            batch_imgs, batch_lbls, widths = lineOCR.generate_lineocr_samples(
                                batch_size=args.batch_size, is_dynamic=args.is_dynamic)
                            target_lbls = Labels_To_Sparse_Tuple(batch_lbls)

                            # print(batch_imgs[0])
                            # print(batch_lbls[0])
                            feeds = {image: batch_imgs, label: target_lbls, width: widths}
                            result, tcost = sess.run([decoded, cost], feed_dict=feeds)
                            test_score+=tcost
                            predicted_lbls = Sparse_Tuple_To_Labels(result[0])
                            for seq, rseq in zip(predicted_lbls, batch_lbls):
                                stri=''
                                rstri=''
                                for s in seq:
                                    stri+=lineOCR.label2char[s]
                                for s in rseq:
                                    rstri+=lineOCR.label2char[s]
                                if ii==0:
                                    print('{} vs {}'.format(seq,rseq))
                                    print('predict {} vs real {}'.format(stri, rstri))
                                abs_acc_score.append(abs_acc(stri, rstri))
                                set_acc_score.append(set_acc(stri, rstri))
                                ned.append(nedit_dist(stri, rstri))
                        test_score /= args.n_test
                        print('*Result:  {}'.format(test_score))
                        print('metric abs acc {} set acc {} nedit {}'.format(
                            np.mean(abs_acc_score), np.mean(set_acc_score), np.mean(ned)))
                        summary = tf.Summary()
                        summary.value.add(tag='train_loss', simple_value=test_score)
                        summary.value.add(tag='abs_acc_score', simple_value=np.mean(abs_acc_score))
                        summary.value.add(tag='set_acc_score', simple_value=np.mean(set_acc_score))
                        summary.value.add(tag='ned', simple_value=np.mean(ned))
                        train_writer.add_summary(summary, epoch)
                        train_writer.flush()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default="train")
    parser.add_argument('--model_path', default='./model/conv-gru64-1000')
    parser.add_argument('--log_dir', default='./log/conv-gru64-1000')
    parser.add_argument('--test_file_out', default='./data/test_result_ctc64.txt')
    parser.add_argument('--n_epochs', default=100000, type=int)
    parser.add_argument('--n_train', default=10, type=int)
    parser.add_argument('--n_test', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--decay_rate', default=0.9, type=float)
    parser.add_argument('--decay_staircase', default=False, type=str2bool)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--eval_step', default=10, type=int)
    parser.add_argument('--decay_steps', default=2**16, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--is_restore', default=False, type=str2bool)
    parser.add_argument('--is_dynamic', default=False, type=str2bool)
    args = parser.parse_args()

    # args.is_restore = True
    # args.is_dynamic = False
    # args.mode = 'test_file'


    print(args)

    train_name(args)