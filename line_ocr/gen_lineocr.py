import numpy as np
import pickle
import os
import sys
import json
import cv2
from keras.preprocessing.image import ImageDataGenerator
import json
from matplotlib import pyplot as pltfr
#import ocrolib.hwocr.mnist_helper as mh
import pkg_resources
import unicodedata
#from ..hwocr import mnist_helper as mh

PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../')
import ocrolib.hwocr.mnist_helper as mh

def split_bylen(item, maxlen):
    '''
    Requires item to be sliceable (with __getitem__ defined)
    '''
    return [item[ind:ind+maxlen] for ind in range(0, len(item), maxlen)]

def read_img_from_file_nice(filename, h_shape=64):
    img = cv2.imread(filename, cv2.CV_8UC1)
    img = mh.do_cropping(img, max_cobj=10, negated=True, ignore_width=True)
    img = cv2.resize(img, (img.shape[1] * h_shape // img.shape[0], h_shape),
                     interpolation=cv2.INTER_CUBIC)
    # _, img = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                            cv2.THRESH_BINARY, 3, 2)
    img = 255 - img
    img = img.reshape([img.shape[0], img.shape[1], 1])
    return img

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

"""
This generator assumes there are image and config files in the source
- image data: pickle file contains matrix X and Y of the images and their index label
- meta data: json files contain the mapping between index and the character and vice versa (.label2char.json and .label2char.json)
- config file: json file controls the generate options
"""
class LineOCRGenerator:
    def __init__(self, h_shape=32, limit_per_char=10,
                 config_dir='./data/line_config.json',
                 save_gallery_dir='./gallery/', fix_amount=-1):
        self.limit_per_char=limit_per_char
        self.char2imgs = {}
        self.char2label = {}
        self.label2char = {}
        self.all_stri = []
        self.fix_amout=fix_amount
        self.batch_db=[]
        self.save_gallery_dir=save_gallery_dir
        if not os.path.isdir( self.save_gallery_dir):
            os.mkdir(self.save_gallery_dir)
        self.db_name = ''
        self.h_shape = h_shape
        self.configs= json.load(open(pkg_resources.resource_filename(__name__, config_dir)))
        self.datagen = ImageDataGenerator(rotation_range=self.configs["rotation_range"],
                                          zoom_range=self.configs["zoom_range"],
                                          shear_range=self.configs["shear_range"],
                                          data_format='channels_last')


    def load_db(self, data_dir):
        self.db_name +=   os.path.basename(data_dir)[:-4]
        # char2label = json.load(open(data_dir[:-4] + '.char2label.json'))
        label2char = json.load(open(pkg_resources.resource_filename(__name__,data_dir[:-4] + '.label2char.json')))
        print(label2char)
        print(data_dir)
        X_train, y_train, X_test, y_test = pickle.load(open(pkg_resources.resource_filename(__name__,data_dir), 'rb'))
        # self.datagen.fit(X_train)


        for i in range(X_train.shape[0]):
            if i%1000==0:
                llprint("\r{}/{}".format(i,X_train.shape[0]))
            try:
                if len(y_train.shape) == 1:
                    id = y_train[i]
                else:
                    id = np.argmax(y_train[i])
                if not label2char[str(id)] in self.char2imgs:
                    self.char2imgs[label2char[str(id)]] = [X_train[i][0] * 255]
                    # print(np.max(X_train[i][0]))
                    # print(np.min(X_train[i][0]))
                else:
                    if len(self.char2imgs[label2char[str(id)]]) < self.limit_per_char:
                        self.char2imgs[label2char[str(id)]].append(X_train[i][0] * 255)
            except:
                pass
        print('\ndone train data')
        for i in range(X_test.shape[0]):
            if i % 1000 == 0:
                llprint("\r{}/{}".format(i, X_test.shape[0]))
            try:
                if len(y_test.shape) == 1:
                    id = y_test[i]
                else:
                    id = np.argmax(y_test[i])
                if not label2char[str(id)] in self.char2imgs:
                    self.char2imgs[label2char[str(id)]] = [X_test[i][0] * 255]
                    # print(np.max(X_train[i][0]))
                    # print(np.min(X_train[i][0]))
                else:
                    if len(self.char2imgs[label2char[str(id)]]) < self.limit_per_char:
                        self.char2imgs[label2char[str(id)]].append(X_test[i][0] * 255)
            except:
                pass

        print('\ndone test data')

    def finalize_all_db(self):
        self.char2label['_pad_']=0
        self.label2char[0] = '_pad_'
        for k,v in sorted(self.char2imgs.items()):
            cl = len(self.label2char)
            for k2 in k:
                if k2 not in self.char2label:
                    self.char2label[k2]=cl
                    self.label2char[cl] = k2
            new_v=[]
            for im in v:
                new_v.append(255-cv2.resize(im, (self.h_shape, self.h_shape), interpolation=cv2.INTER_NEAREST))
            self.char2imgs[k]=new_v
        print('num classes {}'.format(len(self.char2label)))

    def save_gallery(self, limit=5):
        print('start save gallery')
        print('number of char {}'.format(len(self.char2imgs)))
        ndir=self.save_gallery_dir+'/'+self.db_name+'/'
        if not os.path.isdir(ndir):
            os.mkdir(ndir)
        for k,v in self.char2imgs.items():
            for ci, c in enumerate(v):
                cv2.imwrite(ndir+ k +'_' +str(ci) + '.png', c)
                if ci>limit:
                    break

    def generate_single_image(self, char):
        img_list=[]
        if char not in self.char2imgs:
            for k,v in self.char2imgs.items():
                if char in k:
                    img_list=v
                    break
        else:
            img_list=self.char2imgs[char]
        if img_list:
            img_list2=[]
            for im in img_list:
                if len(im.shape)==2:
                    img_list2.append(np.expand_dims(im,2))
                else:
                    img_list2.append(im)
            for x, _ in self.datagen.flow(np.asarray(img_list2),
                                      np.asarray(len(img_list2)*[0]),
                                      batch_size=1):
                # print(x[0][:][:].shape)
                return x[0][:], char
        return None, None


    def generate_sequence_image(self, stri):
        img_seq=[]
        char_seq=[]
        for char in stri:
            x, c = self.generate_single_image(char)
            if x is not None:
                img_seq.append(x)
                char_seq.append(c)
            # else:
            #   print(char)
        return img_seq, char_seq

    def combine_img_seq(self, img_seq):
        #assume all 2d images have the same length, background black, white stroke
        if not img_seq:
            return None
        h = img_seq[0].shape[0]
        w = img_seq[0].shape[1]
        n_img_seq=[]
        img_id=0
        while img_id<len(img_seq):
            img=img_seq[img_id]
            space=np.random.randint(self.configs['space_range'][0], self.configs['space_range'][1],1)[0]
            if space>0 or img_id+1==len(img_seq):
                n_img_seq.append(img)
            else:
                neww=w*2-abs(space)
                mat1=np.zeros((h, neww, 1))
                mat1[:,:w, :]=img
                mat2 = np.zeros((h, neww, 1))
                mat2[:,-w:, :]=img_seq[img_id+1]
                mat3=np.maximum(mat1,mat2)
                # print(mat3.shape)
                n_img_seq.append(mat3)
                img_id+=1
            #make sapce image
            if space>=1:
                smat=np.zeros((h, space, 1))
                n_img_seq.append(smat)
            img_id += 1
            # print(img.shape)
        nimg=np.concatenate(n_img_seq,axis=1)

        if self.configs['erode']:
            rd = np.random.rand()
            nerod=0
            for k,v in self.configs['erode']:
                if rd>k:
                    nerod=v
                    break
            if nerod>0:
                kernel = 1*np.ones((2, 2), np.uint8)
                nimg =  cv2.erode(nimg, kernel, iterations=nerod)
                nimg = np.expand_dims(nimg,2)
        # print(nimg.shape)
        return nimg


    def equip_txt(self, txt_file='./data/japanese_name/name_full_kanji_katakana.txt',
                  is_random=False,
                  top_freq=True,
                  max_len=-1,
                  ignore_strange=True,
                  combine_text=1,
                  max_char_per_line=20):
        nfname = txt_file[:-4] + '_{}.txt'.format(max_len)
        all_stri = []
        if top_freq and not is_random and os.path.isfile(nfname):
            txt_file=nfname
            max_len=-1
        try:
            with open(txt_file) as f:
                for l in f:
                    l = unicodedata.normalize('NFKC', l)
                    if ignore_strange and len(l)>0 and l[0] in self.char2label:
                        all_stri.append(str(l).strip())
        except:
            with open(pkg_resources.resource_filename(__name__, txt_file), encoding='utf-8') as f:
                for l in f:
                    l = unicodedata.normalize('NFKC', l)
                    if ignore_strange and len(l) > 0 and l[0] in self.char2label:
                        all_stri.append(str(l).strip())
        if max_len>0:
            if not top_freq:
                if not is_random:
                    all_stri=all_stri[:max_len]
                else:
                    all_stri=np.random.choice(all_stri, max_len)
                    #print(self.all_stri)
            else:
                cw={}
                for si, st in enumerate(all_stri):
                    if si%100==0:
                        llprint('\rcounted {}/{}'.format(si, len(all_stri)))
                    for c in st:
                        if c not in cw:
                            cw[c]=0
                        cw[c]+=1
                list_score=[]
                print('\n-----')
                for si, st in enumerate(all_stri):
                    if si%100==0:
                        llprint('\rscored {}/{}'.format(si, len(all_stri)))
                    cur_s=0
                    for c in st:
                        cur_s+=cw[c]
                    list_score.append(cur_s)
                print('\n-----')
                sort_strs=[x for _, x in sorted(zip(list_score, all_stri), reverse=True)]
                all_stri = sort_strs[:max_len]
                with open(nfname, 'w', encoding='utf-8') as f:
                    for si, st in enumerate(all_stri):
                        if si % 100 == 0:
                            llprint('\rwrited {}/{}'.format(si, len(all_stri)))
                        f.write(st)
                        f.write('\n')
        # raise False
        # np.random.shuffle(all_stri)
        print('num name {}'.format(len(all_stri)))
        if combine_text>1:
            c=0
            new_all_stri=[]
            while c<len(all_stri)-combine_text:
                news=''
                for j in range(combine_text):
                    news+=all_stri[c+j]
                new_all_stri.append(news)
                c+=combine_text
            all_stri=new_all_stri

        for stri in all_stri:
            if len(stri)<max_char_per_line:
                self.all_stri.append(stri)
            else:
                stris=split_bylen(stri, max_char_per_line)
                for st in stris:
                    self.all_stri.append(st)
        print('num string {}'.format(len(self.all_stri)))

    def test_generate_lineocr(self, num=10, save_dir='./data/japanese_name/line/'):
        text_ind = np.random.randint(0, len(self.all_stri), num)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        for i in range(num):
            if i%1==0:
                llprint("\r{}/{}".format(i, num))
            stri=self.all_stri[text_ind[i%len(self.all_stri)]]
            imgs, chars=self.generate_sequence_image(stri)
            chars=''.join(chars)
            if len(imgs)!=len(stri):
                print('st wrong')
            if imgs:
                img=self.combine_img_seq(imgs)
                cv2.imwrite(save_dir + chars + '_' + str(i) + '.png', 255-img)


    def prepare_samples(self, all_imgs, all_chars, batch_size):

        max_w = 0
        max_h = 0
        max_c = 0
        for im in all_imgs:
            max_h = max(max_h, im.shape[0])
            max_w = max(max_w, im.shape[1])
            max_c = max(max_c, im.shape[2])
        batch_input = np.zeros([batch_size, max_h, max_w, max_c])
        batch_output = []
        # print(self.char2label)
        for b in range(batch_size):
            if b < len(all_imgs):
                im = all_imgs[b]
                chars = all_chars[b]
                batch_input[b, :im.shape[0], :im.shape[1], :im.shape[2]] = im / 255
                # x=255*np.transpose(batch_input[b],axes=[2,0,1])
                # plt.imshow(x[0],cmap='gray')
                # plt.show()
                labels = []
                for c in chars:
                    if c in self.char2label:
                        labels.append(self.char2label[c])
                    else:
                        labels.append(0)
                batch_output.append(labels)
            else:
                batch_output.append([0])
        return batch_input, batch_output, np.ones([batch_size]) * batch_input.shape[2]

    def prepare_dynamic_samples(self, all_imgs, all_chars, batch_size):
        max_w = 0
        max_h = 0
        max_c = 0
        for im in all_imgs:
            max_h = max(max_h, im.shape[0])
            max_w = max(max_w, im.shape[1])
            max_c = max(max_c, im.shape[2])
        batch_input = np.zeros([batch_size, max_h, max_w, max_c])
        widths = np.zeros([batch_size])
        batch_output = []
        # print(self.char2label)
        for b in range(batch_size):
            if b < len(all_imgs):
                im = all_imgs[b]
                chars = all_chars[b]
                batch_input[b, :im.shape[0], :im.shape[1], :im.shape[2]] = im / 255
                widths[b] = im.shape[1]
                # x=255*np.transpose(batch_input[b],axes=[2,0,1])
                # plt.imshow(x[0],cmap='gray')
                # plt.show()
                labels = []
                for c in chars:
                    if c in self.char2label:
                        labels.append(self.char2label[c])
                    else:
                        labels.append(0)
                batch_output.append(labels)
            else:
                batch_output.append([0])
        # print(batch_output)
        return batch_input, batch_output, widths

    def generate_lineocr_samples(self, batch_size=10, is_dynamic=False):
        all_imgs=[]
        all_chars=[]

        text_ind = np.random.randint(0, len(self.all_stri), batch_size)
        for i in range(batch_size):
            # if i%1000==0:
            #     llprint("\r{}/{}".format(i, batch_size))

            if self.fix_amout<0:
                stri = self.all_stri[text_ind[i]]
                imgs, chars = self.generate_sequence_image(stri)
            else:
                if len(self.batch_db)<self.fix_amout:
                    stri = self.all_stri[text_ind[i]]
                    imgs, chars = self.generate_sequence_image(stri)
                    self.batch_db.append((stri, imgs, chars))
                else:
                    rind = np.random.choice(range(len(self.batch_db)), size=1)[0]
                    stri, imgs, chars = self.batch_db[rind]
            # if len(imgs)!=len(stri):
            #     print('st wrong')
            if imgs:
                all_imgs.append(self.combine_img_seq(imgs))
                all_chars.append(chars)

        if is_dynamic:
            return self.prepare_dynamic_samples(all_imgs, all_chars, batch_size)
        else:
            return self.prepare_samples(all_imgs, all_chars, batch_size)


    def generate_samples_from_files(self, dir_test='./data/japanese_name/test_data/', is_dynamic=False):
        imgs = []
        chars=[]
        for file in sorted(os.listdir(dir_test)):
            filename = dir_test + file
            if os.path.isfile(filename):
                img = read_img_from_file_nice(filename, self.h_shape)

                imgs.append(img)
                file=file.split('.')[0]
                chars.append(file.split('_')[0])

        if is_dynamic:
            return self.prepare_dynamic_samples(imgs, chars, len(imgs))
        else:
            return self.prepare_samples(imgs, chars, len(imgs))

    def generate_one_sample(self, filename, is_dynamic=False):
        img = read_img_from_file_nice(filename, self.h_shape)
        label = filename.split('.')[0]
        if is_dynamic:
            return self.prepare_dynamic_samples([img], [label], 1)
        else:
            return self.prepare_samples([img], [label], 1)



if __name__ == '__main__':
    lineOCR = LineOCRGenerator(h_shape=64, limit_per_char=1000)
    lineOCR.load_db(data_dir='../hwocr/data/kanji_hira_plain.pkl')
    lineOCR.load_db(data_dir='../hwocr/data/full_katakana_quote.nice.pkl')
    lineOCR.load_db(data_dir='../hwocr/data/mnist_full.nice.pkl')
    lineOCR.load_db(data_dir='../hwocr/data/latin_special.pkl')
    lineOCR.finalize_all_db()
    print(lineOCR.char2label)
    lineOCR.equip_txt(max_len=-1, is_random=True, top_freq=False, combine_text=1, txt_file='./data/japanese_name/name_full_kanji_katakana.txt')
    lineOCR.equip_txt(max_len=-1, is_random=True, top_freq=False, combine_text=1, txt_file='./data/japanese_address/full_thres_95.txt')
    # lineOCR.save_gallery(limit=3)
    lineOCR.test_generate_lineocr(num=20, save_dir='./data/test20name_add/')
    # batch_input, batch_output = lineOCR.generate_lineocr_samples(batch_size=100)
    # print(batch_input.shape)
    # print(batch_output)
