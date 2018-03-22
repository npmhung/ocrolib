import os
import sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../')
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import json
import random
import argparse
import numpy as np
import cv2
import matplotlib as mpl
# mpl.use('Agg')
import ocrolib.hwocr.run_hw2 as rh
from matplotlib import pyplot as plt



def add_blend_img(bg, train_img,alpha =0.8, img_rows=64, img_cols=64, shift_hor=0, shift_ver=0):
    # print(shift_hor)
    # print(shift_ver)
    im1 = cv2.resize(bg, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    # im2 = cv2.resize(train_img, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    im1 = cv2.normalize(im1, None, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # plt.imshow(im1, cmap='gray', interpolation='nearest')
    # plt.show()
    im2 = cv2.normalize(train_img, None, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # plt.imshow(im2, cmap='gray', interpolation='nearest')
    # plt.show()
    # out = im1 * (1.0 - alpha) + im2 * alpha
    if shift_hor!=0:
        im1s = 255*np.ones((im1.shape[0] + abs(shift_hor), im1.shape[1]), np.uint8)
        im2s = 255*np.ones((im2.shape[0] + abs(shift_hor), im2.shape[1]), np.uint8)

        if shift_hor>0:
            im1s[shift_hor:, :] = im1
            im2s[:-shift_hor, :] = im2
        else:
            im1s[:shift_hor, :] = im1
            im2s[-shift_hor:, :] = im2
        im1=im1s
        im2=im2s

    if shift_ver!=0:
        im1s = 255*np.ones((im1.shape[0], im1.shape[1] + abs(shift_ver)), np.uint8)
        im2s = 255*np.ones((im2.shape[0], im2.shape[1] + abs(shift_ver)), np.uint8)
        if shift_ver>0:
            im1s[:,shift_ver:] = im1
            im2s[:,:-shift_ver] = im2
        else:
            im1s[:,:shift_ver] = im1
            im2s[:,-shift_ver:] = im2
        im1=im1s
        im2=im2s

    out = cv2.min(im1,im2)
    out = cv2.resize(out, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    out = cv2.normalize(out, None, 0.0, 255.0, cv2.NORM_MINMAX)
    # plt.imshow(out, cmap='gray', interpolation='nearest')
    # plt.show()
    return out

def test_blend(imgname='./test_images/blend.png', label2imgs=None, shift_range=[0,0]):
    if label2imgs is None:
        label2imgs = pickle.load(open('./data/full.gallery.pkl', 'rb'))
    bg = cv2.imread(imgname, cv2.CV_8UC1)
    li=list(label2imgs.items())
    ind=random.randint(0, len(li)-1)
    print('{} vs {}'.format(ind,len(li)))
    for ve in li[ind][1]:
        r1l=list(range(-shift_range[0], shift_range[0], 5))
        random.shuffle(r1l)
        if not r1l:
            r1l=[0]
        r2l = list(range(-shift_range[1], shift_range[1], 5))
        random.shuffle(r2l)
        if not r2l:
            r2l=[0]
        for rh in r1l:
            for rv in r2l:
                o = add_blend_img(bg, ve, shift_hor=rh, shift_ver=rv)
                return o

def collect_noise_bg(root_dir='./test_images/noise_bg2'):
    all_bg_im=[]
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            # print(name)
            if 'png' in name.lower() or 'jpg' in name.lower():
                folder = root.split(os.sep)[-2]
                filename = os.path.join(root, name)
                bg = cv2.imread(filename, cv2.CV_8UC1)
                all_bg_im.append(bg)
    return all_bg_im

def test_many_blend(root_dir='./test_images/noise_bg2', shift_range=[0,0]):
    label2imgs = pickle.load(open('./data/mnist_full.nice.gallery.pkl', 'rb'))
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            print(name)
            if 'png' in name.lower() or 'jpg' in name.lower():
                # folder = root.split(os.sep)[-2]
                filename = os.path.join(root, name)
                print(filename)
                o=test_blend(filename, label2imgs, shift_range)
                plt.imsave('./test_images/blend_out/{}'.format(name), o, cmap='gray')






def rotate_translate_bound(image, angle, x, y, border,img_rows=64,img_cols=64):
    # print(angle,x,y,border)
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    img_rot = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=[255])

    M = np.float32([[1, 0, x], [0, 1, y]])
    im_shift = cv2.warpAffine(img_rot, M,  (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=[255])

    hei=h
    wid=w
    if wid >= hei:
        top, bottom = [int(abs(wid - hei) / 2)] * 2
        left, right = 0, 0
    if wid < hei:
        top, bottom = 0, 0
        left, right = [int(abs(wid - hei) / 2)] * 2


    top, bottom, left, right = [top + border, bottom + border, left + border, right + border]
    img_border = cv2.copyMakeBorder(im_shift, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255])
    img_scale = cv2.resize(img_border, (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    return img_scale



def test_rotate():
    img = cv2.imread('./seed_new_labels/1-\'.png',cv2.CV_8UC1)
    img2 = rotate_translate_bound(img, 10, 5, -6, 5)
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.imshow(img2, cmap='gray')
    plt.show()

def gen_from_seed(seed_dir, name_seed='quote', split=0.8):
    all_imgs=[]
    label2char={}
    char2label = {}
    l=0
    all_labels=[]
    for file in sorted(os.listdir(seed_dir)):
        filename = seed_dir + file
        # print(filename)
        if os.path.isfile(filename) and ('jpg' in filename.lower() or 'png' in filename.lower()): \
                # and ('-' == filename[-5]):
            print(filename)
            img = cv2.imread(filename, cv2.CV_8UC1)

            for r in range(-10,11,10):
                for tx in range(-20,21,10):
                    for ty in range(-20,21,10):
                        for bor in range(0,15,5):
                            # c = filename[-5]
                            c=file[5]
                            if c not in char2label:
                                label2char[l]=c
                                char2label[c]=l
                                l+=1
                            all_labels.append(char2label[c])
                            img2 = rotate_translate_bound(img, r, tx, ty, bor)
                            _,out = rh.load_single_img_nice(img2)
                            #out = cv2.GaussianBlur(img2, (3, 3), 0)
                            #out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                            #                            115, 1)
                            # out = cv2.Canny(img2, 100, 200)
                            #kernel = np.ones((3, 3), np.uint8)
                            #out = 255 - cv2.erode(255 - out, kernel, iterations=1)

                            #out = cv2.normalize(out, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            # print(c)
                            # plt.imshow(out[0], cmap='gray')
                            # plt.show()

                            all_imgs.append(out)
    all_imgs = np.asarray(all_imgs)
    all_labels2=np.zeros((len(all_imgs), l))
    for ind, la in enumerate(all_labels):
        all_labels2[ind][la]=1

    print('---------')
    print(all_imgs.shape)
    print(all_labels2.shape)
    numt=int(all_imgs.shape[0]*split)
    fd=seed_dir+'/'+name_seed+'.pkl'
    pickle.dump((all_imgs[:numt],all_labels2[:numt],all_imgs[numt:], all_labels2[numt:]),
                open(fd,'wb'))
    json.dump(char2label, open(fd[:-4] + '.char2label.json', 'w'))
    json.dump(label2char, open(fd[:-4] + '.label2char.json', 'w'))


if __name__=='__main__':
    # test_blend()
    # gen_from_seed('./seed_new_labels/','only_quote')
    # gen_from_seed('./seed_new_labels/', 'only_hyp')
    gen_from_seed('./test_images/model_mnist_hyp_wrong/add_data/chosen_seed/', 'strange_number')
    # test_many_blend(root_dir='./test_images/model_mnist_hyp_wrong/add_data/noise/', shift_range=[20,20])
