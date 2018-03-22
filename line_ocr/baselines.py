
import os
os.environ['TESSDATA_PREFIX'] = '/usr/local/share/'
import sys
PATH_CUR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH_CUR)
sys.path.append('../../')

import numpy as np
from c_metrics import  *
import time
import shutil
import ocrolib.hwocr.run_hw2 as run_hw
from keras.models import model_from_json
import cv2
import json

from PIL import Image
import pytesseract
import unicodedata
import ocrolib.line_ocr.gen_lineocr as gen_lineocr

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def run_tesseract(image):
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    cv2_im = gen_lineocr.read_img_from_file_nice(image)
    im = Image.fromarray(np.reshape(cv2_im, [cv2_im.shape[0],cv2_im.shape[1]]))
    ocr_value = pytesseract.image_to_string(im, lang='jpn', config='--psm 6')
    ocr_value = unicodedata.normalize("NFKC", ocr_value)
    if " " in ocr_value:
        ocr_value = ocr_value.replace(" ", "")
    ocr_value = ocr_value.strip()
    return ocr_value


def predict_hw_single(model, label2chars, dir_img='./tmp/hw_single_images/'):
    print('==================PREDICT SINGLE===================')

    new_images, im_names = run_hw.load_image(dir_img)
    final_str=''
    for i in range(new_images.shape[0]):
        bimg=np.asarray([new_images[i]])
        result_predict = model.predict_classes(bimg, batch_size=1)
        res=label2chars[result_predict[0]]
        final_str+=res
    return final_str


def predict_hw_full(dir_img, model, label2chars, tmp_dir='./tmp/hw_single_images/'):
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    cut_cell_region.regionCutCell(dir_img, os.path.basename(dir_img), tmp_dir)
    return predict_hw_single(model, label2chars, tmp_dir)


def predict_cut_ocr(dir_test='./data/japanese_name/test_data/',
                 tmp_dir='./tmp/hw_single_images/',
                 limit_num=10):
    json_file = open("../hworc/save/model7_1.json", 'r')
    label2chars = json.load(open('../hworc/data/full.label2char.json'))
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    run_hw.load_model_weights('../hworc/save/M7_1-all_weights.h5', model)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    abs_acc_score=[]
    set_acc_score=[]
    ned=[]
    c=0
    for file in sorted(os.listdir(dir_test)):
        filename = dir_test + file
        if c>limit_num:
            break
        c+=1
        if os.path.isfile(filename):
            stri=predict_hw_full(file,model, label2chars, tmp_dir)
            file = file.split('.')[0]
            rstri = file.split('_')[0]
            abs_acc_score.append(abs_acc(stri, rstri))
            set_acc_score.append(set_acc(stri, rstri))
            ned.append(nedit_dist(stri, rstri))

    print('metric abs acc {} set acc {} nedit {}'.format(
        np.mean(abs_acc_score), np.mean(set_acc_score), np.mean(ned)))

def predict_ocr_tesseract(dir_img, tmp_dir='./tmp/'):
    img = cv2.imread(dir_img, cv2.CV_8UC1)
    out = cv2.blur(img, (3, 3))
    out = cv2.adaptiveThreshold(out, 1.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    out = cv2.medianBlur(out, 5)
    # plt.imshow(out)
    # plt.show()
    tmp_dirimg=tmp_dir+'./timg.png'
    cv2.imwrite(tmp_dirimg, np.asarray(out*256, np.int32))
    os.system("tesseract {} {}/out -l jpn --oem 3 --psm 6".format(tmp_dirimg,tmp_dir))
    with open('{}/out.txt'.format(tmp_dir),'r') as f:
        str=''
        for l in f:
            str+=l
        # print(str)
        return str

def predict_ocr_tesseract_all(dir_test='./data/japanese_name/real_data/name_packaged/',
                              tmp_dir='./tmp/', outf='./data/test_result_tess.txt',
                              limit_num=1000):
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    abs_acc_score=[]
    set_acc_score=[]
    ned=[]
    c=0
    fo=open(outf,'w')
    for file in sorted(os.listdir(dir_test)):
        filename = dir_test + file
        if c>limit_num:
            break
        c+=1
        if os.path.isfile(filename):
            # stri=predict_ocr_tesseract(filename, tmp_dir)
            stri=run_tesseract(filename)
            file = file.split('.')[0]
            rstri=file.split('_')[0]
            abs_acc_score.append(abs_acc(stri, rstri))
            set_acc_score.append(set_acc(stri, rstri))
            ned.append(nedit_dist(stri, rstri))
            fo.write('{} vs {}\n'.format(stri, rstri))

    fo.close()

    print('metric abs acc {} set acc {} nedit {}'.format(
        np.mean(abs_acc_score), np.mean(set_acc_score), np.mean(ned)))


from base64 import b64encode
from os import makedirs
from os.path import join, basename
from sys import argv
import json
import requests

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
RESULTS_DIR = './data/google_res'
makedirs(RESULTS_DIR, exist_ok=True)

def make_image_data_list(image_filenames):
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    img_requests = []
    for imgname in image_filenames:
        with open(imgname, 'rb') as f:
            ctxt = b64encode(f.read()).decode()
            img_requests.append({
                'image': {'content': ctxt},
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 1
                }]
            })
    return img_requests

def make_image_data(image_filenames):
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(image_filenames)
    return json.dumps({"requests": imgdict}).encode()

def request_ocr(api_key, image_filenames):
    response = requests.post(ENDPOINT_URL,
                             data=make_image_data(image_filenames),
                             params={'key': api_key},
                             headers={'Content-Type': 'application/json'})
    return response

def predict_google_api(dir_test='./data/japanese_name/real_data/name_packaged/',
                        outf='./data/test_result_google_api.txt',
                       limit_num=2000):
    api_key = "AIzaSyBEplqNVzDK1zWZFmkkUopN_HjNKC3chd0"
    abs_acc_score = []
    set_acc_score = []
    ned = []
    fo = open(outf,'w')
    c=0
    files=os.listdir(dir_test)
    for file in sorted(files):
        filename = dir_test + file
        if c > limit_num:
            break
        c += 1
        if os.path.isfile(filename):
            # print(filename)
            response = request_ocr(api_key, [filename])
            if response.status_code != 200 or response.json().get('error'):
                print(response.text)
            else:
                for idx, resp in enumerate(response.json()['responses']):
                    # save to JSON file
                    # imgname = image_filenames[idx]
                    # jpath = join(RESULTS_DIR, basename(imgname) + '.json')
                    # with open(jpath, 'w') as f:
                    #     datatxt = json.dumps(resp, indent=2)
                    #     print("Wrote", len(datatxt), "bytes to", jpath)
                    #     f.write(datatxt)
                    #
                    # print the plaintext to screen for convenience
                    llprint("\r-------------------{}/{}--------------------------".format(c,len(files)))
                    pre=''
                    if resp:
                        t = resp['textAnnotations'][0]
                        # print("    Bounding Polygon:")
                        # print(t['boundingPoly'])
                        # print("    Text:")
                        pre=t['description']
                        # print(pre)
                    else:
                        print('fail!!!')
                    # print(file)
                    file = file.split('.')[0]
                    rstri = file.split('_')[0]
                    abs_acc_score.append(abs_acc(pre, rstri))
                    set_acc_score.append(set_acc(pre, rstri))
                    ned.append(nedit_dist(pre, rstri))
                    fo.write('{} vs {}\n'.format(pre, rstri))
    fo.close()
    print('\nmetric abs acc {} set acc {} nedit {}'.format(
        np.mean(abs_acc_score), np.mean(set_acc_score), np.mean(ned)))


def test_tesseract():
    start_time = time.time()
    predict_ocr_tesseract_all()
    print("--- %s seconds ---" % (time.time() - start_time))

def test_cut_ocr():
    start_time = time.time()
    predict_cut_ocr()
    print("--- %s seconds ---" % (time.time() - start_time))

def test_google_api():
    start_time = time.time()
    predict_google_api()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # test_tesseract()
    test_google_api()
    # test_cut_ocr()
