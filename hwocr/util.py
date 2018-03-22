import pickle
import os
import json



def extract_wrong_case(dir_save='./test_images/wrong_cases_img'):
    from shutil import copyfile
    import scipy.misc


    nlabel2imgs = pickle.load(open('./data/mnist_full.gallery.pkl', 'rb'))
    nchar2label = json.load(open('./data/mnist.char2label.json'))
    klabel2imgs = pickle.load(open('./data/full_katakana_quote.gallery.pkl', 'rb'))
    kchar2label = json.load(open('./data/full_katakana_quote.char2label.json'))
    khlabel2imgs = pickle.load(open('./data/kanji_hira.gallery.pkl', 'rb'))
    khchar2label = json.load(open('./data/kanji_hira.char2label.json'))
    label2img={"text":khlabel2imgs,"number":nlabel2imgs,"kata":klabel2imgs}
    char2label = {"text": khchar2label, "number": nchar2label, "kata": kchar2label}

    results=pickle.load(open('results.pkl','rb'))


    acc=0
    for i,r in enumerate(results):
        if r['true_label']=='-':
            continue
        prefix='true'
        if r['true_label']!=r['ocr_result']:
            prefix='false'
        else:
            acc+=1
        fname = dir_save+'/'+prefix
        if not os.path.isdir(fname):
            os.mkdir(fname)
        fname = fname + '/' + r['type']
        if not os.path.isdir(fname):
            os.mkdir(fname)
        cfo = fname + '/' + r['true_label']
        cfn = cfo + '/' + os.path.basename(r['filename'])
        if not os.path.isdir(cfo):
            os.mkdir(cfo)
        copyfile(r['filename'], cfn)

        # print(r['true_label'].encode('unicode-escape'))

        try:
            l = int(char2label['number'][r['true_label']])
            lim = nlabel2imgs
        except:
            try:
                l = int(char2label['kata'][r['true_label']])
                lim = klabel2imgs
            except:
                try:
                    lim = label2img['text']
                    l = int(char2label['text'][r['true_label']])
                except:
                    print('st wrong {} {}'.format(r['true_label'], r['true_label'].encode('unicode-escape')))
                    # continue
        for ii in range(5):
            cfntr = cfn + '.train{}'.format(ii) + cfn[-4:]
            # print(label2img[r['type']][l])
            # print(lim.keys())
            # print(type(l))
            scipy.misc.imsave(cfntr, lim[l][ii])
    print(acc/len(results))
    print(acc)
    print(len(results))