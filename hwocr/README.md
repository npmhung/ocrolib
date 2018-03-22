To setup model for basic prediction task:

put file xxx_weights.h5 and file xxx.json into /ocrolib/hwocr/save

put file xxx.label2char.json into ocrolib/hwocr/data/

To call predict and report function:

Step1: from ocrolib.hwocr import run_hw

Step2: if run report for nissay, call run_hw.report2nissay('path_to_dir_contain_cut_images',
'path_to_xxx.label2char.json','path_to_xxx.json','path_to_xxx_weights.h5')

eg: report2nissay('./test_images/runhocr/',
                  './data/full_nissay.label2char.json',
                  './save/model_nissay.json',
                  './save/weight_nissay.h5')


Step2: if run report for nissay use seperated model, call run_hw.report2nissay_number_sep('path_to_dir_contain_cut_images',
'path_to_xxx.label2char.json','path_to_xxx.json','path_to_xxx_weights.h5',
'path_to_yyy.label2char.json','path_to_yyy.json','path_to_yyy_weights.h5')

yyy is the name of number model

#api seq number model

report2nissay_number_sep('./test_images/outpu_for_OCR_text_number/',
                  './data/full_nissay.label2char.json',
                  './save/model_nissay.json',
                  './save/weight_nissay.h5',
                  './data/full_mnist.label2char.json',
                  './save/model_mnist.json',
                  './save/weight_mnist.h5')

#api sep number+kata model

report2nissay_number_sep_kata_origin('./test_images/sougou20019/',
                                     './data/full_nissay.label2char.json',
                                     './save/model_nissay.json',
                                     './save/weight_nissay.h5',
                                     './data/full_mnist.label2char.json',
                                     './save/model_mnist.json',
                                     './save/weight_mnist.h5',
                                     './data/full_katakana_quote.label2char.json',
                                     "./save/model_katakana_quote.json",
                                     './save/weight_katakana_quote.h5')

#api sep number+kata+kanhi model

report2nissay_number_sep_kata_kanhi_origin('./test_images/sougou20019/',
                                         './data/full_mnist.label2char.json',
                                         './save/model_mnist.json',
                                         './save/weight_mnist.h5',
                                         './data/full_katakana_quote.label2char.json',
                                         "./save/model_katakana_quote.json",
                                         './save/weight_katakana_quote.h5'
                                         './data/kanji_higa.label2char.json',
                                         "./save/model_kanji_higa.json",
                                          './save/weight_kanji_higa.h5'
                                               )