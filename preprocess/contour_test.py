import cv2
from ocrolib.wordcut.extras import findContour

dir = '/media/warrior/MULTIMEDIA/Newworkspace' \
      '/Nissay/output/run_form_cut_4_template/03/sougou20004/3/'
filename = dir  + 'line_1_1.png'

img = cv2.imread(filename)
findContour(img)
