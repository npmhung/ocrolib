# Import the cv2 library
import cv2
# Read the image you want connected components of
dir = '/media/warrior/MULTIMEDIA/Newworkspace/Nissay/output/wordcut/cell_trimming/cellword/03.0002.07/'
filename = dir  + '01.0001.06.png'
src = cv2.imread(filename)
# Threshold it so it becomes binary
ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# You need to choose 4 or 8 for connectivity type
connectivity = 4
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]

print('num lb ', num_labels)
print('labels', labels)
print('stats ', stats)
print('cent ', centroids)