import cv2
import imutils
import argparse
import numpy as np
from processing_dataset import *


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name to register")
args = vars(ap.parse_args())

WIDTHDIVIDER = 1
cap = cv2.VideoCapture(0)

img_count = 0
apply_gamma = False

os.mkdir("dataset/"+ args['name'])
while cap.isOpened():
    ret, img = cap.read()
    if(ret != True):
        break
    img = imutils.resize(img, width=int(img.shape[1]/WIDTHDIVIDER))
    if(apply_gamma):
        #print('Gamma')
        img = adjust_gamma(img, gamma = 1.5)        #gamma correction?


    cv2.namedWindow('Frame')
    cv2.imshow('Frame', img)
    #print('=Det:',timef-timei)
    #print('=Rec:',time2f-time2i)

    fin = cv2.waitKey(1) & 0xFF
    if(fin == ord('q')):
        #Save embs
        #np.savetxt('data.txt', embeddings ) 
        #data_saved = np.loadtxt('data.txt')

        break
    if(fin == ord('s')):
        #Save embs
        #np.savetxt('data.txt', embeddings ) 
        #data_saved = np.loadtxt('data.txt')
        fname = "dataset/"+ args['name'] +"/" + args['name'] + str(img_count) + ".jpg"
        cv2.imwrite( fname, img )
        img_count += 1
        print('IMAGE SAVED', fname)
        continue
    if(fin == ord('g')):
        apply_gamma = not apply_gamma
        print(apply_gamma)
        continue



cap.release()
cv2.destroyAllWindows()
processingDataset("dataset/", WIDTHDIVIDER)


