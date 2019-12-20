import insightface
import urllib
import urllib.request
import cv2
import numpy as np

def scaller(img):
    oheight,owidth,_ = img.shape
    if(oheight > owidth):
        width = 112
        p = (width/owidth)
    else:
        print('============ERROR WITH SCALLER 1 ============')
    height = oheight*p
    dim = (int(width), int(height) )
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    res = height - 112

    resized = resized[ int(res/2):(int(res/2) + 112), :,:]
    if(resized.shape != (112,112,3) ):
        print('============ERROR WITH SCALLER 2 ============')

    return resized

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

url = 'https://github.com/deepinsight/insightface/raw/master/deploy/Tom_Hanks_54745.png'
url2 = 'http://es.web.img2.acsta.net/pictures/16/04/26/10/00/472541.jpg'
url3 =  'https://www.mujerhoy.com/noticias/201902/25/media/cortadas/amy-adams-retoques-cara-kMPC-U707576199665ME-644x483@MujerHoy.jpg'

img = url_to_image(url)
img2 = url_to_image(url2)
img3 = url_to_image(url3)

model = insightface.model_zoo.get_model('arcface_r100_v1')

model.prepare(ctx_id = -1)

#img3 = scaller(img3)
#print(img2.shape)

cv2.imshow('img',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
#emb = model.get_embedding(img3)
#emb2 = model.get_embedding(img3)

#print(emb)