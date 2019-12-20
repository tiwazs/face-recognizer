import insightface
import urllib
import urllib.request
import cv2
import numpy as np

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

#url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
url = "https://ichef.bbci.co.uk/news/660/cpsprodpb/371C/production/_107980141_gettyimages-842090336.jpg"
img = url_to_image(url)

model = insightface.model_zoo.get_model('retinaface_r50_v1')

model.prepare(ctx_id = -1, nms=0.4)

bboxs, landmarks = model.detect(img, threshold=0.5, scale=1.0)

for bbox, landmark in zip(bboxs, landmarks):
    #print('Bbox',(bbox[0],bbox[1]), (bbox[2],bbox[3]))
    cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 255, 0), 2)
#print('number', len(bbox))

cv2.imshow('Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()