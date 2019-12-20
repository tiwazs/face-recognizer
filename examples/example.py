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

url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
img = url_to_image(url)

model = insightface.app.FaceAnalysis()

ctx_id = -1

model.prepare(ctx_id = ctx_id, nms=0.4)

faces = model.get(img)
for idx, face in enumerate(faces):
    print("Face [%d]:"%idx)
    print("\tage:%d"%(face.age))
    gender = 'Male'
    if face.gender==0:
        gender = 'Female'
    print("\tgender:%s"%(gender))
    print("\tembedding shape:%s"%face.embedding.shape)
    print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
    print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
    print("")