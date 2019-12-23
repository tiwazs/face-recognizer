import insightface
import cv2
import numpy as np
import time
import imutils
from ast import literal_eval
import pickle

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def compare_embeddings(emb1t, emb2t):
    emb1 = emb1t.flatten()
    emb2 = emb2t.flatten()
    from numpy.linalg import norm
    sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return sim

#def compare_faces(data_emb, emb):

def find_match(data, stored_data, thresh = 0.5):
    ldata = np.linalg.norm(data, axis=1)[None].T
    lstored = np.linalg.norm(stored_data, axis=1)[None].T
    num = np.dot(data, stored_data.T)
    den = np.dot(ldata, lstored.T)
    similarity = num/den
    thresh_vec = np.zeros( (similarity.shape[0],1) ) + thresh
    similarity = np.column_stack(( thresh_vec,similarity ))
    matches = np.argmax(similarity, axis = 1)
    return matches



def scaller(img):
    oheight,owidth,_ = img.shape
    if(oheight >= owidth):
        width = 112
        p = (width/owidth)
        height = oheight*p
        dim = (int(width), int(height) )
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        res = height - 112

        resized = resized[ int(res/2):(int(res/2) + 112), :,:]
        if(resized.shape != (112,112,3) ):
            print('============ERROR WITH SCALLER 1 ============')

        return resized
    elif(owidth > oheight):
        #print('============ERROR WITH SCALLER 1 ============')
        #print(oheight)
        height = 112
        p = (height/oheight)
        width = owidth*p
        dim = (int(width), int(height) )
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        res = width - 112

        resized = resized[ :, int(res/2):(int(res/2) + 112),:]
        if(resized.shape != (112,112,3) ):
            print('============ERROR WITH SCALLER 2 ============')

        return resized
    else:
        return
    
    

INPUTVIDEO = '/home/pdi/Face_Recognizer/examples/test3.mp4'
SKIPFRAMES = 5
WIDTHDIVIDER = 5
SKIP_VIDEO = 0 - 2


#data_saved = np.loadtxt('data.txt')
with open('database/embeddings.pickle', 'rb') as f:
    data_saved = pickle.load(f)
    if(data_saved is not None):
        print('input loaded')
    else:
        print('problem with saved embeddings')
with open('database/names.pickle', 'rb') as f:
    names = pickle.load(f)
    if(names is not None):
        print('output loaded')
    else:
        print('problem with output')
names = ['Uknown'] + names

#names.append('Uknown')
#with open('names.txt') as fi:
#    [names.append( line.split('\n')[0]) for line in fi]

print(names)
model = insightface.model_zoo.get_model('retinaface_r50_v1')

model.prepare(ctx_id = 0, nms=0.4)

recognizer = insightface.model_zoo.get_model('arcface_r100_v1')

recognizer.prepare(ctx_id = 0)

cap = cv2.VideoCapture(0)


skipped = 0
timei = 0
timef = 0
time2f = 0
time2i = 0
frame = 0
while cap.isOpened():

    ret, img = cap.read()
    frame += 1 
    if(ret != True):
        break
    img = imutils.resize(img, width=int(1920/WIDTHDIVIDER))

    #img = adjust_gamma(img, gamma = 1.5)        #gamma correction?
    if(skipped <= SKIP_VIDEO):
        skipped += 1
        continue

    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print(frame)
    timei = time.time()
    bboxs, landmarks = model.detect(img, threshold=0.5, scale=1.0)
    timef = time.time()

    faces = []
    embeddings = []

    
    if( bboxs is not None):
        todel = []
        for i in range(bboxs.shape[0]):
            if(any(x<0 for x in bboxs[i])):
                todel.append(i)
        for i in todel:
            bboxs = np.delete(bboxs, i, 0)

        for bbox, landmark in zip(bboxs, landmarks):
            #if(not any(x<0 for x in bbox)):
            #print('Bbox',(bbox[0],bbox[1]), (bbox[2],bbox[3]))
            face = scaller( img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]),:] )

            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 255, 0), 1)
            #print(face.shape)
            faces.append( face )

        embeddings = np.zeros( (1,512) )
        #print('number', len(bbox))
        
        if(faces):
            time2i = time.time()
            for face in faces:
                if(face is not None):
                    embeddings = np.row_stack(( embeddings,recognizer.get_embedding(face)  ))
            embeddings = np.delete(embeddings, 0, 0)
            time2f = time.time()

            if(embeddings is not None):
                matches = find_match(embeddings,data_saved, 0.7)
                indx = 0    
                for bbox in bboxs:
                    cv2.putText(img, names[matches[indx]], (int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    indx += 1

    #print(embeddings.shape)

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



cap.release()
cv2.destroyAllWindows()