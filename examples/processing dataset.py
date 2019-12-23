import insightface
import cv2
import numpy as np
import time
import imutils
import os
import pickle

def compare_embeddings(emb1t, emb2t):
    emb1 = emb1t.flatten()
    emb2 = emb2t.flatten()
    from numpy.linalg import norm
    sim = np.dot(emb1, emb2)/(norm(emb1)*norm(emb2))
    return sim

#def compare_faces(data_emb, emb):

def find_match(data, stored_data, thresh = 0.2):
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
            print('============ERROR WITH SCALLER 2 ============')

        return resized
    elif(owidth > oheight):
        print('============ERROR WITH SCALLER 1 ============')
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

model = insightface.model_zoo.get_model('retinaface_r50_v1')

model.prepare(ctx_id = -1, nms=0.4)

recognizer = insightface.model_zoo.get_model('arcface_r100_v1')

recognizer.prepare(ctx_id = -1)

path = '/home/pdi/Face_Recognizer/dataset/'

WIDTHDIVIDER = 4


embeddings = []
embeddings = np.zeros( (1,512) )
names = []

for folder in os.listdir(path):
    for img_name in os.listdir(path + folder + '/'):
        img = cv2.imread(path + folder + '/' + img_name)
        img = imutils.resize(img, width=int(1920/WIDTHDIVIDER))

        bboxs, landmarks = model.detect(img, threshold=0.5, scale=1.0)
        faces = []

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

            #print('number', len(bbox))
            
            if(faces):
                for face in faces:
                    if(face is not None):
                        embeddings = np.row_stack(( embeddings,recognizer.get_embedding(face)  ))
                        names.append(folder)
                if(embeddings is not None):
                    indx = 0    
                    for bbox in bboxs:
                        cv2.putText(img, str(folder), (int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        indx += 1
        print('=')
        #cv2.imshow('cc',img)
        #cv2.waitKey(0)


embeddings = np.delete(embeddings , 0, 0)

#np.savetxt('data.txt', embeddings )
#with open('names.txt', 'w') as f:
#    for name in names:
#        f.write("%s\n" % name)
with open('database/'+ 'embeddings.pickle', 'wb') as f:
    pickle.dump(embeddings, f)
with open('database/'+ 'names.pickle', 'wb') as f:
    pickle.dump(names, f)


cv2.destroyAllWindows()