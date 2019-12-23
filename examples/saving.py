import insightface
import cv2
import numpy as np
import time
import imutils

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
    
    

INPUTVIDEO = '/home/pdi/Face_Recognizer/examples/pulp_fiction.mp4'
SKIPFRAMES = 5
WIDTHDIVIDER = 5
SKIP_VIDEO = 0 - 2
data_saved = np.loadtxt('data.txt')

model = insightface.model_zoo.get_model('retinaface_r50_v1')

model.prepare(ctx_id = -1, nms=0.4)

recognizer = insightface.model_zoo.get_model('arcface_r100_v1')

recognizer.prepare(ctx_id = -1)

cap = cv2.VideoCapture(INPUTVIDEO)
skipped = 0
time2f = 0
time2i = 0
frame = 0

embs_to_save = np.zeros( (1,512) )
names_to_save = []
en = 0
sk = 0
while cap.isOpened():
    
    fin = cv2.waitKey(1) & 0xFF

    ret, img = cap.read()
    frame += 1
    
    if(ret != True):
        break

    if(en == 1):
        sk += 1
        if(sk == 50):
            sk = 0
            en = 0
        continue
    img = imutils.resize(img, width=int(1920/WIDTHDIVIDER))
    
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
                indx = 0    
                for bbox in bboxs:
                    cv2.putText(img, str(indx), (int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    indx += 1

    #print(embeddings.shape)

    cv2.namedWindow('Frame')
    cv2.imshow('Frame', img)
    #print(time2f-time2i)

    fin = cv2.waitKey(1) & 0xFF
    if(fin == ord('s')):
        print('Wchich face to save')
        indx_s = int(input())

        embs_to_save = np.row_stack(( embs_to_save, embeddings[indx_s] ))
        
        print('Input Name')
        name =input()
        names_to_save.append(name)
        print(name)

        #data_saved = np.loadtxt('data.txt')

    if(fin == ord('q')):
        #Save embs
        #np.savetxt('data.txt', embeddings )
        embs_to_save = np.delete(embs_to_save , 0, 0)
        print(names_to_save)
        np.savetxt('data.txt', embs_to_save )

        with open('names.txt', 'w') as f:
            for name in names_to_save:
                f.write("%s\n" % name)

        break
    if(fin == ord('k') ):
        en = 1



cap.release()
cv2.destroyAllWindows()