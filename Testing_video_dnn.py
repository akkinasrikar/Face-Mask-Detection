from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os

dir_path=str(os.getcwd())
prototxt_path=dir_path+"\\deploy.prototxt"
caffe_weights_path=dir_path+"\\res10_300x300_ssd_iter_140000.caffemodel"


net=cv2.dnn.readNet(prototxt_path,caffe_weights_path)
model=load_model(dir_path+"\\mobilenet_v2.model")

def predict_mask(image,facenet,masknet):
    (h,w)=image.shape[:2]
    blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))
    facenet.setInput(blob)
    detections=facenet.forward()
    
    faces=[]
    locs=[]
    preds=[]
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]

        if confidence>0.5:
            Box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (start_X,start_Y,end_X,end_Y)=Box.astype('int')
            (start_X,start_Y)=(max(0,start_X),max(0,start_Y))
            (end_X,end_Y)=(min(w-1,end_X), min(h-1,end_Y))

            face=image[start_Y:end_Y, start_X:end_X]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            faces.append(face)
            locs.append((start_X,start_Y,end_X,end_Y))
            
    if len(faces)>0:
        faces=np.array(faces,dtype='float32')
        preds=masknet.predict(faces,batch_size=1)
        
    return (locs,preds)

vs=VideoStream(src=0).start()

while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=1080)
    
    (locs,preds)=predict_mask(frame,net,model)
    
    for (box,pred) in zip(locs,preds):
        (start_X,start_Y,end_X,end_Y)=box
        (Mask,withoutMask)=pred
        
        if Mask>withoutMask:
            label='Mask'
        else:
            label='No Mask'
            
        if label=="Mask":
            color=(0,255,0)
        else:
            color=(0,0,255)

        cv2.putText(frame,label,(start_X,start_Y-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        
        cv2.rectangle(frame,(start_X,start_Y),(end_X,end_Y),color,2)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF
    
    if key==ord('q'):
        break
        
cv2.destroyAllWindows()
vs.stop()
        
        
    