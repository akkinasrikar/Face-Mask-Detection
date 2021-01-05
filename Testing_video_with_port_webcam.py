from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

dir_path=str(os.getcwd())
prototxt_path=dir_path+"\\deploy.prototxt"
caffe_weights_path=dir_path+"\\res10_300x300_ssd_iter_140000.caffemodel"
Tested_dir=dir_path+"\\Tested images"

net=cv2.dnn.readNet(prototxt_path,caffe_weights_path)
md=dir_path+"\\mobilenet_v2.model"
model=load_model(dir_path+"\\mobilenet_v2.model")
image_dir=dir_path+"\\images"
l=len(os.listdir(image_dir))
img_dir=image_dir+"\\"+os.listdir(image_dir)[np.random.randint(l)]

def predict(img):
    image=img
    (h,w)=image.shape[:2]
    blob=cv2.dnn.blobFromImage(image,1.0,(300,300),(104.0,177.0,123.0))

    net.setInput(blob)
    detections=net.forward()

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
            face=np.expand_dims(face,axis=0)
            
            (Mask,withoutMask)=model.predict(face)[0]
            
            if Mask>withoutMask:
                label='Mask'
            else:
                label='No Mask'
                
            if label=="Mask":
                color=(0,255,0)
            else:
                color=(0,0,255)
        
            label="{}: {:.2f}%".format(label,max(Mask,withoutMask)*100)
            
            cv2.putText(image,label,(start_X,start_Y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,3)
            cv2.rectangle(image,(start_X,start_Y),(end_X,end_Y),color,2)
    return image


url="http://192.168.1.3:8080/video"
cap=cv2.VideoCapture(url)
while(True):
    ret,frame=cap.read()
    if frame is not None:
        try:
            mg=predict(frame)
            cv2.imshow("frame",mg)
        except:
            continue
    q=cv2.waitKey(1)
    if q==ord('q'):
        break
cv2.destroyAllWindows()






