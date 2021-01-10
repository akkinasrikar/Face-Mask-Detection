import streamlit as st 
st.title("Mask Prediction Web App")

import cv2
import cvlib as cv
from matplotlib import pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import cv2
import time


dir_path=str(os.getcwd())
model=load_model(dir_path+"\\mobilenet_v2.model")


def mask_detection(image):
	faces, confidences = cv.detect_face(image)
	img=cv2.resize(image,(224,224))
	img=img_to_array(img)
	img=preprocess_input(img)
	img=np.expand_dims(img,axis=0)
	(Mask,withoutMask)=model.predict(img)[0]

	if Mask>withoutMask:
	    label='Mask'
	else:
	    label='No Mask'
	            
	if label=="Mask":
	    color=(0,255,0)
	else:
	    color=(255,0,0)
	    
	label="{}: {:.2f}%".format(label,max(Mask,withoutMask)*100)
	for (startX, startY, endX, endY) in faces:
		cv2.rectangle(image, (startX, startY), (endX, endY),color, 3)
		cv2.putText(image,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
	return image




uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	pil_image = Image.open(uploaded_file).convert('RGB') 
	open_cv_image = np.array(pil_image)
	pred=mask_detection(open_cv_image)
	st.write("Predicting....")
	my_bar = st.progress(0)
	for percent_complete in range(100):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
	st.image(pred, caption='Uploaded Image.', use_column_width=True)
	st.write("")

