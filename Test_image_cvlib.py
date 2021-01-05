import cv2
import cvlib as cv
from matplotlib import pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

dir_path=str(os.getcwd())
model=load_model(dir_path+"\\mobilenet_v2.model")
image_dir=dir_path+"\\images"
l=len(os.listdir(image_dir))

def mask_detection(image,i):
	Tested_dir=dir_path+"\\Tested images\\"+str(i)+".png"
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
	    color=(0,0,255)
	    
	label="{}: {:.2f}%".format(label,max(Mask,withoutMask)*100)
	for (startX, startY, endX, endY) in faces:
		cv2.rectangle(image, (startX, startY), (endX, endY),color, 3)
		cv2.putText(image,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
	cv2.imwrite(Tested_dir,image)
	cv2.imshow("Image",image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def show_img_with_matplotlib(color_img, title, pos):

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


print(f"Total Images testing {l}")
for i in range(l):
	img_dir=image_dir+"\\"+os.listdir(image_dir)[i]
	image=cv2.imread(img_dir)
	try:
		mask_detection(image,i)
	except:
		continue
	if cv2.waitKey(10) & 0xFF == ord('m'):
		break


plt.show()


