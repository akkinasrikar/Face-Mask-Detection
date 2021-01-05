import cv2
import numpy as np
url="http://192.168.1.3:8080/video"
#url="http://[2409:4030:4295:405b::1e1:b8ac]:8080"
cap=cv2.VideoCapture(url)
while(True):
    ret,frame=cap.read()
    if frame is not None:
        cv2.imshow("frame",frame)
    q=cv2.waitKey(1)
    if q==ord('q'):
        break
cv2.destroyAllWindows()