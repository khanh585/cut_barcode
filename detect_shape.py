import cv2
import numpy as np
from utils import stackImages

wiImg = 640
heImg = 480

cap = cv2.VideoCapture(0)
cap.set(3, wiImg)
cap.set(4, heImg)
cap.set(10, 150)

im_contours = None


def getContours(img):
    padding = 7
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(im_contours, cnt, -1, (10, 20,10), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            obj_cor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            obj_type = ''
            if obj_cor == 3:
                obj_type = "Tri"
            elif obj_cor == 4:
                asp_ratio = w / float(h)
                print(asp_ratio)
                if asp_ratio > 0.95 and asp_ratio < 1.05:
                    obj_type = "Square"
                else:
                    obj_type = "Rect"
                print(obj_type)
            else:
                obj_type = "Cir"
            
            cv2.rectangle(im_contours, (x - padding, y - padding), (x + w + padding, y + h + padding), (100, 10, 100), 2)
            cv2.putText(im_contours, obj_type, (x + w // 2 , y + h // 2 ), cv2.FONT_HERSHEY_COMPLEX, 0.7, (70, 0, 50), 2)



def preProcess(im):
    im_blank = np.zeros_like(im)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray,(7,7), 1)
    im_canny = cv2.Canny(im_blur,90,90)
    cv2.imshow("stack1", im_canny)


    getContours(im_canny)

  


while True:
    success, img = cap.read()

    if(success):
        img = cv2.resize(img, (wiImg, heImg))
        im_blank = np.zeros_like(img)
        im_contours = img.copy()
        preProcess(img)
        cv2.imshow("stack", im_contours)
        cv2.waitKey(1)

    else:
        print('break')
        break



