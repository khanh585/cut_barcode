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
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            obj_cor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            if(obj_cor == 4):
                cv2.drawContours(im_contours, cnt, -1, (200, 20,10), 2)
                cv2.rectangle(im_contours, (x - padding, y - padding), (x + w + padding, y + h + padding), (100, 10, 100), 2)
                if area > maxArea and len(approx) == 4:
                    biggest = approx
                    maxArea = area
    return biggest


def preProcess(im):
    im_blank = np.zeros_like(im)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray,(7,7), 1)
    im_canny = cv2.Canny(im_blur,90,90)

    kernel = np.ones((5,5))
    imDial = cv2.dilate(im_canny,kernel, iterations=2)
    kernel = np.ones((7,7))
    
    imThres = cv2.erode(imDial,kernel, iterations=1)
    cv2.imshow("imThres", imThres)

    return imThres

def reOrder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myNewPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)

    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]

    return myNewPoints



def getWarp(img, biggest):
    if len(biggest) != 0:
        biggest = reOrder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[wiImg,0], [0,heImg], [wiImg, heImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (wiImg, heImg))

        imgCropped = imgOutput[20:imgOutput.shape[0]+20, 20: imgOutput.shape[1]+20]
        imgCropped = cv2.resize(imgCropped, (wiImg, heImg))

        return imgCropped
    else: 
        return img

  


while True:
    success, img = cap.read()

    if(success):
        img = cv2.resize(img, (wiImg, heImg))
        im_blank = np.zeros_like(img)
        im_contours = img.copy()
        imThres = preProcess(img)
        biggest =  getContours(imThres)
        img_Warp = getWarp(img, biggest)
        cv2.imshow("stack", img_Warp)
        cv2.waitKey(1)

    else:
        print('break')
        break



