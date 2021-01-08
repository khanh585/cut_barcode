import cv2
import numpy as np
from utils import stackImages
from utils import resize


wiImg = 640
heImg = 480




imGray = None
imBlur = None
imCanny = None

imDial =None
imThres = None
imContours = None


def preProcess(img):
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imBlur = cv2.GaussianBlur(imGray,(5,5),1)
    imCanny = cv2.Canny(imBlur, 95,95)
    cv2.imshow("imCanny", imCanny)
    kernel = np.ones((7,7))
    imDial = cv2.dilate(imCanny,kernel, iterations=2)
    kernel = np.ones((12,12))
    imThres = cv2.erode(imDial,kernel, iterations=1)
    cv2.imshow("imThres", imThres)

    return imThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            cv2.drawContours(imContours, cnt, -1, (200, 20,10), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,False)
            
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest
        
def readVideo():
    cap = cv2.VideoCapture(0)
    cap.set(3, wiImg)
    cap.set(4, heImg)
    cap.set(10, 150)
    while True:
        success, img = cap.read()
        if(success):
            img = cv2.resize(img, (wiImg, heImg))
            im_blank = np.zeros_like(img)
            preProcess(img)
            imContours = img.copy()
            getContours(imThres)
            stack =  stackImages(1, ([img]))
            cv2.imshow("stack", stack)
            cv2.waitKey(1)
        else:
            break
    
def readImage():
    img = cv2.imread('frame/ff206.png')
    img = resize(img,1500)
    im_blank = np.zeros_like(img)
    preProcess(img)
    imContours = img.copy()
    getContours(imThres)
    stack =  imContours
    cv2.imshow("stack", stack)
    cv2.waitKey(20*1000)

readImage()



