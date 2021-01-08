import cv2
import numpy as np
import math  

def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    # width = imgArray[0][0].shape[1]
    # height = imgArray[0][0].shape[0]
    width = 640
    height = 480
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



def getContours(img, raw):
    padding = 7
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    im_contours = raw.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(im_contours, cnt, -1, (10, 20,10), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            
            cv2.rectangle(im_contours, (x - padding, y - padding), (x + w + padding, y + h + padding), (100, 10, 100), 2)
            

def resize(img, wi):
    # [0] == he , [1] == wi
    he = int(wi * img.shape[0] / img.shape[1])
    rate = he / img.shape[0]
    return cv2.resize(img,(wi,he)), rate


def findHeAndWi(area, peri):
    # calculate the discriminant
    a = 2
    b = -peri
    c = 2* area

    d = (b**2) - (4*a*c)
    dis = b * b - 4 * a * c  
    sqrt_val = math.sqrt(abs(dis)) 

    # find two solutions
    x1 = (-b + sqrt_val)/(2 * a)
    x2 = (-b - sqrt_val)/(2 * a)

    wi = x1 
    he = area / wi

    print(area, peri, he, wi)

    return he, wi

findHeAndWi(area=15, peri=16)
