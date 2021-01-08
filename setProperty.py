import cv2
import numpy as np
from utils import resize, findHeAndWi


def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
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

path = 'frame/ff206.png'
rate = 0
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("size","TrackBars",800,5000,empty)
cv2.createTrackbar("kernelBlur","TrackBars",37,51,empty)
cv2.createTrackbar("cannyValue","TrackBars",32,51,empty)
cv2.createTrackbar("kernelDial","TrackBars",3,51,empty)
cv2.createTrackbar("kernelThres","TrackBars",3,51,empty)

def preProcess(img, kernelBlur, cannyValue, kernelDial, kernelThres):
    try: 
        imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imBlur = cv2.GaussianBlur(imGray,kernelBlur,1)
        imCanny = cv2.Canny(imBlur, cannyValue,cannyValue)
        kernel = np.ones(kernelDial)
        imDial = cv2.dilate(imCanny,kernel, iterations=2)
        kernel = np.ones(kernelThres)
        imThres = cv2.erode(imDial,kernel, iterations=1)
        return imBlur, imCanny, imDial ,imThres
    except:
        return img, img, img ,img

def setProperty():
    cap = cv2.VideoCapture('video/Demo.mp4')
    index = 0
    while True:
        # img = cv2.imread(path)
        index += 1
        success, img = cap.read()
        imgReal = img.copy()
        size = cv2.getTrackbarPos("size", "TrackBars")
        img, rate = resize(img, size)
        blur = cv2.getTrackbarPos("kernelBlur", "TrackBars")
        canny = cv2.getTrackbarPos("cannyValue", "TrackBars")
        dial = cv2.getTrackbarPos("kernelDial", "TrackBars")
        thres = cv2.getTrackbarPos("kernelThres", "TrackBars")

        imgBlur, imgCanny, imgDial, imgThres = preProcess(img = img, kernelBlur=(blur,blur), cannyValue=canny, kernelDial=(dial,dial), kernelThres=(thres,thres))
        # print(blur, canny, dial, thres)

        biggest, imgContour, area, peri = getContours(img, imgThres)
        he, wi = findHeAndWi(area = area, peri = peri)
        img_Warp = getWarp(imgReal, biggest, wiImg=wi, heImg=he, rate = rate, index = index)

        #show result
        imgStack = stackImages(0.6,([imgContour, imgBlur, imgCanny], [imgDial, imgThres, img_Warp]))
        cv2.imshow("Stacked Images", imgStack)
        cv2.waitKey(1)

def getContours(img, imgThres):
    try:
        padding = 7
        biggest = np.array([])
        maxArea = 0
        contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        im_contours = img.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 7000 and area <15000:
                cv2.drawContours(im_contours, cnt, -1, (10, 20,10), 3)
                peri = cv2.arcLength(cnt, True) 
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                x,y,w,h = cv2.boundingRect(approx)
                obj_cor = len(approx)
                if(obj_cor == 4):
                    cv2.putText(im_contours, '%s'%area, (x , y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (70, 0, 50), 2)
                    cv2.rectangle(im_contours, (x - padding, y - padding), (x + w + padding, y + h + padding), (100, 10, 100), 2)
                    if area > maxArea and len(approx) == 4:
                        biggest = approx
                        maxArea = area
        return biggest, im_contours, area, peri
    except Exception as e:
        print(e)
        return biggest, img

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

def getWarp(img, biggest, wiImg, heImg, rate, index):
    rate = 1/ rate
    wiImg = int(wiImg * rate)
    heImg = int(heImg * rate)
    biggest = biggest * rate
    try:
        if len(biggest) != 0:
            biggest = reOrder(biggest)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0,0],[wiImg,0], [0,heImg], [wiImg, heImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgOutput = cv2.warpPerspective(img, matrix, (wiImg, heImg))

            imgCropped = imgOutput[20:imgOutput.shape[0]+20, 20: imgOutput.shape[1]+20]
            imgCropped = cv2.resize(imgCropped, (wiImg, heImg))
            
            # path = 'frame_bar/ffb%s.png'%index
            # cv2.imwrite(path, imgCropped)

            return imgCropped
        else:
            print('else')
            return img
    except Exception as e:
        print(e)
        return img


setProperty()