import cv2
import numpy as np
from utils import resize, findHeAndWi, getCoordinatesRect


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
    rate = 1 / rate
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

def setPropertyBarCode(img, img4k, x,y):
    imgReal = img4k.copy()
    size = 1500
    blur = 5
    canny = 65
    dial = 7
    padding = 15
    thres = 24
    imgReal, rate = resize(imgReal, size)


    imgBlur, imgCanny, imgDial, imgThres = preProcess(img = img, kernelBlur=(blur,blur), cannyValue=canny, kernelDial=(dial,dial), kernelThres=(thres,thres))

    imgContour, listBarCode = getContoursBarCode(img, imgThres, imgReal, padding, x,y, rate)


    # if listBarCode:
    #     i = 0
    #     for barcode in listBarCode:
    #         i+=1
    #         img_Warp = getWarp(imgReal, barcode['coordinates'], wiImg=barcode['wi'], heImg=barcode['he'], rate = rate, index = 0)
    #         # cv2.imshow("listImg%s"%i, img_Warp)
    # imgStack = stackImages(0.7,([imgContour]))
    

    cv2.imshow("Stacked Images 321", imgContour)
    cv2.waitKey(1)

def getContoursBarCode(img, imgThres, img4k, padding, rx,ry, rate):
    try:
        padding = padding
        coordinates = np.array([])
        contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        im_contours = img4k.copy()
        listBarCode = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                peri = cv2.arcLength(cnt, True) 
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                x,y,w,h = cv2.boundingRect(approx)
                
                obj_cor = len(approx)
                if(obj_cor >= 4):
                    x = int(((x + rx) * rate) ) 
                    y = int(((y + ry) * rate) ) 
                    w = int(w * rate)
                    h = int(h * rate)
                    cv2.rectangle(im_contours, (x , y), (x + w , y + h ), (100, 10, 100), 2)

                    # listBarCode.append({'coordinates': getCoordinatesRect(x,y,w,h,padding), 'wi': w, 'he':h})
        return im_contours, listBarCode
    except Exception as e:
        print(e)
        return img4k, None

def setPropertyContainer():
    cap = cv2.VideoCapture('video/Demo.mp4')
    index = 0
    size = 800
    blur = 37
    canny = 32
    dial = 3
    thres = 3

    while True:
        # img = cv2.imread(path)
        index += 1
        success, img = cap.read()
        imgReal = img.copy()
        img, rate = resize(img, size)
      

        imgBlur, imgCanny, imgDial, imgThres = preProcess(img = img, kernelBlur=(blur,blur), cannyValue=canny, kernelDial=(dial,dial), kernelThres=(thres,thres))
        # print(blur, canny, dial, thres)

        imgContour, listContainer = getContoursContainer(img, imgThres)

        for container in listContainer:
            img_Warp = getWarp(imgReal, biggest=container['coordinates'], wiImg=container['wi'], heImg=container['wi'], rate = rate, index = index)
            setPropertyBarCode(img_Warp, imgReal, container['x'] / rate, container['y'] / rate)

        #show result
        # imgStack = stackImages(0.6,([imgContour, imgBlur, imgCanny], [imgDial, imgThres, img_Warp]))
        # cv2.imshow("Stacked Images", imgStack)
        cv2.waitKey(1)

def getContoursContainer(img, imgThres):
    try:
        padding = 7
        contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        im_contours = img.copy()
        listContainer = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 7000 and area <15000:
                peri = cv2.arcLength(cnt, True) 
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                x,y,w,h = cv2.boundingRect(approx)
                obj_cor = len(approx)
                if(obj_cor == 4):
                    cv2.putText(im_contours, '%s'%area, (x , y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (70, 0, 50), 2)
                    cv2.rectangle(im_contours, (x - padding, y - padding), (x + w + padding, y + h + padding), (100, 10, 100), 2)
                    if len(approx) == 4:
                        dic = {'coordinates': approx, 'wi': w, 'he':h,'x':x, 'y':y}
                        listContainer.append(dic)
        return im_contours, listContainer
    except Exception as e:
        print(e)
        return img, listContainer


setPropertyContainer()