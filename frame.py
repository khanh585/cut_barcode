import cv2

cap = cv2.VideoCapture('video/Demo.mp4')
i = 0
while True:
    success, img = cap.read()
    if success:
        i += 1
        path = 'frame/ff%s.png'%i
        cv2.imwrite(path, img)
    else:
        break

cap.release()
cv2.destroyAllWindows()


