import cv2
import numpy as np

jarak = 270
focal_length = 887

cap = cv2.VideoCapture(0)


panel = np.zeros([100, 700], np.uint8)
cv2.namedWindow('panel')

def nothing(x):
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

def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            tinggi = (jarak*w)/focal_length
            lebar_bahu = (jarak*h)/focal_length
            print("Tinggi = " + str(tinggi))
            print("Lebar bahu = " + str(lebar_bahu))

cv2.createTrackbar('L – h', 'panel', 0, 179, nothing)
cv2.createTrackbar('U – h', 'panel', 179, 179, nothing)

cv2.createTrackbar('L – s', 'panel', 0, 255, nothing)
cv2.createTrackbar('U – s', 'panel', 255, 255, nothing)

cv2.createTrackbar('L – v', 'panel', 0, 255, nothing)
cv2.createTrackbar('U – v', 'panel', 255, 255, nothing)

cv2.createTrackbar('S ROWS', 'panel', 0, 480, nothing)
cv2.createTrackbar('E ROWS', 'panel', 480, 480, nothing)
cv2.createTrackbar('S COL', 'panel', 0, 640, nothing)
cv2.createTrackbar('E COL', 'panel', 640, 640, nothing)

while True:
    success, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #frame = cv2.resize(frame, (720, 1280))
    #print(frame.shape)
    imgContour = frame.copy()
    imgBlur = cv2.GaussianBlur(frame, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    
    s_r = cv2.getTrackbarPos('S ROWS', 'panel')
    e_r = cv2.getTrackbarPos('E ROWS', 'panel')
    s_c = cv2.getTrackbarPos('S COL', 'panel')
    e_c = cv2.getTrackbarPos('E COL', 'panel')
    
    roi = frame[s_r: e_r, s_c: e_c]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos('L – h', 'panel')
    u_h = cv2.getTrackbarPos('U – h', 'panel')
    l_s = cv2.getTrackbarPos('L – s', 'panel')
    u_s = cv2.getTrackbarPos('U – s', 'panel')
    l_v = cv2.getTrackbarPos('L – v', 'panel')
    u_v = cv2.getTrackbarPos('U – v', 'panel')
    
    lower_green = np.array([l_h, l_s, l_v])
    upper_green = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    
    bg = cv2.bitwise_and(roi, roi, mask=mask)
    fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    getContours(mask_inv, bg)

    # print output
    imgStack = stackImages(0.8, ([roi, bg]))
    cv2.imshow("Result", imgStack)

    cv2.imshow('bg', bg)
    cv2.imshow('fg', fg)
    
    cv2.imshow('panel', panel)
    
    k = cv2.waitKey(30) 
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
