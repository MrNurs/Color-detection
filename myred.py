import cv2 as cv
import numpy as np


cam = cv.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame = cv.flip(frame, 1)
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # color red
    lowerRed1 = np.array([0,120,60],dtype=np.uint8)
    upperRed1 = np.array([10,255,255],dtype=np.uint8)

    lowerRed2 = np.array([170,120,60],dtype=np.uint8)
    upperRed2 = np.array([180,255,255],dtype=np.uint8)


    # masking
    mask1 = cv.inRange(hsv, lowerRed1, upperRed1)
    mask2 = cv.inRange(hsv, lowerRed2, upperRed2)
    mask = cv.bitwise_or(mask1, mask2)
    
    # noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)


    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      
    if(len(contours) >= 1):
        # taking the biggest contour
        cnt = max(contours, key=cv.contourArea)
        area = cv.contourArea(cnt)

        if area > 600:
            # drawing
            x, y ,w ,h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f"RED {area}",
                       (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv.LINE_AA)
        else:
            cv.putText(frame, "Red object too small", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame, f"NO RED {len(contours)}", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow("mask", mask)
    cv.imshow("result",frame)



    if (cv.waitKey(1) & 0xFF) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()