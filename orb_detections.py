import cv2
import numpy as np
import matplotlib.pyplot as plt


x_axis = []
i = 1
n_exp = []
n_exp1 = []
n_exp2 = []
n_exp3 = []

cap = cv2.VideoCapture('og_video.MP4')
cap1 = cv2.VideoCapture('exp1.wmv')
cap2 = cv2.VideoCapture('exp2.wmv')
cap3 = cv2.VideoCapture('exp3.wmv')
while (cap.isOpened()):
    x_axis.append(i)
    i+=1

#-------------------------- original ----------------------
    ## Capture frame-by-frame
    ret, frame = cap.read()
 
    ## Display the resulting frame
    #cv2.imshow('Frame', frame)
 
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    # find the keypoints with ORB
    kp = orb.detect(frame, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(frame, kp)
    n_exp.append(len(kp))

    # draw only keypoints location,not size and orientation
    img = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
        
    cv2.imshow('img', img)
#-------------------------- image 1 ----------------------
    ## Capture frame-by-frame
    ret1, frame1 = cap1.read()
 
    ## Display the resulting frame
    #cv2.imshow('Frame', frame)
 
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=200000)
    # find the keypoints with ORB
    kp1 = orb.detect(frame1, None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(frame1, kp1)
    n_exp1.append(len(kp1))

    # draw only keypoints location,not size and orientation
    img1 = cv2.drawKeypoints(frame1, kp1, None, color=(0,255,0), flags=0)
        
    cv2.imshow('img1', img1)
#-------------------------- image 2 ----------------------
    ## Capture frame-by-frame
    ret2, frame2 = cap2.read()
 
    ## Display the resulting frame
    #cv2.imshow('Frame', frame)
 
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=200000)
    # find the keypoints with ORB
    kp2 = orb.detect(frame2, None)
    # compute the descriptors with ORB
    kp2, des2 = orb.compute(frame2, kp2)
    n_exp2.append(len(kp2))

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(frame2, kp2, None, color=(0,255,0), flags=0)
    cv2.imshow('img2', img2)    
#-------------------------- image 3 ----------------------
    ## Capture frame-by-frame
    ret3, frame3 = cap3.read()
 
    ## Display the resulting frame
    #cv2.imshow('Frame', frame)
 
    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=200000)
    # find the keypoints with ORB
    kp3 = orb.detect(frame3, None)
    # compute the descriptors with ORB
    kp3, des3 = orb.compute(frame3, kp3)
    n_exp3.append(len(kp3))

    # draw only keypoints location,not size and orientation
    img3 = cv2.drawKeypoints(frame3, kp3, None, color=(0,255,0), flags=0)
    
    
    cv2.imshow('img3', img3)

    ## define q as the exit button
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
 
## release the video capture object
cap.release()
cap1.release()
cap2.release()
cap3.release()
## Closes all the windows currently opened.
cv2.destroyAllWindows()


plt.plot(x_axis, n_exp, label = "Original")
plt.plot(x_axis, n_exp1, label = "Exp 1")
plt.plot(x_axis, n_exp2, label = "Exp 2")
plt.plot(x_axis, n_exp3, label = "Exp 3")
plt.legend()
plt.show()

