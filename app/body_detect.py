import cv2
import numpy as np
import sys

def denoise(frame):
    frame = cv2.medianBlur(frame,5)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    return frame 

def facedetect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	return frame[y-30 :y+h + 30, x-30: x+h + 30]


def nothing(x):
    pass

def ObjectTract(frame):
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()

    bbox = cv2.selectROI("tracking", frame)

    k = tracker.init(frame, bbox)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        ok, bbox = tracker.update(frame)
        
        if ok:
                # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display the resulting frame
        cv2.imshow('frame',frame)
        #cv2.imshow('Erode',erodemask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.namedWindow('PeriApprox')
cv2.createTrackbar('Delta','PeriApprox',0,100,nothing)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): 
    print("Could not open the video camera")
    sys.exit()

ret, frame = cap.read() 
if not ret:
    print("Cannot read frame")
    sys.exit()

face_cascade = cv2.CascadeClassifier('/home/bobumble/Documents/scouter_app/app/haarcascade_frontalface_default.xml')
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

     # Define an initial bounding box
    face = facedetect(frame)
    fgmask = fgbg.apply(frame)

    image, contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
    screenCnt = None
    for c in contours :
        peri = cv2.arcLength(c, True)
        delta = cv2.getTrackbarPos('Delta','PeriApprox')/100
        approx = cv2.approxPolyDP(c, delta * peri, True)

        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('fgmask', fgmask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


