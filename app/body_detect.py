import cv2
import numpy as np

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

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/bobumble/Documents/scouter_app/app/haarcascade_frontalface_default.xml')
cv2.namedWindow('BinaryThreshold')
cv2.namedWindow('Erosion/Dilation')


#TrackBar for the Filtering
cv2.createTrackbar('minH','BinaryThreshold',0,255,nothing)
cv2.createTrackbar('minS','BinaryThreshold',0,255,nothing)
cv2.createTrackbar('minV','BinaryThreshold',0,255,nothing)
cv2.createTrackbar('maxH','BinaryThreshold',255,255,nothing)
cv2.createTrackbar('maxS','BinaryThreshold',255,255,nothing)
cv2.createTrackbar('maxV','BinaryThreshold',255,255,nothing)

#Trackbar for the erosion/dilation
cv2.createTrackbar('Erode','Erosion/Dilation',0,10,nothing)
cv2.createTrackbar('Dilate','Erosion/Dilation',0,10,nothing)
cv2.createTrackbar('Kernel','Erosion/Dilation',1,10,nothing)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    face = facedetect(frame)
    # Our operations on the frame come here
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

    minH = cv2.getTrackbarPos('minH','BinaryThreshold')
    maxH = cv2.getTrackbarPos('maxH','BinaryThreshold')
    minS = cv2.getTrackbarPos('minS','BinaryThreshold')
    maxS = cv2.getTrackbarPos('maxS','BinaryThreshold')
    minV = cv2.getTrackbarPos('minV','BinaryThreshold')
    maxV = cv2.getTrackbarPos('maxV','BinaryThreshold')

    # define range of blue color in HSV
    lower_bound = np.array([minH,minS,minV])
    upper_bound = np.array([maxH,maxS,maxV])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    erodemask = mask
    #applying erosion and dilation
    erode = cv2.getTrackbarPos('Erode','Erosion/Dilation')
    dilate = cv2.getTrackbarPos('Dilate','Erosion/Dilation')
    element = cv2.getTrackbarPos('Kernel','Erosion/Dilation')
    kernel = np.ones((element,element), np.uint8)
    erodemask = cv2.erode(erodemask,  kernel, iterations=erode)
    erodemask = cv2.dilate(erodemask,  kernel, iterations=dilate)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)

    """
    blur = cv2.GaussianBlur(hsv,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    """

    # Display the resulting frame
    cv2.imshow('hsv',hsv)
    cv2.imshow('frame',frame)
    cv2.imshow('bg',mask)
    cv2.imshow('Erode',erodemask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


