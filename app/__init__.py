from flask import Flask
import cv2

#start the application
#Get access to the camera and the microphone
#Once a person gets within view, start all the other functions (use the anti-cyclic dependency scheme as from the flask tutorials

from app import face_detect, body_detect, sound_detect
