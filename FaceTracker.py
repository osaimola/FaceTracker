# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import cv2

# load haar cascade classifier trained to recognize faces
face_cascade = cv2.CascadeClassifier(
    r"haarcascade_frontalface_default.xml")

# get input from webcam
video_input = cv2.VideoCapture(0)

while True:
    check, frame = video_input.read()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # return list with coordinates of detected faces in an x, y, width & height arrangement
    faces = face_cascade.detectMultiScale(grayscale_frame, scaleFactor=1.05, minNeighbors=5)

    try:
        x, y, w, h = faces[0]

        """ compute coordinates for adjusted face frame
                face detection is aquare from eyes to lips so add padding to capture 
                chin and hair and keep the 1.33 ratio of width to height"""
        a = h / 2
        x1 = x - (1.6665 * a)
        x2 = h + (x + (1.6665 * a))
        y1 = y - a
        y2 = h + (y + a)
        cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]

        # if new frame is within logical image bounds (480x640 input used), crop, resize and display image
        if x1 >= 0 and y1 >= 0 and x2 <= 640 and y2 <= 480:
            # display resulting image
            cropped_image = cv2.resize(cropped_image, (640, 480))
            image = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            horizontal_stacked = numpy.hstack((cropped_image, image))
            cv2.imshow("FaceFinder", horizontal_stacked)
        else:
            # display full image with outline on detected face
            # cv2.rectangle(source, top_left_coordinates, bottom_right_coordinates, BGR_color_values, line_thickness
            image = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            horizontal_stacked = numpy.hstack((image, frame))
            cv2.imshow("FaceFinder", horizontal_stacked)

    # if no faces detected, show original frame
    except IndexError:
        horizontal_stacked = numpy.hstack((frame, frame))
        cv2.imshow("FaceFinder", horizontal_stacked)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video_input.release()