from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2 as cv
import imutils
import time
"""
This file is for tracking a red ball through video streaming
"""

pts = deque(maxlen=14) #Queue to keep track of the path of the ball

vs = cv.VideoCapture(0)

time.sleep(2.0) #Give it some time to settle

#Looping frame by frame till the application is running
while True:
	ret,frame = vs.read()
	
	if not ret: #Check if the frame was successfully read 
		break
	frame =frame

	# resize the frame, blur it, and convert it to the HSV colour space 

	frame = imutils.resize(frame, width=600)
	blurred = cv.GaussianBlur(frame, (11, 11), 0) #Blurring to get rid of excess noise in the image
	hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    #Defining lower and upper bounds for both brighter and darker shades of red to improve accurracy
	bright_red_lower = (0, 100, 100)
	bright_red_upper = (10, 255, 255)
	bright_red_mask = cv.inRange(hsv, bright_red_lower, bright_red_upper)
	dark_red_lower = (160, 100, 100)
	dark_red_upper = (179, 255, 255)
	dark_red_mask = cv.inRange(hsv, dark_red_lower, dark_red_upper)
	
	weighted = cv.addWeighted(bright_red_mask, 1.0, dark_red_mask, 1.0, 0.0)

    # some morphological operations to remove small blobs 
	erode = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
	dilate = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
	eroded = cv.erode(weighted,erode)
	dilated = cv.dilate(eroded,dilate)
	

    # find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	 
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > 15:
			# draw the circle and centroid on the frame
			cv.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv.circle(frame, center, 5, (0, 0, 255), -1)
	# update the points queue
	pts.appendleft(center)
	

    # loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(14 / float(i + 1)) * 2.5)
		cv.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame 
	cv.imshow("Frame", frame)
	key = cv.waitKey(1) & 0xFF
	# if the 'q' key is pressed - Exit
	if key == ord("q"):
		break

vs.release()
cv.destroyAllWindows()