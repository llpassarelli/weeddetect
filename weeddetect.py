#!/usr/bin/python
# Python 2/3 compatibility
# dev by a human
# llpassarelli@gmail.com - 2017
from __future__ import print_function
import cv2
import numpy as np
import time

def filter(mask):
	kernel = np.ones((7,7), np.uint8)
	erode = cv2.erode(mask,kernel,iterations = 1)
	dilate = cv2.dilate(erode, kernel,iterations = 2)
	result = cv2.GaussianBlur(dilate,(3,3),1)
	return result

def update(*arg):
	start = time.time()
	h0 = cv2.getTrackbarPos('h min', 'control')
	h1 = cv2.getTrackbarPos('h max', 'control')
	s0 = cv2.getTrackbarPos('s min', 'control')
	s1 = cv2.getTrackbarPos('s max', 'control')
	v0 = cv2.getTrackbarPos('v min', 'control')
	v1 = cv2.getTrackbarPos('v max', 'control')
	lower = np.array((h0,s0,v0))
	upper = np.array((h1,s1,v1))
	mask = cv2.inRange(hsv, lower, upper)
	filtered = filter(mask)
	#contours
	ret, contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	total_contours = len(contours)
	result = src.copy()
	total_area = 0
	contours_error = []
	contours_ok=[]
	for cnt in contours:
		area = cv2.contourArea (cnt)
		print ("area: ", area)
		total_area += area
		if (area > 4000 and area < 52500):
			contours_ok.append(cnt)
		else:
			print ("ERROR AREA: ", area)
			contours_error.append(cnt)

	if len(contours)>0:
		media = total_area/len(contours)
		print ("\tAREA MEDIA: ", media)
	cv2.imshow('original', src)
	#draw
	for crc in contours_ok:
	            (x,y),radius = cv2.minEnclosingCircle(crc)
	            center = (int(x),int(y))
	            radius = int(radius)
	            centerTxt = (int(x),int(y))
	            cv2.circle(result, center,radius,(0,250,0), 1)
	            cv2.circle(result, center,3,(0,250,0), 6)
	#time
	stop = time.time()
	diff = stop - start
	t = str("%.3f" % diff)
	fps = str("%.0f" % (1/diff))
	text = "t["+t+"] fps:["+fps+"] AREAS:["+str(len(contours_ok))+"]"
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(result,text,(15,30), font, .6, (255, 255, 255), 1)
	#show
	print ("\tTOTAL: ",total_contours)
	print ("\tOK: ", len(contours_ok))
	print ("\tERRORS: ", len(contours_error))
	cv2.imshow('mask', mask)
	cv2.imshow('filter: ', filtered)
	cv2.imshow ('agrostorm', result)

def main():
	cv2.namedWindow('control', 0)
	cv2.createTrackbar('h min', 'control', 30, 255, update)
	cv2.createTrackbar('h max', 'control', 255, 255, update)
	cv2.createTrackbar('s min', 'control', 60, 255, update)
	cv2.createTrackbar('s max', 'control', 255, 255, update)
	cv2.createTrackbar('v min', 'control', 0, 255, update)
	cv2.createTrackbar('v max', 'control', 255, 255, update)
	im = cv2.resize(src, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
	cv2.imshow('control',im)
	update()
	while 1:
		ch = cv2.waitKey(30)
		if (ch == 27):
			break
			cv2.destroyAllWindows()

if __name__ == '__main__':
	import sys
	try:
		fn = sys.argv[1]
		print("parametro:", fn)
	except:
		fn = 'grama.jpg'

	src = cv2.imread(fn)
	src = cv2.resize(src, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
	print("resized shape:", src.shape)
	hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
	main()
