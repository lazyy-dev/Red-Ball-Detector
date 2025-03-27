 
import cv2 as cv
import sys
import numpy as np
import glob
import os


def image_detect(path):
    img = cv.imread(path)
 
    if img is None:
        sys.exit("Could not read the image.")

    # Converting the input to HSV colour scheme
    hsv= cv.cvtColor(img, cv.COLOR_BGR2HSV)
    

    #Defining lower and upper bounds for both brighter and darker shades of red to improve accurracy
    bright_red_lower = (0, 125, 125)
    bright_red_upper = (10, 255, 255)
    bright_red_mask = cv.inRange(hsv, bright_red_lower, bright_red_upper)

    dark_red_lower = (160, 125, 125)
    dark_red_upper = (179, 255, 255)
    dark_red_mask = cv.inRange(hsv, dark_red_lower, dark_red_upper)
 
    weighted = cv.addWeighted(bright_red_mask, 1.0, dark_red_mask, 1.0, 0.0)

    # Blurring the image
    blurred = cv.GaussianBlur(weighted,(17,17),3,3)

    # some morphological operations to remove small blobs 
    erode = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
    eroded = cv.erode(blurred,erode)
    dilated = cv.dilate(eroded,dilate)

    detected_circles = cv.HoughCircles(
        dilated, cv.HOUGH_GRADIENT, 1, 200,
        param1=100, param2=22, minRadius=20, maxRadius=250
    )

    # on the color-masked, blurred and morphed image I apply the cv2.HoughCircles-method to detect circle-shaped objects 
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))  # Convert float to integer

        for circle in detected_circles[0, :]:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])  # Convert to int
            cv.circle(img, (x, y), r, (0, 255, 0), thickness=3)

    # Show image
    cv.imshow("Detected Red Ball", img)
    cv.waitKey(0)  # Keep window open
    cv.destroyAllWindows()


        
        
if __name__=='__main__':
    #Red data --Pass
    image_detect(r"sample_data\big_red_ball.jpg")
    image_detect(r"sample_data\images\IMG_2283_JPG.rf.dae46b04cd7985cfb7f4f48f8e7f783c.jpg")
    image_detect(r"sample_data\images\IMG_2475_JPG.rf.702f60bb59fa116f1f67b1fe83a90baa.jpg")
    image_detect(r"sample_data\images\IMG_2536_JPG.rf.ed09afe0859409575e91c8b2efb32bc8.jpg")
    image_detect(r"sample_data\images\IMG_2423_JPG.rf.214d8481c8af897a0ad9210e8c710de3.jpg")
    image_detect(r'sample_data\images\IMG_2407_JPG.rf.48898a4bcf29d35d2150cada340e59d8.jpg')
    image_detect(r"sample_data\cricket_red.jpg")
    image_detect(r"sample_data\red_shiny.jpg")

    image_detect(r"yolo\data\images\01d339a8-14_jpg.rf.71b6ee9f3259eea33b285c4608da9837.jpg")
    # image_detect(r"")
    # image_detect(r"")
    # image_detect(r"")


    #Fail
    image_detect(r"sample_data\red_all.jpg")
    image_detect(r"sample_data\multi_red.jpg")
    image_detect(r"sample_data\mix_ball.jpg")
    image_detect(r"sample_data\Not_red.jpg")
    image_detect(r"sample_data\test_Red.jpg")
    image_detect(r"sample_data\images\IMG_2422_JPG.rf.07e7c031fac51803bdc7d0790323d012.jpg")
    image_detect(r"yolo\data\images\f33b2868-frame_04850_png.rf.f81fcfcfe964aa2e45269cd0e5e58f67.jpg")



