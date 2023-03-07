import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform 

parser = argparse.ArgumentParser(description='Matin sudoku detector')
parser.add_argument('--input', type=str, help='path of your input image')
parser.add_argument('--output', type=str, help='path of your output image', default='output.jpg')

args = parser.parse_args()

img = cv2.imread(args.input)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blurred = cv2.GaussianBlur(img_gray, (7,7), 3)

thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]

contours = sorted(contours, key=cv2.contourArea, reverse=True)

sudoku_contour = None

for contour in contours:
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        sudoku_contour = approx
        break
    
if sudoku_contour is None:
    print('Not found!')

else:

    sudoku = cv2.drawContours(img , [sudoku_contour] , -1, (0,255,0), 20)

  
    warped = four_point_transform(img, approx.reshape(4,2))
    warped = cv2.resize(warped, (500, 500))


    cv2.imwrite(args.output, warped)
    cv2.waitKey()