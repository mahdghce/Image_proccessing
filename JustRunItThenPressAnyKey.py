import numpy as np
import cv2
from imutils.video import FPS
import imutils

# Detect Blue Channel
img = cv2.imread('mgh1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('Original image', img)
cv2.imshow('Result image', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# Gray Image
img = cv2.imread('mgh4.jpg')
cv2.imshow('Original image', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# Gaussian Filter
img = cv2.imread('mgh4.jpg', 0)
blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Original image", img)
cv2.imshow("Gaussian Filter", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# Rotate Image
img = cv2.imread('mgh2.jpg')
(h, w) = img.shape[:2]
center = (w / 2, h / 2)
angle90 = 90
angle180 = 180
angle270 = 270
scale = 1.0
temp = cv2.getRotationMatrix2D(center, angle90, scale)
rotated90 = cv2.warpAffine(img, temp, (h, w))
cv2.imshow('Original Image', img)
cv2.imshow('Rotated image', rotated90)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# Resize Image
img = cv2.imread('mgh1.jpg', cv2.IMREAD_UNCHANGED)
scale_percent = 60
width = int(img.shape[1] / 2)
height = int(img.shape[0])
dim = (width, height)
reSized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Original image', img)
cv2.imshow('Re_sized image', reSized)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# Edge Detection
img = cv2.imread('mgh4.jpg', )
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noiseLessImg = cv2.GaussianBlur(grayImg, (3, 3), 0)
L = cv2.Laplacian(noiseLessImg, cv2.CV_64F)
SX = cv2.Sobel(noiseLessImg, cv2.CV_64F, 1, 0, ksize=5)
SY = cv2.Sobel(noiseLessImg, cv2.CV_64F, 0, 1, ksize=5)
cv2.imshow("Original edge detection image", img)
cv2.imshow("Laplace edge detection image", L)
cv2.imshow("Sbl_X edge detection image", SX)
cv2.imshow("Sbl_Y edge detection image", SY)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# Image Segmentation
img = cv2.imread('mgh3.jpg')
originalImg = cv2.imread('mgh3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret1, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
ret2, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv2.imshow('Original image', originalImg)
cv2.imshow('Segmentation result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# Face Detection
faceCascade = cv2.CascadeClassifier("HarCascade_FrontFace_default.xml")
image = cv2.imread('mgh4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

c = cv2.waitKey(0)
if 'q' == chr(c & 255):
    exit()

# 0.5 frame per second
stream = cv2.VideoCapture('test.mp4')
fps = FPS().start()
while True:
    (grabbed, frame) = stream.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    cv2.imshow("Result video", frame)
    # in each 500 milliseconds frame will change which means 0.5 second
    c = cv2.waitKey(500)
    fps.update()
    if 'q' == chr(c & 255):
        cv2.destroyAllWindows()
        exit()
stream.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
