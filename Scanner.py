from point_transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils

image=cv2.imread("cheque_02.jpeg")
test=cv2.imread("cheque.jpg")
cv2.imshow("image", image)
ratio=image.shape[0]/500.0
orig=image.copy()
image=imutils.resize(image,height=500)

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray, (5,5), 0)
edged=cv2.Canny(gray, 75, 200)

print("Edge Detection")
##cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.imwrite("Edge.jpg", edged)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

cnts=cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
cnts=sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c, 0.02*peri, True)
    
    if len(approx)==4:
        screenCnt=approx
        break

print("Find Contours")
cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)
cv2.imshow("Outline", image)
cv2.imwrite("contour.jpg",image)

print(screenCnt.reshape(4,2)*ratio)
wraped=four_point_transform(orig, screenCnt.reshape(4,2)*ratio)

wraped=cv2.cvtColor(wraped, cv2.COLOR_BGR2GRAY)
T=threshold_local(wraped, 11, offset=10, method="gaussian")

print("Perspective Transform")
wraped=imutils.resize(wraped, height=278)
cv2.imshow("Scanned", wraped)
cv2.imwrite("cheque_02_scanned.jpg",wraped)
cv2.imshow("Test",test)
print(test.shape)
