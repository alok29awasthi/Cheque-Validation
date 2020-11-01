import numpy as np
import cv2
import matplotlib.pyplot as plt

img=cv2.imread('cheque.jpg',cv2.IMREAD_COLOR)

amt_word_1=img[90:112,130:450]
amt_word_2=img[112:142,50:470]
amt_num=img[112:132,550:640]
acc_num=img[155:175,60:240]
sign=img[175:235,500:1000]
ocr=img[240:280,60:1000]

cv2.imwrite('amt_word_1.jpg',amt_word_1)
cv2.imwrite('amt_word_2.jpg',amt_word_2)
cv2.imwrite('amt_num.jpg',amt_num)
cv2.imwrite('acc_num.jpg',acc_num)
cv2.imwrite('sign.jpg',sign)
cv2.imwrite('ocr.jpg',ocr)

cv2.imshow('amt_word_1.jpg',amt_word_1)
cv2.imshow('amt_word_2.jpg',amt_word_2)
cv2.imshow('amt_num.jpg',amt_num)
cv2.imshow('acc_num.jpg',acc_num)
cv2.imshow('sign.jpg',sign)
cv2.imshow('ocr.jpg',ocr)

##cv2.imshow('image',ocr)

##cv2.waitKey(0)
##cv2.destroyAllWindows()
