import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
def flip_image(img):
    w, h = img.shape[::-1]
    center = (w/2,h/2)
    M = cv.getRotationMatrix2D(center, 180, 1.0)
    imgUP = cv.warpAffine(img, M, (w, h))
    return imgUP
output_file = "res.png"
if (len(sys.argv) > 2):
    output_file = sys.argv[2]
img_rgb = cv.imread(sys.argv[1])
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('blank001.png',0)
w, h = template.shape[::-1]
templateUp = flip_image(template)

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
res2 = cv.matchTemplate(img_gray,templateUp,cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
min_val2, max_val2, min_loc2, max_loc2 = cv.minMaxLoc(res2)
pt = max_loc
flip = False
if (max_val < max_val2):
    pt = max_loc2
    flip = True
print((flip,max_val, max_val2, pt))
#cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#cv.imwrite('res.png',img_rgb)
sub = img_gray[pt[1]:pt[1]+h,pt[0]:pt[0]+w]
if flip:
    sub = flip_image(sub)
cv.imwrite(output_file,sub)
