# 850 x 1100 page size for 8.5 inch and 11 inch page
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
def flip_image(img):
    w, h = img.shape[::-1]
    center = (w/2,h/2)
    M = cv.getRotationMatrix2D(center, 180, 1.0)
    imgUP = cv.warpAffine(img, M, (w, h))
    return imgUP
output_file = "res.png"
if (len(sys.argv) > 2):
    output_file = sys.argv[2]
read_image = sys.argv[1]
img_rgb = cv.imread(read_image)
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
# img_gray = cv.equalizeHist( img_gray )
template = cv.imread('blackbox.png',0)
bad_template = cv.imread('whitebox.png',0)
w, h = template.shape[::-1]

res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
# threshold = 0.7
# sort by value
points = np.unravel_index(np.argsort(res, axis=None)[-64:], res.shape)
#print(points)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# loc = res #np.where( res >= threshold )
loc = points
#loc = list(zip(*loc[::-1]))
# print("initial loc")
oloc = list(zip(*loc[::-1])) 
for pt in oloc:
    # print(pt)
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,128,30), 3)
# cv.imwrite('res.png',img_rgb)

def similar(pt1,pt2,threshold=15):
    return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 < threshold**2
def avgpt(pts):
    l = len(pts)
    if l == 0:
        return (0,0)
    x = 0
    y = 0
    for pt in pts:
        x += pt[0]
        y += pt[1]
    return (int(x/(1.0*l)),int(y/(1.0*l)))

def filter_similar_points(pts):
    # keep earlier points
    out = list()
    # N^2 just remove dupes
    while len(pts) > 0:
        pt = pts[0]
        #print(pt)
        simpts = [npt for npt in pts if similar(pt,npt)]
        pts.pop(0)
        apt = avgpt(simpts)
        out.append(apt)
        #out.append(pt)
        pts = [npt for npt in pts if not similar(pt,npt)]
    return out

def is_white_box(pt):
    avgpixel = np.sum(img_gray[pt[1]:pt[1]+h,pt[0]:pt[0]+w])/(1.0*w*h)
    return avgpixel > 160

pts = oloc
newloc = filter_similar_points(list(pts))
newloc = [pt for pt in newloc if not is_white_box(pt)]

# filter out white boxes
# newloc = [pt for pt in pts if not is_white_box(pt)]
# newloc = pts
K=7
if len(newloc) < K:
    newloc = oloc
    # newloc = filter_similar_points(list(pts))
    newloc = [pt for pt in newloc if not is_white_box(pt)]

# blank will freak out
if len(newloc) >= K:
    Z=np.float32(np.array(newloc))
    #find 7 centers
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    center = np.int32(center)
    pts = list(map(tuple, center))
    
    newloc = pts
#print(pts)
#for pt in pts:
#    # print(pt)
#    cv.rectangle(img_rgb, pt, (int(pt[0]) + w, pt[1] + h), (0,0,255), 3)
#cv.imwrite('res.png',img_rgb)

#exit(1)

#print("newloc")
#print(newloc)
#print(len(loc))
#print(len(newloc))
#for pt in newloc:
#    if not (pt in oloc):
#        print("Not in original")
#        print(pt)


# classify

points = sorted(newloc,key=lambda x: x[0])
# center points
points = [(x[0]+w/2,x[1]+h/2) for x in points]
digits_mid = [40,63,82,104,124,144,166,186,206,229]
zipped_centers = list(zip(digits_mid, range(0,10)))
def closest_digit(point,h=276,templateh=276):
    my = point[1]
    ny = templateh*my/(1.0*h)
    closest = 0
    diff = templateh
    for y,d in zipped_centers:
        newdiff = abs(ny - y)
        if newdiff < diff:
            diff = newdiff
            closest = d
    return closest
# print([(p[0],p[1]) for p in points])
sid = "".join([str(closest_digit(x)) for x in points])
if sid == "":
    sid = "blank"
expected = os.path.basename(read_image).split(".")[0]
if "blank" in expected:
    expected = "blank"
print(f"{sid} {expected} {sid==expected}")


for pt in newloc:
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,0), 3)
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
cv.imwrite(output_file+"."+str(sid)+".png",img_rgb)

# check:
#  image difference
#  per column generation
