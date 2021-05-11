import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

img = cv2.imread('data/processed/covid/b/5/239.png', 0)
# edges = cv2.Canny(img,50,150,3)
ret,edges = cv2.threshold(img,50,200,cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
edges = cv2.morphologyEx(edges,cv2.MORPH_OPEN,kernel)

img2 = cv2.imread('data/processed/control/normal-chest-ct-lung-window-1(1).png', 0)
# edges2 = cv2.Canny(img2,50,150,3)
ret,edges2 = cv2.threshold(img2,100,150,cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
edges2 = cv2.morphologyEx(edges2,cv2.MORPH_OPEN,kernel)

# f = np.fft.fft2(edges)
# fshift = np.fft.fftshift(f)
# magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(edges, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

cv2.imwrite("a.png", edges)
cv2.imwrite("b.png", edges2)


plt.show()