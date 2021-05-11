import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

img = cv2.imread('data/processed/covid/b/5/239.png', 0)
ret,edges = cv2.threshold(img,30,100,cv2.THRESH_BINARY)
edges = cv2.Canny(edges,50,150,3)

img2 = cv2.imread('data/processed/control/normal-chest-ct-lung-window-1(1).png', 0)
ret,edges2 = cv2.threshold(img2,125,150,cv2.THRESH_BINARY)
edges2 = cv2.Canny(edges2,125,150,3)

f = np.fft.fft2(edges)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

f2 = np.fft.fft2(edges2)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = 20*np.log(np.abs(fshift2))

plt.subplot(121),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# cv2.imwrite("c.png", magnitude_spectrum)
# cv2.imwrite("d.png", magnitude_spectrum2)

plt.show()