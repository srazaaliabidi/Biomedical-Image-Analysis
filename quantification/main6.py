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

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)
# Low pass filter
mask = np.ones((rows, cols, 2), np.uint8)
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

magnitude_spectrum = 20*np.log(np.abs(f_ishift))

f2 = np.fft.fft2(img2)
fshift2 = np.fft.fftshift(f2)

f_ishift2 = np.fft.ifftshift(fshift2)
img_back2 = np.fft.ifft2(f_ishift2)
img_back2 = np.abs(img_back2)

magnitude_spectrum2 = 20*np.log(np.abs(f_ishift2))

plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# cv2.imwrite("c.png", magnitude_spectrum)
# cv2.imwrite("d.png", magnitude_spectrum2)

plt.show()