import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv.imread('graffiti.png', cv.IMREAD_GRAYSCALE)

# Compute FFT and shift the zero frequency to the center
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Calculate the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.axis('off')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.axis('off')
plt.show()

# Apply a mask to filter frequencies (e.g., low-pass filter)
rows, cols = img.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# Apply the mask and inverse DFT to get the filtered image
fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Display the filtered image
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image'), plt.axis('off')
plt.show()
