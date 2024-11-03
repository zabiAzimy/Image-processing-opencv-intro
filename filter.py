import numpy as np
import cv2

def make_gaussian(size, fwhm=3, center=None) -> np.ndarray:
    """ Make a square gaussian kernel.

    :param size is the length of a side of the square
    :param fwhm is full-width-half-maximum, which
    :param can be thought of as an effective radius.
    :return np.array
    """
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
    return k / np.sum(k)

def convolution_2d(img, kernel) -> np.ndarray:
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    offset = kernel.shape[0] // 2
    padded_img = np.pad(img, pad_width=offset, mode='constant', constant_values=0)
    result = np.zeros_like(img, dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i, j] = (kernel * padded_img[i:i + kernel.shape[0], j:j + kernel.shape[1]]).sum()

    return np.clip(result, 0, 255)

if __name__ == "__main__":
    # 1. Load image in grayscale
    img = cv2.imread("graffiti.png", cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape[:2]

    # Create a grayscale image in RGB format
    gray_img_rgb = cv2.merge([img, img, img])

    # Load the original color image
    color_img = cv2.imread("graffiti.png", cv2.IMREAD_COLOR)

    # Concatenate images side by side
    combined_image = np.concatenate((gray_img_rgb, color_img), axis=1)

    # Display the side-by-side images
    cv2.imshow("Gray and Color Side-by-Side", combined_image)

    # Image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    # 2. Use image kernels
    sobel_x = convolution_2d(img, sobelmask_x).astype(np.uint8)
    sobel_y = convolution_2d(img, sobelmask_y).astype(np.uint8)

    # 3. Compute magnitude of gradients
    mog = np.sqrt(sobel_x ** 2 + sobel_y ** 2).astype(np.uint8)

    # Show resulting images
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("Magnitude of Gradients", mog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
