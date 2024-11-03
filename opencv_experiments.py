import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Wait for key press to switch modes
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1  # Color space conversion to HSV
    elif ch == ord('2'):
        mode = 2  # Color space conversion to LAB
    elif ch == ord('3'):
        mode = 3  # Color space conversion to YUV
    elif ch == ord('4'):
        mode = 4  # Adaptive Gaussian Thresholding
    elif ch == ord('5'):
        mode = 5  # Otsu Thresholding
    elif ch == ord('6'):
        mode = 6  # Canny Edge Detection
    elif ch == ord('q'):
        break

    # Apply processing based on mode
    if mode == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif mode == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    elif mode == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    elif mode == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif mode == 5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == 6:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.Canny(gray, 100, 200)

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
