import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow('Learning from images: SIFT feature visualization')

# Initialize SIFT detector
sift = cv2.SIFT_create()

while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame if necessary (Optional)
    # frame = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale as SIFT works on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints on the original colored frame
    frame_with_keypoints = cv2.drawKeypoints(
        frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the frame with keypoints
    cv2.imshow('Learning from images: SIFT feature visualization', frame_with_keypoints)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
