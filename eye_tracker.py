import cv2
import numpy as np
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import time

# Function to create a mask on the eyes
def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]

# Function to find the position of the eyeballs
def find_eyeball_position(end_points, cx, cy):
    x_ratio = (end_points[0] - cx) / (cx - end_points[2])  # Horizontal
    y_ratio = (cy - end_points[1]) / (end_points[3] - cy)  # Vertical

    if x_ratio > 2.0:
        return 1  # Looking right
    elif x_ratio < 0.5:
        return 2  # Looking left
    elif y_ratio < 0.5:
        return 3  # Looking up
    else:
        return 0  # Normal (centered)

# Function to perform contouring on the eyes
def contouring(thresh, mid, img, end_points, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        if right:
            cx += mid
        
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass

# Function to process the threshold image
def process_thresh(thresh):
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh

# Function to print eye position and display a message if not on screen
def print_eye_pos(img, left, right):
    if left == right and left != 0:
        text = ''
        
        if left == 1:
            text = 'Looking right'
        elif left == 2:
            text = 'Looking left'
        elif left == 3:
            text = 'Looking up'
        
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (30, 30), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        print("Please focus on the screen")  # Message when eyeballs are not detected

# Initialize face detector and landmark model
face_model = get_face_detector()
landmark_model = get_landmark_model()

# Define left and right eye landmarks
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

# Open the camera
cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass

cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

# Initialize a timer variable to keep track of time
last_message_time = time.time()

while True:
    ret, img = cap.read()
    rects = find_faces(img, face_model)
    eyeball_pos_left, eyeball_pos_right = None, None  # Initialize the positions

    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)

        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)

        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)

    if eyeball_pos_left is None and eyeball_pos_right is None:
        current_time = time.time()
        if current_time - last_message_time >= 2:  # Check if 30 seconds have passed
            print("Please focus on the screen")
            last_message_time = current_time  # Update the last message time

    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
