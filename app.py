# person_detection_app.py
import streamlit as st
import cv2
import numpy as np

st.title("Person Detection using OpenCV + Streamlit")

# Start webcam
run = st.checkbox("Start Webcam")

# Load the pre-trained HOG + SVM detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)  # 0 means default webcam

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Detect people
    (regions, _) = hog.detectMultiScale(frame,
                                        winStride=(4, 4),
                                        padding=(8, 8),
                                        scale=1.05)

    # Draw rectangles around detected people
    for (x, y, w, h) in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert BGR to RGB (for Streamlit display)
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
