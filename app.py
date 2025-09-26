# person_detection_upload.py
import streamlit as st
import cv2
import numpy as np

st.title("Person Detection (Image Upload) using OpenCV")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Load HOG + SVM people detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    (regions, _) = hog.detectMultiScale(frame,
                                        winStride=(4, 4),
                                        padding=(8, 8),
                                        scale=1.05)

    for (x, y, w, h) in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image")
