# person_detection_yolo.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.title("Person Detection using Machine Learning (YOLOv8)")

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # lightweight YOLOv8 model

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Run YOLO detection
    results = model(img)

    # Draw bounding boxes for persons only
    person_count = 0
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == "person":
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show output
    st.image(img_rgb, caption=f"Persons detected: {person_count}")
    st.success(f"Number of persons detected: {person_count}")
