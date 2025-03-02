import cv2
import numpy as np
import os

# Define paths for the model files
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"

# Ensure model files exist
if not os.path.exists(prototxt_path):
    raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Caffe model file not found: {model_path}")

print("Files exist. Proceeding with model loading...")

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

if net.empty():
    raise Exception("Failed to load model. Ensure the .prototxt and .caffemodel files are correct.")

print("Model loaded successfully!")

def detect_face(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    print(f"Blob shape: {blob.shape}")  # Debugging output

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            return frame  # Return frame with detected face
    return frame  # If no face is detected, return original frame

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_face(frame)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
