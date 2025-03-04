import base64
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FOCAL_LENGTH = 540  
KNOWN_WIDTH = 0.15  

# Load Face SSD model with ShuffleNetV2 backbone
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"

if not os.path.exists(modelFile) or not os.path.exists(configFile):
    raise FileNotFoundError("Model files not found! Ensure they exist in the same directory.")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face_and_distance(frame):
    try:
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (640, 640), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3: 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                perceived_width = endX - startX
                
                if perceived_width > 0:
                    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width
                else:
                    distance = None
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"{distance:.2f}m", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                logger.info(f"Detected face at distance: {distance:.2f} meters")
                return distance, frame

        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return None, frame
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return None, frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if "image" not in data:
                await websocket.send_json({"error": "No image received"})
                continue

            try:
                img_data = base64.b64decode(data["image"])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error("Failed to decode image")
                    await websocket.send_json({"error": "Failed to decode image"})
                    continue

                distance, processed_frame = detect_face_and_distance(frame)
                
                _, buffer = cv2.imencode('.jpg', processed_frame)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "image": jpg_as_text,
                    "distance": float(distance) if distance is not None else -1,
                    "faceDetected": distance is not None
                })
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await websocket.send_json({"error": f"Error processing frame: {str(e)}"})
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
