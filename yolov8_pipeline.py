# yolov8_pipeline.py

from ultralytics import YOLO
import os
import cv2

# 0. Setup Paths
PROJECT_DIR = 'project'
DATA_DIR = os.path.join(PROJECT_DIR, 'dataset')
MODEL_NAME = 'yolov8n.pt'  # Or yolov8s.pt, yolov8m.pt etc.
NUM_EPOCHS = 100

# 1. Create Dataset YAML
yaml_content = f"""
path: {DATA_DIR}
train: images/train
val: images/val
nc: 1  # Change to your number of classes
names: ['object']  # Change to your actual class names
"""

with open('custom.yaml', 'w') as f:
    f.write(yaml_content)

# 2. Train the model
model = YOLO(MODEL_NAME)  # Load pretrained YOLOv8
model.train(data='custom.yaml', epochs=NUM_EPOCHS)

# 3. Load trained model
trained_model_path = "runs/detect/train5/weights/best.pt"
model = YOLO(trained_model_path)

# 4. Inference on image
results = model('test.jpg', save=True)  # Replace with your test image path

# 5. Real-time inference (Webcam)
cap = cv2.VideoCapture(0)  # Use 0 for webcam or path to video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
