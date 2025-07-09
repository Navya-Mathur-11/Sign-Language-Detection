from ultralytics import YOLO

model = YOLO("runs/detect/train13/weights/best.pt")

# Lower confidence, resized image
results = model("/Users/NavyaMathur/Desktop/RealTimeObjectDetection/Project/datasets/project/test.jpg", save=True, conf=0.1)

for r in results:
    print("Boxes:")
    print(r.boxes)
    if r.boxes.shape[0] == 0:
        print("⚠️ No detections found.")
    else:
        r.show()  # or r.plot(), or r.save()

