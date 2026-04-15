from ultralytics import YOLO
import cv2
import sys

# Load the trained model
model = YOLO('best (2).pt')

# Get image path from command line or use default
image_path = sys.argv[1] if len(sys.argv) > 1 else 'path/to/your/image.jpg'

# Predict
results = model.predict(image_path, conf=0.25, save=True)

# Display predictions
for result in results:
    boxes = result.boxes
    if len(boxes) > 0:
        print(f"\nDetected {len(boxes)} object(s):")
        for i in range(len(boxes)):
            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            class_name = model.names[cls]
            print(f"  - {class_name}: {conf:.2%}")
    else:
        print("No objects detected")

