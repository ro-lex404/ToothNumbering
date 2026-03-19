# Load your best custom model
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("best.pt") #Path to best.pt

# Run inference
results = model.predict(
    source="test-image-1.jpg",
    conf=0.2,      # Ignore boxes with < 25% confidence
    iou=0.3,       # If two boxes overlap by > 45%, keep only the higher confidence one
    agnostic_nms=True # Set to True if you want to prevent overlaps between DIFFERENT classes
)

# Process results
for r in results:
    count = len(r.boxes.cls)
    print(r.boxes.cls)   # Detected tooth numbers
    print(r.boxes.conf)  # Confidence scores

print("Number of teeth", count)

# The 'results' object contains the plotted image in the '.plot()' method
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    
    plt.figure(figsize=(10, 10))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.show()