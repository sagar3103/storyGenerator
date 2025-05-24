from ultralytics import YOLO
import cv2
import json
import sys

# Load the model
model = YOLO('yolov8n.pt')  # Using the nano model for faster inference

def detect_objects(input_path, output_path):
    # Run inference
    results = model(input_path)
    
    # Get detection results
    detections = []
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            
            detections.append({
                "name": class_name,
                "confidence": confidence,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })
    
    # Save image with annotations
    img = cv2.imread(input_path)
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        label = f"{det['name']} {det['confidence']:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    
    return detections

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python detect.py <input_image_path> <output_image_path>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Run detection
    results = detect_objects(input_path, output_path)
    
    # Print results as JSON so they can be parsed by the Node.js server
    print(json.dumps(results))