from ultralytics import YOLO
import cv2
import json
import sys

# Load the model
model = YOLO('yolov8n.pt')  # nano model

def detect_objects(input_path, output_path):
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        print(json.dumps({"error": f"Failed to load image: {input_path}"}))
        sys.exit(1)
    
    try:
        # Run inference
        results = model(img)
        
        # Get detection results
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                confidence = box.conf[0].cpu().item()
                class_id = int(box.cls[0].cpu().item())
                class_name = model.names[class_id]
                
                detections.append({
                    "name": class_name,
                    "confidence": confidence,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })
                
                # Draw on image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save image with annotations
        cv2.imwrite(output_path, img)
        
        # Return detections
        print(json.dumps(detections, indent=2))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python detect.py <input_image_path> <output_image_path>"}))
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    detect_objects(input_path, output_path)