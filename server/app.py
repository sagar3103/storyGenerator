from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import os
import numpy as np

app = Flask(__name__)

# Configure upload and output folders
UPLOAD_FOLDER = 'server/uploads'
OUTPUT_FOLDER = 'server/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        file.save(input_path)
        
        try:
            # Load image
            img = cv2.imread(input_path)
            if img is None:
                return jsonify({'error': 'Failed to load image'}), 400
            
            # Run inference
            results = model(img)
            
            # Process results
            detections = []
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
                    cv2.putText(img, label, (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save annotated image
            cv2.imwrite(output_path, img)
            
            return jsonify({
                'originalImage': f'http://localhost:5000/uploads/{filename}',
                'processedImage': f'http://localhost:5000/outputs/{filename}',
                'detectedObjects': detections
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True)