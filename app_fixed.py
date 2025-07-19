import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import torch
import sys
import time
import platform
from pathlib import Path

# Fix for PosixPath issue on Windows
if platform.system() == 'Windows':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# Add YOLOv5 directory to path
ROOT_DIR = Path().absolute()  # Use current directory as root
YOLO_DIR = ROOT_DIR / 'yolov5'
sys.path.append(str(YOLO_DIR))

# Import YOLOv5 modules
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = 'object_detection_secret_key'

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(ROOT_DIR, 'results')
STATIC_FOLDER = os.path.join(ROOT_DIR, 'static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Load YOLOv5 model
def load_model():
    # Import download utility
    from utils.downloads import attempt_download
    
    try:
        print("Loading YOLOv5 model...")
        
        # Use the specific path provided for the weights
        custom_model_path = ROOT_DIR / 'Yolov9ObjectDetectionGoogleColab' / 'content' / 'yolov5' / 'runs' / 'train' / 'exp20' / 'weights' / 'best.pt'
        
        # Check if the custom model exists
        if custom_model_path.exists():
            print(f"Found custom model at: {custom_model_path}")
            weights_path = str(custom_model_path)
        else:
            # If the custom model doesn't exist, try an alternative path
            print(f"Custom model not found at: {custom_model_path}")
            weights_path = 'yolov5s.pt'  # Default to YOLOv5s if no custom model found
            print(f"Falling back to default model: {weights_path}")
        
        print(f"Using model: {weights_path}")
        
        # Download the model if it doesn't exist
        weights_path = attempt_download(weights_path)
        print(f"Model path after download: {weights_path}")
        
        # Load model
        print("Selecting device...")
        device = select_device('')  # Use CPU by default, or '0' for GPU if available
        print(f"Using device: {device}")
        
        print("Loading model into memory...")
        model = DetectMultiBackend(weights_path, device=device)
        stride, names, pt = model.stride, model.names, model.pt
        
        print(f"Model loaded successfully. Stride: {stride}")
        print(f"Model classes: {names}")
        
        imgsz = check_img_size((640, 640), s=stride)  # Check image size
        print(f"Input image size: {imgsz}")
        
        # Warmup model
        print("Warming up model...")
        model.warmup(imgsz=(1, 3, *imgsz))
        print("Model warmup complete")
        
        return model, imgsz, stride, names, pt, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

# Global variables for model
model, imgsz, stride, names, pt, device = None, None, None, None, None, None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, output_path, conf_thres=0.25, iou_thres=0.45):
    global model, imgsz, stride, names, pt, device
    
    # Load model if not already loaded
    if model is None:
        model, imgsz, stride, names, pt, device = load_model()
    
    try:
        print(f"Processing image: {image_path}")
        print(f"Confidence threshold: {conf_thres}")
        print(f"IoU threshold: {iou_thres}")
        
        # Load image directly with OpenCV
        img0 = cv2.imread(image_path)
        if img0 is None:
            print(f"Error: Could not read image file {image_path}")
            return None
        
        print(f"Image loaded successfully. Shape: {img0.shape}")
        
        # Prepare image for model
        img = img0.copy()
        
        # Resize image to model input size
        img_resized = cv2.resize(img, (imgsz[0], imgsz[1]))
        print(f"Image resized to: {img_resized.shape}")
        
        # Convert BGR to RGB and transpose to model input format
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
        img_resized = np.ascontiguousarray(img_resized)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).to(device)
        img_tensor = img_tensor.float()  # uint8 to fp16/32
        img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor[None]  # expand for batch dim
        
        print(f"Input tensor shape: {img_tensor.shape}")
        
        # Inference
        print("Running inference...")
        pred = model(img_tensor, augment=False, visualize=False)
        
        # NMS
        print("Applying non-maximum suppression...")
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
        
        # Process detections
        annotator = Annotator(img0, line_width=3, example=str(names))
        
        detection_count = 0
        for i, det in enumerate(pred):  # per image
            if len(det):
                print(f"Found {len(det)} detections")
                detection_count = len(det)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    print(f"Detected: {label} at {xyxy}")
                    annotator.box_label(xyxy, label, color=colors(c, True))
            else:
                print("No detections found")
        
        # Save results
        result_img = annotator.result()
        success = cv2.imwrite(output_path, result_img)
        
        if not success:
            print(f"ERROR: Failed to save result image to {output_path}")
            return None
            
        print(f"Image processing complete: {output_path}")
        print(f"Result file exists: {os.path.exists(output_path)}")
        print(f"Result file size: {os.path.getsize(output_path)} bytes")
        
        if detection_count == 0:
            print("WARNING: No objects were detected. Try lowering the confidence threshold.")
        
        return output_path
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_video(video_path, output_path, conf_thres=0.25, iou_thres=0.45):
    global model, imgsz, stride, names, pt, device
    
    # Load model if not already loaded
    if model is None:
        model, imgsz, stride, names, pt, device = load_model()
    
    print(f"Processing video: {video_path}")
    print(f"Confidence threshold: {conf_thres}")
    print(f"IoU threshold: {iou_thres}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    process_every = 5  # Process every 5th frame for speed
    total_detections = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            process_this_frame = frame_count % process_every == 0
            
            if process_this_frame:
                # Prepare image
                img = frame.copy()
                
                # Resize image to model input size
                img_resized = cv2.resize(img, (imgsz[0], imgsz[1]))
                
                # Convert BGR to RGB and transpose to model input format
                img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
                img_resized = np.ascontiguousarray(img_resized)
                
                # Convert to tensor
                img_tensor = torch.from_numpy(img_resized).to(device)
                img_tensor = img_tensor.float()  # uint8 to fp16/32
                img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor[None]  # expand for batch dim
                
                # Inference
                pred = model(img_tensor, augment=False, visualize=False)
                
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
                
                # Process detections
                for i, det in enumerate(pred):  # per image
                    annotator = Annotator(frame, line_width=3, example=str(names))
                    
                    if len(det):
                        # Count detections
                        total_detections += len(det)
                        
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                        
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            
                            # Print detection info every 100 frames
                            if frame_count % 100 == 0:
                                print(f"Frame {frame_count}: Detected {names[c]} with confidence {conf:.2f}")
                    
                    frame = annotator.result()
            
            # Write frame to output video
            out.write(frame)
            
            # Show progress
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count}/{total_frames}")
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Video processing complete: {output_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections: {total_detections}")
        
        if total_detections == 0:
            print("WARNING: No objects were detected in the video. Try lowering the confidence threshold.")
        
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to release resources even if there's an error
        if cap.isOpened():
            cap.release()
        if out.isOpened():
            out.release()
            
        return None

@app.route('/')
def index():
    return render_template('index.html')

# Add this route for CSS files
@app.route('/css/<path:filename>')
def css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            print(f"File saved to: {file_path}")
            
            # Get confidence threshold from form
            conf_thres = float(request.form.get('confidence', 0.25))
            print(f"Using confidence threshold: {conf_thres}")
            
            # Process file based on type
            file_ext = filename.rsplit('.', 1)[1].lower()
            result_filename = f"result_{int(time.time())}_{filename}"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            print(f"Result will be saved to: {result_path}")
            
            if file_ext in ['jpg', 'jpeg', 'png']:
                print(f"Processing image file: {filename}")
                # Process image
                result = process_image(file_path, result_path, conf_thres=conf_thres)
                file_type = 'image'
            else:
                print(f"Processing video file: {filename}")
                # Process video
                result = process_video(file_path, result_path, conf_thres=conf_thres)
                file_type = 'video'
            
            if result is None:
                flash('Error processing file. Please try again with a different file or lower confidence threshold.')
                return redirect(url_for('index'))
            
            # Verify the result file exists
            if not os.path.exists(result_path):
                print(f"ERROR: Result file was not created: {result_path}")
                flash('Error: Result file was not created. Please try again.')
                return redirect(url_for('index'))
            
            print(f"Result file created successfully: {result_path}")
            print(f"Rendering result template with file: {result_filename}")
            
            return render_template('result.html', 
                                result_file=result_filename, 
                                file_type=file_type)
        
        except Exception as e:
            print(f"Error in upload_file: {e}")
            import traceback
            traceback.print_exc()
            flash(f'Error: {str(e)}')
            return redirect(url_for('index'))
    
    flash('File type not allowed')
    return redirect(url_for('index'))

@app.route('/results/<filename>')
def results(filename):
    print(f"Serving result file: {filename}")
    try:
        return send_from_directory(app.config['RESULTS_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving result file: {e}")
        import traceback
        traceback.print_exc()
        return "File not found", 404

if __name__ == '__main__':
    # Load model at startup
    model, imgsz, stride, names, pt, device = load_model()
    
    # Print model information
    print("Model loaded successfully with", len(names), "classes:")
    for i, name in names.items():
        print(f"  Class {i}: {name}")
    
    # Run the app
    app.run(host='0.0.0.0', debug=True)