import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
import threading
from werkzeug.utils import secure_filename
import requests

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit

# Import required modules from original script
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

try:
    import ailia
    import face_detect_crop
    from face_detect_crop import crop_face, get_kps
    import face_align
    import image_infer
    from image_infer import setup_mxnet, get_landmarks
    from masks import face_mask_static
except ImportError:
    print("Warning: Required modules not found. Make sure all dependencies are installed.")
    pass

# Constants
CROP_SIZE = 224
IOU_DEFAULT = 0.4

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure folders
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model file paths
WEIGHT_G_PATH = os.path.join(MODEL_DIR, 'G_unet_2blocks.onnx')
MODEL_G_PATH = os.path.join(MODEL_DIR, 'G_unet_2blocks.onnx.prototxt')
WEIGHT_ARCFACE_PATH = os.path.join(MODEL_DIR, 'scrfd_10g_bnkps.onnx')
MODEL_ARCFACE_PATH = os.path.join(MODEL_DIR, 'scrfd_10g_bnkps.onnx.prototxt')
WEIGHT_BACKBONE_PATH = os.path.join(MODEL_DIR, 'arcface_backbone.onnx')
MODEL_BACKBONE_PATH = os.path.join(MODEL_DIR, 'arcface_backbone.onnx.prototxt')
WEIGHT_LANDMARK_PATH = os.path.join(MODEL_DIR, 'face_landmarks.onnx')
MODEL_LANDMARK_PATH = os.path.join(MODEL_DIR, 'face_landmarks.onnx.prototxt')
WEIGHT_PIX2PIX_PATH = os.path.join(MODEL_DIR, '10_net_G.onnx')
MODEL_PIX2PIX_PATH = os.path.join(MODEL_DIR, '10_net_G.onnx.prototxt')

# Model download URLs
MODEL_DOWNLOAD_URLS = {
    'G_unet_2blocks.onnx': 'https://storage.googleapis.com/ailia-models/sber-swap/G_unet_2blocks.onnx',
    'G_unet_2blocks.onnx.prototxt': 'https://storage.googleapis.com/ailia-models/sber-swap/G_unet_2blocks.onnx.prototxt',
    'scrfd_10g_bnkps.onnx': 'https://storage.googleapis.com/ailia-models/sber-swap/scrfd_10g_bnkps.onnx',
    'scrfd_10g_bnkps.onnx.prototxt': 'https://storage.googleapis.com/ailia-models/sber-swap/scrfd_10g_bnkps.onnx.prototxt',
    'arcface_backbone.onnx': 'https://storage.googleapis.com/ailia-models/sber-swap/arcface_backbone.onnx',
    'arcface_backbone.onnx.prototxt': 'https://storage.googleapis.com/ailia-models/sber-swap/arcface_backbone.onnx.prototxt',
    'face_landmarks.onnx': 'https://storage.googleapis.com/ailia-models/sber-swap/face_landmarks.onnx',
    'face_landmarks.onnx.prototxt': 'https://storage.googleapis.com/ailia-models/sber-swap/face_landmarks.onnx.prototxt',
    '10_net_G.onnx': 'https://storage.googleapis.com/ailia-models/sber-swap/10_net_G.onnx',
    '10_net_G.onnx.prototxt': 'https://storage.googleapis.com/ailia-models/sber-swap/10_net_G.onnx.prototxt'
}

# Global processing status
processing_status = {
    'is_processing': False,
    'progress': 0,
    'error': None,
    'output_path': None
}

# Loading lock for thread safety
loading_lock = threading.Lock()

# Download status
download_status = {
    'is_downloading': False,
    'progress': 0,
    'current_file': None,
    'error': None
}

def download_file(url, filepath):
    """Download a file with progress tracking"""
    try:
        print(f"Downloading {url} to {filepath}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(block_size):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    # Update progress
                    percentage = int(downloaded / total_size * 100) if total_size > 0 else 0
                    socketio.emit('download_progress', {
                        'filename': os.path.basename(filepath),
                        'percentage': percentage
                    })
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        download_status['error'] = str(e)
        return False

def download_missing_models():
    """Download missing model files"""
    global download_status
    
    download_status.update({
        'is_downloading': True,
        'progress': 0,
        'current_file': None,
        'error': None
    })
    
    missing_files = []
    model_paths = [
        WEIGHT_G_PATH, MODEL_G_PATH, 
        WEIGHT_ARCFACE_PATH, MODEL_ARCFACE_PATH,
        WEIGHT_BACKBONE_PATH, MODEL_BACKBONE_PATH,
        WEIGHT_LANDMARK_PATH, MODEL_LANDMARK_PATH,
        WEIGHT_PIX2PIX_PATH, MODEL_PIX2PIX_PATH
    ]
    
    for file_path in model_paths:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if not missing_files:
        download_status['is_downloading'] = False
        socketio.emit('download_complete', {'message': 'All models already present'})
        return
    
    for filepath in missing_files:
        filename = os.path.basename(filepath)
        if filename in MODEL_DOWNLOAD_URLS:
            download_status['current_file'] = filename
            url = MODEL_DOWNLOAD_URLS[filename]
            
            # Ensure model directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Download file
            if not download_file(url, filepath):
                socketio.emit('download_error', {'error': f'Failed to download {filename}'})
                download_status['is_downloading'] = False
                return
    
    download_status['is_downloading'] = False
    socketio.emit('download_complete', {'message': 'All models downloaded successfully'})

@app.route('/download_models', methods=['POST'])
def start_model_download():
    """Start model file download"""
    if download_status['is_downloading']:
        return jsonify({'error': 'Download already in progress'}), 400
    
    # Start download in background thread
    thread = threading.Thread(target=download_missing_models)
    thread.start()
    
    return jsonify({'status': 'started'})

class FaceSwapProcessor:
    """Face swap processing class"""
    
    def __init__(self, source_path, target_path, use_sr=False, iou=0.4):
        self.source_path = source_path
        self.target_path = target_path
        self.use_sr = use_sr
        self.iou = iou
        self.output_path = None
        self.error = None
        
    def update_progress(self, progress):
        """Update processing progress"""
        processing_status['progress'] = progress
        socketio.emit('progress_update', {'progress': progress})
        
    def preprocess(self, img, half_scale=True):
        """Preprocess image for inference"""
        from PIL import Image
        
        if half_scale:
            im_h, im_w, _ = img.shape
            img = np.array(Image.fromarray(img).resize(
                (im_w // 2, im_h // 2), Image.LANCZOS))

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, net_iface, net_G, src_embeds, tar_img):
        """Perform the face swap prediction"""
        kps = get_kps(tar_img, net_iface, nms_threshold=self.iou)

        if kps is None:
            return None

        M, _ = face_align.estimate_norm(kps[0], CROP_SIZE, mode='None')
        crop_img = cv2.warpAffine(tar_img, M, (CROP_SIZE, CROP_SIZE), borderValue=0.0)

        new_size = (256, 256)
        img = cv2.resize(crop_img, new_size)
        img = self.preprocess(img, half_scale=False)
        img = img.astype(np.float16)

        # Feedforward
        output = net_G.predict([img, src_embeds])
        y_st = output[0]

        y_st = y_st[0].transpose(1, 2, 0)
        y_st = y_st * 127.5 + 127.5
        y_st = np.clip(y_st, 0, 255).astype(np.uint8)

        final_img = cv2.resize(y_st, (CROP_SIZE, CROP_SIZE))

        return final_img, crop_img, M
    
    def face_enhancement(self, net_pix2pix, output):
        """Apply super resolution to face"""
        final_img, crop_img, M = output
        final_img = cv2.resize(final_img, (256, 256))
        final_img = final_img.astype(np.float32)
        final_img = final_img / 255.0
        final_img = np.expand_dims(final_img, axis=0)
        final_img = np.transpose(final_img, (0, 3, 1, 2))
        final_img = net_pix2pix.predict(final_img)
        final_img = final_img * 255
        final_img = np.clip(final_img, 0, 255)
        final_img = final_img.astype(np.uint8)
        final_img = np.transpose(final_img, (0, 2, 3, 1))
        final_img = final_img[0]
        final_img = cv2.resize(final_img, (CROP_SIZE, CROP_SIZE))
        return final_img, crop_img, M
    
    def get_final_img(self, output, tar_img, net_lmk):
        """Create the final image with face mask"""
        final_img, crop_img, tfm = output

        h, w = tar_img.shape[:2]
        final = tar_img.copy()

        landmarks = get_landmarks(net_lmk, final_img)
        landmarks_tgt = get_landmarks(net_lmk, crop_img)

        mask, _ = face_mask_static(crop_img, landmarks, landmarks_tgt, None)
        mat_rev = cv2.invertAffineTransform(tfm)

        swap_t = cv2.warpAffine(final_img, mat_rev, (w, h), borderMode=cv2.BORDER_REPLICATE)
        mask_t = cv2.warpAffine(mask, mat_rev, (w, h))
        mask_t = np.expand_dims(mask_t, 2)

        final = mask_t * swap_t + (1 - mask_t) * final
        final = final.astype(np.uint8)

        return final
    
    def process(self):
        """Run face swap process"""
        try:
            # Check if modules are available
            if 'ailia' not in sys.modules:
                self.error = "Required modules not found. Make sure all dependencies are installed."
                return False
                
            self.update_progress(10)
            
            # Initialize models
            net_iface = ailia.Net(MODEL_ARCFACE_PATH, WEIGHT_ARCFACE_PATH, env_id=0)
            net_back = ailia.Net(MODEL_BACKBONE_PATH, WEIGHT_BACKBONE_PATH, env_id=0)
            net_G = ailia.Net(MODEL_G_PATH, WEIGHT_G_PATH, env_id=0)
            net_lmk = ailia.Net(MODEL_LANDMARK_PATH, WEIGHT_LANDMARK_PATH, env_id=0)
            if self.use_sr:
                net_pix2pix = ailia.Net(MODEL_PIX2PIX_PATH, WEIGHT_PIX2PIX_PATH, env_id=0)
            else:
                net_pix2pix = None
            
            self.update_progress(30)
            
            # Load source image
            if os.name == 'nt':
                # Windows - handle Japanese file paths
                img_data = np.fromfile(self.source_path, dtype=np.uint8)
                src_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            else:
                src_img = cv2.imread(self.source_path, cv2.IMREAD_COLOR)
            
            if src_img is None:
                self.error = f"Failed to load source image: {self.source_path}"
                return False
                
            # Convert to RGB
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            
            # Crop face
            src_img = crop_face(src_img, net_iface, CROP_SIZE, nms_threshold=self.iou)
            
            if src_img is None:
                self.error = "Source face not recognized"
                return False
                
            # Prepare source embeddings
            img = self.preprocess(src_img)
            output = net_back.predict([img])
            src_embeds = output[0]
            src_embeds = src_embeds.astype(np.float16)
            
            self.update_progress(50)
            
            # Load target image
            if os.name == 'nt':
                # Windows - handle Japanese file paths
                img_data = np.fromfile(self.target_path, dtype=np.uint8)
                tar_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            else:
                tar_img = cv2.imread(self.target_path, cv2.IMREAD_COLOR)
                
            if tar_img is None:
                self.error = f"Failed to load target image: {self.target_path}"
                return False
            
            # Run face swap
            output = self.predict(net_iface, net_G, src_embeds, tar_img)
            if output is None:
                self.error = "Target face not recognized"
                return False
                
            self.update_progress(70)
            
            # Apply super resolution if enabled
            if self.use_sr:
                output = self.face_enhancement(net_pix2pix, output)
                
            # Get final image
            res_img = self.get_final_img(output, tar_img, net_lmk)
            
            self.update_progress(90)
            
            # Save the result
            filename = f"swap_{int(time.time())}.png"
            self.output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            
            # Handle Japanese file paths for output
            if os.name == 'nt':
                cv2.imencode('.png', res_img)[1].tofile(self.output_path)
            else:
                cv2.imwrite(self.output_path, res_img)
            
            self.update_progress(100)
            return True
            
        except Exception as e:
            import traceback
            self.error = f"Error: {str(e)}\n\nDebug info:\n{traceback.format_exc()}"
            return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/check_models')
def check_models():
    """Check if model files exist"""
    missing_files = []
    
    model_files = [
        WEIGHT_G_PATH, MODEL_G_PATH, 
        WEIGHT_ARCFACE_PATH, MODEL_ARCFACE_PATH,
        WEIGHT_BACKBONE_PATH, MODEL_BACKBONE_PATH,
        WEIGHT_LANDMARK_PATH, MODEL_LANDMARK_PATH,
        WEIGHT_PIX2PIX_PATH, MODEL_PIX2PIX_PATH
    ]
    
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(os.path.basename(file_path))
            
    return jsonify({
        'all_present': len(missing_files) == 0,
        'missing_files': missing_files
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    try:
        print(f"Request files: {list(request.files.keys())}")
        
        # Get uploaded file (either source or target)
        uploaded_file = None
        file_type = None
        
        if 'source' in request.files:
            uploaded_file = request.files['source']
            file_type = 'source'
        elif 'target' in request.files:
            uploaded_file = request.files['target']
            file_type = 'target'
        else:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        if uploaded_file.filename == '':
            print(f"Empty filename for {file_type}")
            return jsonify({'error': 'No file selected'}), 400
        
        # Ensure file is of supported type
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
        if not uploaded_file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            print(f"Invalid file type: {uploaded_file.filename}")
            return jsonify({'error': f'Invalid file type. Supported types: {ALLOWED_EXTENSIONS}'}), 400
        
        # Save uploaded file with secure filename
        timestamp = int(time.time())
        filename = secure_filename(f"{file_type}_{timestamp}_{uploaded_file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"Saving {file_type} file to: {filepath}")
        
        try:
            uploaded_file.save(filepath)
            print(f"{file_type} file saved successfully")
            
            # Return the path for the uploaded file
            response_data = {}
            if file_type == 'source':
                response_data['source_path'] = filepath
            else:
                response_data['target_path'] = filepath
                
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
    
    except Exception as e:
        print(f"Unexpected error in upload_files: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_face_swap():
    """Process face swap request"""
    global processing_status
    
    # Check if processing is already in progress
    if processing_status['is_processing']:
        return jsonify({'error': 'Processing already in progress'}), 400
    
    data = request.get_json()
    source_path = data.get('source_path')
    target_path = data.get('target_path')
    use_sr = data.get('use_sr', False)
    iou = data.get('iou', IOU_DEFAULT)
    
    if not source_path or not target_path:
        return jsonify({'error': 'Missing file paths'}), 400
    
    # Reset processing status
    processing_status.update({
        'is_processing': True,
        'progress': 0,
        'error': None,
        'output_path': None
    })
    
    # Process in background thread
    def run_processing():
        global processing_status
        processor = FaceSwapProcessor(source_path, target_path, use_sr, iou)
        success = processor.process()
        
        processing_status.update({
            'is_processing': False,
            'progress': 100,
            'error': processor.error,
            'output_path': processor.output_path if success else None
        })
        
        # Emit completion signal
        if success:
            socketio.emit('processing_complete', {
                'output_path': processor.output_path.replace(app.config['OUTPUT_FOLDER'], '/output')
            })
        else:
            socketio.emit('processing_error', {'error': processor.error})
    
    thread = threading.Thread(target=run_processing)
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/status')
def get_status():
    """Get current processing status"""
    return jsonify(processing_status)

@app.route('/output/<filename>')
def serve_output(filename):
    """Serve processed images"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('status_update', processing_status)

@app.route('/test')
def test():
    """Test endpoint to verify server is running properly"""
    return jsonify({'status': 'OK', 'message': 'Server is running'})

if __name__ == '__main__':
    # Create required directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Ensure upload and output directories exist and are writable
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Check directory permissions
    try:
        # Test write permission
        test_file = os.path.join(UPLOAD_FOLDER, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"Upload directory is writable: {UPLOAD_FOLDER}")
    except Exception as e:
        print(f"WARNING: Upload directory is not writable: {UPLOAD_FOLDER}")
        print(f"Error: {e}")
    
    print("Starting SberSwap Web Application...")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Upload directory: {UPLOAD_FOLDER}")
    print(f"Output directory: {OUTPUT_FOLDER}")
    
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')