import sys
import os
import time
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

# Import the required modules from the original script
import numpy as np
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, 'webapp/utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

# We'll need to ensure these modules are in the correct path
# For the packaged application, adjust the paths accordingly
try:
    # Import the necessary modules from the original script
    import ailia
    import face_detect_crop
    from face_detect_crop import crop_face, get_kps
    import face_align
    import image_infer
    from image_infer import setup_mxnet, get_landmarks
    from masks import face_mask_static
except ImportError:
    # If modules are not in path, we'll handle this in the GUI
    print("Warning: Required modules not found. Make sure all dependencies are installed.")
    print("Running script in :", script_dir)
    print("Utils script in :", utils_dir)
    pass

# Constants
CROP_SIZE = 224
IOU_DEFAULT = 0.4

# Default paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


class ImageDropArea(tk.Canvas):
    """Custom Canvas that accepts dropped images and displays them"""
    
    def __init__(self, parent, title="Drop Image Here", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.title = title
        
        # Configure the canvas
        self.config(
            width=250,
            height=250,
            bg='white',
            highlightthickness=2,
            highlightbackground='gray'
        )
        
        # Draw the initial text
        self.create_text(
            self.winfo_reqwidth() // 2,
            self.winfo_reqheight() // 2,
            text=f"{self.title}\n\nDrag & Drop Image Here",
            justify=tk.CENTER
        )
        
        # Initialize variables
        self.image_path = None
        self.image = None
        self.tk_image = None
        
        # Setup drag and drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop)

    def on_drop(self, event):
        """Handle file drop event"""
        # Get the file path (strip braces if they exist)
        file_path = event.data
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
        
        # Handle Japanese file paths properly
        if isinstance(file_path, bytes):
            file_path = file_path.decode('utf-8')
        
        # Check if it's a valid image file
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            messagebox.showerror("Invalid File", "Please drop a valid image file (PNG, JPG, JPEG).")
            return
        
        # Load and display the image
        self.load_image(file_path)
        
    def load_image(self, file_path):
        """Load and display an image from file path"""
        try:
            # Clear the canvas
            self.delete("all")
            
            # Check if file exists
            if not os.path.exists(file_path):
                messagebox.showerror("Error", f"File does not exist: {file_path}")
                return
            
            # Open and resize the image
            try:
                pil_image = Image.open(file_path)
            except UnicodeError as e:
                messagebox.showerror("Error", f"Failed to handle Japanese characters in file path: {str(e)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                return
            
            pil_image = self.resize_image(pil_image)
            
            # Convert to Tkinter image
            self.tk_image = ImageTk.PhotoImage(pil_image)
            
            # Display the image
            self.create_image(
                self.winfo_reqwidth() // 2,
                self.winfo_reqheight() // 2,
                image=self.tk_image,
                anchor=tk.CENTER
            )
            
            # Store the image path
            self.image_path = file_path
            
            # Validate with OpenCV
            # Validate with OpenCV - handle Japanese file paths
            if os.name == 'nt':
                # Windows - use alternative loading method for non-ASCII paths
                test_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                test_img = cv2.imread(file_path)

            if test_img is None:
                messagebox.showwarning("Warning", f"OpenCV cannot read this image file: {file_path}")
                self.image_path = None
                self.delete("all")
                self.create_text(
                    self.winfo_reqwidth() // 2,
                    self.winfo_reqheight() // 2,
                    text=f"{self.title}\n\nInvalid image file",
                    justify=tk.CENTER,
                    fill="red"
                )
            
            # Notify the parent about the image change
            if hasattr(self.parent, 'update_process_button'):
                self.parent.update_process_button()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def resize_image(self, image):
        """Resize the image to fit in the canvas while maintaining aspect ratio"""
        canvas_width = self.winfo_reqwidth()
        canvas_height = self.winfo_reqheight()
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Calculate the scaling factor
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        scale = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        return image.resize((new_width, new_height), Image.LANCZOS)


class ProcessingThread(threading.Thread):
    """Thread for running the face swapping process"""
    
    def __init__(self, source_path, target_path, use_sr=False, iou=0.4, callback=None):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.use_sr = use_sr
        self.iou = iou
        self.callback = callback
        self.output_path = None
        self.error = None
        self.debug_info = []  # For collecting debug information
        
# ProcessingThreadクラスの変更箇所のみ抜粋 

class ProcessingThread(threading.Thread):
    """Thread for running the face swapping process"""
    
    def __init__(self, source_path, target_path, use_sr=False, iou=0.4, callback=None):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        self.use_sr = use_sr
        self.iou = iou
        self.callback = callback
        self.output_path = None
        self.error = None
        self.debug_info = []  # For collecting debug information
        
    def run(self):
        try:
            # Check if the required modules are imported
            if 'ailia' not in sys.modules:
                self.error = "Required modules not found. Make sure all dependencies are installed."
                self._update_progress(100)
                return
                
            self._update_progress(10)
            
            # Initialize models
            net_iface = ailia.Net(MODEL_ARCFACE_PATH, WEIGHT_ARCFACE_PATH, env_id=0)
            net_back = ailia.Net(MODEL_BACKBONE_PATH, WEIGHT_BACKBONE_PATH, env_id=0)
            net_G = ailia.Net(MODEL_G_PATH, WEIGHT_G_PATH, env_id=0)
            net_lmk = ailia.Net(MODEL_LANDMARK_PATH, WEIGHT_LANDMARK_PATH, env_id=0)
            if self.use_sr:
                net_pix2pix = ailia.Net(MODEL_PIX2PIX_PATH, WEIGHT_PIX2PIX_PATH, env_id=0)
            else:
                net_pix2pix = None
            
            self._update_progress(30)

            # Process source image - handle Japanese file paths
            if os.name == 'nt':
                src_img = cv2.imdecode(np.fromfile(self.source_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                src_img = cv2.imread(self.source_path)
            
            if src_img is None:
                self.error = f"Failed to load source image: {self.source_path}. Please check the file path."
                self._update_progress(100)
                return
            
            # Convert color space
            try:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                self.error = f"Failed to convert source image color space: {str(e)}"
                self._update_progress(100)
                return
                
            # Crop face
            src_img = crop_face(src_img, net_iface, CROP_SIZE, nms_threshold=self.iou)
            
            if src_img is None:
                self.error = "Source face not recognized"
                self._update_progress(100)
                return
                
            # Prepare source embeddings - float32を使用
            img = self.preprocess(src_img)
            img = img.astype(np.float16)  # float16からfloat32に変更
            output = net_back.predict([img])
            src_embeds = output[0]
            src_embeds = src_embeds.astype(np.float16)  # float16からfloat32に変更
            
            self._update_progress(50)
            
            # Process target image
            if os.name == 'nt':
                tar_img = cv2.imdecode(np.fromfile(self.target_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                tar_img = cv2.imread(self.target_path)
            
            if tar_img is None:
                self.error = f"Failed to load target image: {self.target_path}. Please check the file path."
                self._update_progress(100)
                return
            
            # 画像のタイプを確認（重要）
            tar_img = cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
            
            # Run face swap
            output = self.predict(net_iface, net_G, src_embeds, tar_img)
            if output is None:
                self.error = "Target face not recognized"
                self._update_progress(100)
                return
                
            self._update_progress(70)
            
            # Apply super resolution if enabled
            if self.use_sr:
                output = self.face_enhancement(net_pix2pix, output)
                
            # Get final image
            res_img = self.get_final_img(output, tar_img, net_lmk)
            
            self._update_progress(90)

            # Convert back to BGR for saving
            res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

            # Save the result - handle Japanese file paths
            filename = f"swap_{int(time.time())}.png"
            self.output_path = os.path.join(OUTPUT_DIR, filename)
            
            if os.name == 'nt':
                cv2.imencode('.png', res_img)[1].tofile(self.output_path)
            else:
                cv2.imwrite(self.output_path, res_img)

            self._update_progress(100)
            
        except Exception as e:
            import traceback
            self.error = f"Error: {str(e)}\n\nDebug info:\n{traceback.format_exc()}"
            self._update_progress(100)

    def predict(self, net_iface, net_G, src_embeds, tar_img):
        """Perform the face swap prediction - 元のコードを維持"""
        kps = get_kps(tar_img, net_iface, nms_threshold=self.iou)

        if kps is None:
            return None

        M, _ = face_align.estimate_norm(kps[0], CROP_SIZE, mode='None')
        crop_img = cv2.warpAffine(tar_img, M, (CROP_SIZE, CROP_SIZE), borderValue=0.0)

        new_size = (256, 256)
        img = cv2.resize(crop_img, new_size)
        img = self.preprocess(img, half_scale=False)
        img = img.astype(np.float16)

        # feedforward
        output = net_G.predict([img, src_embeds])
        y_st = output[0]

        y_st = y_st[0].transpose(1, 2, 0)
        y_st = y_st * 127.5 + 127.5
        y_st = np.clip(y_st, 0, 255).astype(np.uint8)

        final_img = cv2.resize(y_st, (CROP_SIZE, CROP_SIZE))

        return final_img, crop_img, M    

    def face_enhancement(self, net_pix2pix, output):
        """Apply super resolution to face - 元のコードを維持"""
        final_img, crop_img, M = output
        # すでに256x256なので、リサイズを回避
        if final_img.shape[:2] != (256, 256):
            final_img = cv2.resize(final_img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
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
        # より高品質なリサイズ
        if final_img.shape[:2] != (CROP_SIZE, CROP_SIZE):
            final_img = cv2.resize(final_img, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)
        return final_img, crop_img, M

    def get_final_img(self, output, tar_img, net_lmk):
        """Create the final image with face mask - 品質改善のために小さな調整を加える"""
        final_img, crop_img, tfm = output

        h, w = tar_img.shape[:2]
        final = tar_img.copy()

        landmarks = get_landmarks(net_lmk, final_img)
        landmarks_tgt = get_landmarks(net_lmk, crop_img)

        mask, _ = face_mask_static(crop_img, landmarks, landmarks_tgt, None)
        mat_rev = cv2.invertAffineTransform(tfm)

        # より高品質なwarpAffine処理
        swap_t = cv2.warpAffine(final_img, mat_rev, (w, h), 
                               borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_LINEAR)
        mask_t = cv2.warpAffine(mask, mat_rev, (w, h))
        mask_t = np.expand_dims(mask_t, 2)

        final = mask_t * swap_t + (1 - mask_t) * final
        final = final.astype(np.uint8)

        return final

    def preprocess(self, img, half_scale=True):
        """Preprocess image for inference - 品質改善のため、バイキュービック補間を使用"""
        if half_scale:
            im_h, im_w, _ = img.shape
            img = np.array(Image.fromarray(img).resize(
                (im_w // 2, im_h // 2), Image.LANCZOS))

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]

        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        
        return img

    def _update_progress(self, value):
        """Update progress and call callback if provided"""
        if self.callback:
            self.callback(value, self.output_path, self.error)
    
class SberSwapGUI(TkinterDnD.Tk):
    """Main GUI window for SberSwap application"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.source_path = None
        self.target_path = None
        self.output_path = None
        
        # Configure window
        self.title("SberSwap Face Swap")
        self.geometry("900x600")
        self.minsize(800, 600)
        
        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create image frame
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Source image area
        self.source_frame = ttk.LabelFrame(self.image_frame, text="Source Face (The face to use)")
        self.source_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.source_area = ImageDropArea(self, "Source Face")
        self.source_area.pack(in_=self.source_frame, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.source_btn = ttk.Button(self.source_frame, text="Browse...", command=self.browse_source)
        self.source_btn.pack(in_=self.source_frame, pady=(0, 10))
        
        # Target image area
        self.target_frame = ttk.LabelFrame(self.image_frame, text="Target Image (Where to place the face)")
        self.target_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.target_area = ImageDropArea(self, "Target Image")
        self.target_area.pack(in_=self.target_frame, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.target_btn = ttk.Button(self.target_frame, text="Browse...", command=self.browse_target)
        self.target_btn.pack(in_=self.target_frame, pady=(0, 10))
        
        # Result image area
        self.result_frame = ttk.LabelFrame(self.image_frame, text="Result")
        self.result_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        self.result_area = tk.Canvas(
            self,
            width=250,
            height=250,
            bg='white',
            highlightthickness=2,
            highlightbackground='gray'
        )
        self.result_area.pack(in_=self.result_frame, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Draw initial text for result area
        self.result_area.create_text(
            self.result_area.winfo_reqwidth() // 2,
            self.result_area.winfo_reqheight() // 2,
            text="Result will appear here",
            justify=tk.CENTER
        )
        
        self.save_btn = ttk.Button(self.result_frame, text="Save As...", command=self.save_result)
        self.save_btn.pack(in_=self.result_frame, pady=(0, 10))
        
        # Configure the grid columns to be equal
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.columnconfigure(2, weight=1)
        
        # Options frame
        self.options_frame = ttk.LabelFrame(self.main_frame, text="Options")
        self.options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Super resolution checkbox
        self.use_sr_var = tk.BooleanVar(value=False)
        self.use_sr_check = ttk.Checkbutton(
            self.options_frame,
            text="Use Super Resolution",
            variable=self.use_sr_var
        )
        self.use_sr_check.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # IOU slider
        self.iou_frame = ttk.Frame(self.options_frame)
        self.iou_frame.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(self.iou_frame, text="IOU Threshold:").pack(anchor="w")
        
        self.iou_var = tk.DoubleVar(value=IOU_DEFAULT)
        self.iou_slider = ttk.Scale(
            self.iou_frame,
            from_=0.1,
            to=0.9,
            orient=tk.HORIZONTAL,
            length=200,
            variable=self.iou_var,
            command=self.update_iou_label
        )
        self.iou_slider.pack(side=tk.LEFT, padx=(0, 5))
        
        self.iou_label = ttk.Label(self.iou_frame, text=f"{IOU_DEFAULT:.1f}")
        self.iou_label.pack(side=tk.LEFT)
        
        # Process button
        self.process_btn = ttk.Button(
            self.main_frame,
            text="Process",
            command=self.start_processing,
            state=tk.DISABLED
        )
        self.process_btn.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.main_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        # Check for models
        self.check_models()
        
    def check_models(self):
        """Check if model files exist, if not show download instructions"""
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
                
        if missing_files:
            message = "The following model files are missing:\n\n"
            message += "\n".join(missing_files)
            message += "\n\nPlease run 'python setup.py' to download the models."
            
            messagebox.showwarning("Missing Models", message)
    
    def update_iou_label(self, event=None):
        """Update the IOU value label when slider changes"""
        self.iou_label.config(text=f"{self.iou_var.get():.1f}")

    def browse_source(self):
        """Browse for source image"""
        file_path = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            # Ensure proper path handling for Japanese files
            if isinstance(file_path, bytes):
                file_path = file_path.decode('utf-8')
            self.source_area.load_image(file_path)
            self.source_path = file_path
            self.update_process_button()  

    def browse_target(self):
        """Browse for target image"""
        file_path = filedialog.askopenfilename(
            title="Select Target Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            # Ensure proper path handling for Japanese files
            if isinstance(file_path, bytes):
                file_path = file_path.decode('utf-8')
            self.target_area.load_image(file_path)
            self.target_path = file_path
            self.update_process_button()    
    
    def update_process_button(self):
        """Enable process button if both images are set"""
        self.source_path = self.source_area.image_path
        self.target_path = self.target_area.image_path
        
        if self.source_path and self.target_path:
            self.process_btn.config(state=tk.NORMAL)
        else:
            self.process_btn.config(state=tk.DISABLED)
        
    def start_processing(self):
        """Start the face swapping process"""
        if not self.source_path or not self.target_path:
            messagebox.showwarning("Missing Images", "Please select both source and target images.")
            return
            
        # Disable UI during processing
        self.process_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_var.set("Processing...")
        
        # Get parameters
        use_sr = self.use_sr_var.get()
        iou = self.iou_var.get()
        
        # Create processing thread
        self.processing_thread = ProcessingThread(
            self.source_path, 
            self.target_path, 
            use_sr=use_sr,
            iou=iou,
            callback=self.on_processing_update
        )
        
        # Start processing
        self.processing_thread.start()
        
    def on_processing_update(self, progress, output_path, error):
        """Callback for processing thread updates"""
        # Update progress bar
        self.progress_var.set(progress)
        
        # Check if processing is complete
        if progress >= 100:
            if error:
                # Show error
                error_short = error.split('\n')[0]  # Get first line for status
                self.status_var.set(f"Error: {error_short}")
                
                # Show detailed error dialog
                error_dialog = tk.Toplevel(self)
                error_dialog.title("Processing Error")
                error_dialog.geometry("600x400")
                
                # Create scrolled text widget
                text_area = scrolledtext.ScrolledText(error_dialog, wrap=tk.WORD)
                text_area.pack(expand=True, fill='both', padx=10, pady=10)
                text_area.insert(tk.END, error)
                text_area.config(state='disabled')  # Make read-only
                
                # Close button
                close_btn = ttk.Button(error_dialog, text="Close", command=error_dialog.destroy)
                close_btn.pack(pady=(0, 10))
            else:
                # Processing successful
                self.output_path = output_path
                self.status_var.set(f"Processing complete! Result saved to: {output_path}")
                
                # Display result image
                self.display_result_image(output_path)
            
            # Re-enable the process button
            self.process_btn.config(state=tk.NORMAL)
        
    def display_result_image(self, image_path):
        """Display the result image in the result area"""
        # Clear the canvas
        self.result_area.delete("all")
        
        # Load and resize the image
        pil_image = Image.open(image_path)
        
        # Resize to fit the canvas
        canvas_width = self.result_area.winfo_reqwidth()
        canvas_height = self.result_area.winfo_reqheight()
        
        # Get image dimensions
        img_width, img_height = pil_image.size
        
        # Calculate the scaling factor
        width_ratio = canvas_width / img_width
        height_ratio = canvas_height / img_height
        scale = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter image
        self.result_tk_image = ImageTk.PhotoImage(pil_image)
        
        # Display the image
        self.result_area.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.result_tk_image,
            anchor=tk.CENTER
        )

    def save_result(self):
        """Save the result image to a custom location"""
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showwarning("No Result", "No result image to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        
        if file_path:
            # Ensure proper path handling for Japanese files
            if isinstance(file_path, bytes):
                file_path = file_path.decode('utf-8')
            
            # Copy the file
            import shutil
            shutil.copy2(self.output_path, file_path)

def main():
    """Main application entry point"""
    # Create the output and models directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Run the application
    app = SberSwapGUI()
    app.mainloop()


if __name__ == "__main__":
    main()