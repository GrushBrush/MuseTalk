# -*- coding: utf-8 -*-

# --- Standard Library Imports ---
import argparse
import os
import concurrent.futures # For ThreadPoolExecutor
import threading
import queue
import subprocess
import glob
import pickle
import sys
import copy
import json
import traceback          # For detailed error printing in threads
import shutil
import time
import tempfile

# --- Third-party Library Imports ---
import ffmpeg             # Needs python-ffmpeg (or ffmpeg-python) installed
from omegaconf import OmegaConf
import numpy as np
import cv2                # Needs opencv-python installed
import torch
import torch.nn.functional as F # For interpolate
from tqdm import tqdm
from PIL import Image      # Needs Pillow installed

# --- Local Project Imports (Ensure these paths are correct relative to execution) ---
# Assuming script is run via python -m scripts.realtime_stream_sync from MuseTalk root
try:
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen, load_all_model
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    # Import the specific function needed by prepare_material
    from musetalk.utils.blending import get_image_prepare_material
    # Import FaceParsing if used within get_image_prepare_material/face_seg
    # from face_parsing import FaceParsing # Assuming this is available
except ImportError as e:
    print(f"Error importing local musetalk modules: {e}")
    print("Please ensure the script is run correctly relative to the project root")
    print("and all necessary modules (like face_parsing if used) are installed.")
    sys.exit(1)


# --- PyTorch Device Setup ---
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

print("--- PyTorch Device Information ---")
if cuda_available:
    print("✅ CUDA (GPU) detected by PyTorch.")
    try: gpu_name = torch.cuda.get_device_name(0); print(f"✅ Using GPU: {gpu_name}")
    except Exception as e: print(f"⚠️ Could not retrieve GPU name: {e}")
    print(f"✅ Selected device: {device}")
else:
    print("❌ CUDA (GPU) not available or not detected by PyTorch.")
    print(f"✅ Selected device: {device}")
print("-------------------------------")

# --- Load Models ---
print("Loading models...")
try:
    audio_processor, vae, unet, pe = load_all_model()
    print("Models loaded.")
except Exception as e_load:
    print(f"FATAL ERROR loading models: {e_load}")
    traceback.print_exc()
    sys.exit(1)


# --- Set Precision/Device on Sub-modules ---
print("Setting model precision/device on sub-modules...")
try:
    # Check and move/convert PE
    if hasattr(pe, 'to'): pe = pe.to(device)
    if hasattr(pe, 'half'): pe = pe.half()
    else: print("Warning: 'pe' object doesn't have .half() method.")

    # Check and move/convert VAE sub-module
    if hasattr(vae, 'vae'):
        if hasattr(vae.vae, 'to'): vae.vae = vae.vae.to(device)
        if hasattr(vae.vae, 'half'): vae.vae = vae.vae.half()
        else: print("Warning: 'vae.vae' has no .half() method.")
    else: print("Warning: Cannot access 'vae.vae'. VAE structure might differ.")

    # Check and move/convert UNet sub-module
    if hasattr(unet, 'model'):
        if hasattr(unet.model, 'to'): unet.model = unet.model.to(device)
        if hasattr(unet.model, 'half'): unet.model = unet.model.half()
        else: print("Warning: 'unet.model' has no .half() method.")
    else: print("Warning: Cannot access 'unet.model'. UNet structure might differ.")

    print("Precision set (check warnings above).")
except Exception as e_prec: print(f"Warning: Error setting precision/device: {e_prec}")

timesteps = torch.tensor([0], device=device)

# --- FFmpeg Audio Reader Class ---
class FFmpegAudioReader:
    def __init__(self, audio_file): self.audio_file = audio_file; self._probe()
    def _probe(self):
        try:
            print(f"Probing audio: {self.audio_file}"); probe = ffmpeg.probe(self.audio_file)
            a_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            if a_stream is None: raise RuntimeError("No audio stream found")
            self.sample_rate = int(a_stream.get('sample_rate', '48000'))
            self.channels = int(a_stream.get('channels', '2'))
            print(f" Audio Probe OK: SR={self.sample_rate}, Ch={self.channels}")
        except Exception as e: print(f"❌ Probe Error: {e}"); raise
    def read_full_audio(self):
        print("Reading/Converting audio (ffmpeg -> 48k Hz, Stereo, s16le)...")
        sr, ac, fmt = 48000, 2, "s16le"
        try:
            cmd = ["ffmpeg", "-i", self.audio_file, "-f", fmt, "-ac", str(ac), "-ar", str(sr), "-", "-hide_banner", "-loglevel", "error"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate(); rc = process.poll(); stderr_str = err.decode(errors='ignore')
            if rc != 0: print(f"❌ FFmpeg failed {rc}\n{stderr_str}"); return None
            if not out: print(f"❌ FFmpeg no data\n{stderr_str}"); return None
            audio_data = np.frombuffer(out, dtype=np.int16).reshape(-1, ac)
            print(f"✅ Read OK: {len(audio_data)} samples")
            return audio_data
        except FileNotFoundError: print("❌ Error: ffmpeg not found in PATH."); return None
        except Exception as e: print(f"❌ FFmpeg Read Error: {e}\nstderr:\n{stderr_str}"); return None

# --- Audio Splitting Function ---
def split_audio(audio_data, num_chunks):
    if audio_data is None or num_chunks <= 0: return []
    if num_chunks == 1: return [audio_data]
    print(f"Splitting {len(audio_data)} samples into {num_chunks} chunks...")
    chunks = np.array_split(audio_data, num_chunks, axis=0)
    total_samples_in_chunks = sum(len(c) for c in chunks)
    print(f"✅ Split OK: {len(chunks)} chunks. Samples: {total_samples_in_chunks}/{len(audio_data)}")
    if total_samples_in_chunks != len(audio_data): print("Error: Sample mismatch after split!")
    return chunks

# --- GStreamer Pipeline Classes ---
class GStreamerPipeline:
    def __init__(self, width=720, height=1280, fps=25, host="127.0.0.1", port=5000):
        self.width, self.height, self.fps, self.host, self.port = width, height, fps, host, port; self.process = None
        pipeline_str_fdsrc = (
            f"fdsrc fd=0 ! "
            f"videoparse format=bgr width={self.width} height={self.height} framerate={self.fps}/1 ! "
            "queue max-size-buffers=10 ! videoconvert ! video/x-raw,format=NV12 ! "
            "queue max-size-buffers=10 ! cudaupload ! "
            "nvh265enc bitrate=20000 bframes=0 preset=p1 gop-size=3 ! " # bitrate in kbps
            "h265parse config-interval=-1 ! "
            "rtph265pay pt=96 ! "
            f"udpsink host={self.host} port={self.port} sync=true buffer-size=655360"
        )
        print("Starting GStreamer video pipeline...")
        try:
            self.process = subprocess.Popen(f"gst-launch-1.0 -v {pipeline_str_fdsrc}", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0)
            print(f"✅ GStreamer video pipeline started (PID: {self.process.pid})")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_VID_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_VID_ERR"), daemon=True).start()
        except Exception as e: print(f"❌ Failed to start GStreamer video: {e}"); self.process = None
    def _log_stream(self, stream, prefix):
        try:
            for line in iter(stream.readline, b''): print(f"[{prefix}]: {line.decode(errors='ignore').strip()}")
        except: pass; stream.close()
    def send_frame(self, frame):
        if not self.process or self.process.stdin.closed: return False
        try:
            if frame.shape[0]!=self.height or frame.shape[1]!=self.width or frame.dtype!=np.uint8: print(f"Error: send_frame invalid! {frame.shape}, {frame.dtype}"); return False
            self.process.stdin.write(frame.tobytes()); self.process.stdin.flush(); return True
        except BrokenPipeError: print("❌ Error send video: Broken pipe."); self.process = None; return False
        except Exception as e: print(f"❌ Error send video: {e}"); return False
    def stop(self):
        print("Stopping GStreamer video pipeline..."); proc = self.process
        if proc:
            self.process = None;
            try:
                if proc.stdin and not proc.stdin.closed: proc.stdin.close()
            except: pass
            try: proc.wait(timeout=5); print("✅ GStreamer video terminated.")
            except subprocess.TimeoutExpired: print("⚠️ Killing GStreamer video..."); proc.kill(); proc.wait(); print("✅ GStreamer video killed.")
            except Exception as e: print(f"Error waiting video: {e}")
        else: print("Video pipeline not running.")

class GStreamerAudio:
    def __init__(self, host="127.0.0.1", port=5001):
        self.host, self.port, self.process = host, port, None
        pipeline_str = (
            f"fdsrc fd=0 ! audio/x-raw,format=S16LE,channels=2,rate=48000,layout=interleaved ! "
            f"queue max-size-time=3000000000 ! audioconvert ! audioresample ! queue ! "
            f"opusenc bitrate=64000 ! rtpopuspay pt=97 ! "
            f"udpsink host={self.host} port={self.port} sync=true buffer-size=65536"
        )
        print("Starting GStreamer audio pipeline...")
        try:
            self.process = subprocess.Popen(f"gst-launch-1.0 -v {pipeline_str}", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0)
            print(f"✅ GStreamer audio pipeline started (PID: {self.process.pid})")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_AUD_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_AUD_ERR"), daemon=True).start()
        except Exception as e: print(f"❌ Failed to start GStreamer audio: {e}"); self.process = None
    def _log_stream(self, stream, prefix):
        try:
            for line in iter(stream.readline, b''): print(f"[{prefix}]: {line.decode(errors='ignore').strip()}")
        except: pass; stream.close()
    def send_audio(self, audio_data):
        if not self.process or self.process.stdin.closed: return False
        try:
            if audio_data.dtype != np.int16: audio_data = audio_data.astype(np.int16)
            self.process.stdin.write(audio_data.tobytes()); self.process.stdin.flush(); return True
        except BrokenPipeError: print("❌ Error send audio: Broken pipe."); self.process = None; return False
        except Exception as e: print(f"❌ Error send audio: {e}"); return False
    def stop(self):
        print("Stopping GStreamer audio pipeline..."); proc = self.process
        if proc:
            self.process = None;
            try:
                 if proc.stdin and not proc.stdin.closed: proc.stdin.close()
            except: pass
            try: proc.wait(timeout=5); print("✅ GStreamer audio terminated.")
            except subprocess.TimeoutExpired: print("⚠️ Killing GStreamer audio..."); proc.kill(); proc.wait(); print("✅ GStreamer audio killed.")
            except Exception as e: print(f"Error waiting audio: {e}")
        else: print("Audio pipeline not running.")

# --- Misc Helper Functions ---
def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    print(f"Extracting frames from {vid_path} to {save_path}...")
    cap = cv2.VideoCapture(vid_path); count = 0; frame_idx = 0; frame_interval = 1
    if not cap.isOpened(): print(f"Error opening video: {vid_path}"); return
    while count < cut_frame:
        ret, frame = cap.read()
        if not ret: print("Video ended."); break
        if frame_idx % frame_interval == 0:
             filepath = os.path.join(save_path, f"{count:08d}{ext}")
             try: cv2.imwrite(filepath, frame); count += 1
             except Exception as e: print(f"Error write frame {count}: {e}"); break
        frame_idx += 1
    cap.release(); print(f"Finished extracting {count} frames.")

def osmakedirs(path_list):
    for path in path_list:
        try: os.makedirs(path, exist_ok=True)
        except Exception as e: print(f"Error creating dir {path}: {e}")


# --- GPU Worker Function ---
# (Keep process_single_frame_gpu function from previous response)
def process_single_frame_gpu(args):
    """Worker function: Uploads data, Resizes/Blends/Resizes on GPU."""
    self, i, start_idx, res_frame_gpu_nchw, target_w, target_h, device = args
    # res_frame_gpu_nchw is expected [C, H, W], RGB, float32 on GPU

    try:
        # === 1. Get CPU data ===
        current_list_len = len(self.coord_list_cycle)
        if current_list_len == 0: return i, None
        current_idx = (start_idx + i) % current_list_len
        if current_idx >= len(self.coord_list_cycle) or current_idx >= len(self.frame_list_cycle) or \
           current_idx >= len(self.mask_list_cycle) or current_idx >= len(self.mask_coords_list_cycle):
            # print(f"Warn(GPU): Index {current_idx} OOB. Skip {i}.") # Reduce noise
            return i, None
        bbox = self.coord_list_cycle[current_idx]; ori_frame_ref = self.frame_list_cycle[current_idx]
        mask_ref = self.mask_list_cycle[current_idx]; crop_box = self.mask_coords_list_cycle[current_idx] # Load crop_box for body slicing
        x, y, x1, y1 = bbox
        if ori_frame_ref is None or mask_ref is None or crop_box is None: return i, None
        if not isinstance(ori_frame_ref, np.ndarray) or not isinstance(mask_ref, np.ndarray): return i, None

        with torch.no_grad():
            # === 2. Upload CPU data & Prepare ===
            ori_frame_gpu = torch.from_numpy(cv2.cvtColor(ori_frame_ref, cv2.COLOR_BGR2RGB)).to(device, non_blocking=True).float().div_(255.0)
            h_body, w_body = ori_frame_gpu.shape[:2] # Frame dimensions

            # Mask: Ensure grayscale, then Gray uint8 NumPy -> Gray float32 Tensor GPU [H_mask, W_mask]
            mask_gray_ref = mask_ref
            if mask_gray_ref.ndim == 3: mask_gray_ref = cv2.cvtColor(mask_gray_ref, cv2.COLOR_BGR2GRAY)
            elif mask_gray_ref.ndim != 2: print(f"Error(GPU_Worker): mask unexpected dims: {mask_gray_ref.shape}. Skip {i}."); return i, None
            mask_gpu = torch.from_numpy(mask_gray_ref.astype(np.float32)).to(device, non_blocking=True).div_(255.0)
            h_mask_orig, w_mask_orig = mask_gpu.shape # Get original loaded mask dimensions

            # Input Face Prep
            if res_frame_gpu_nchw.ndim != 3 or res_frame_gpu_nchw.shape[0] != 3: return i, None
            if res_frame_gpu_nchw.device.type != device.type: res_frame_gpu_nchw = res_frame_gpu_nchw.to(device)
            if res_frame_gpu_nchw.dtype != torch.float32: res_frame_gpu_nchw = res_frame_gpu_nchw.float()

            # === 3. GPU Resize 1 (Face) ===
            face_tgt_h, face_tgt_w = y1 - y, x1 - x
            if face_tgt_w <= 0 or face_tgt_h <= 0: return i, None
            resized_face_nchw = F.interpolate(res_frame_gpu_nchw.unsqueeze(0), size=(face_tgt_h, face_tgt_w), mode='bilinear', align_corners=False)
            resized_face_gpu = resized_face_nchw.squeeze(0).permute(1, 2, 0) # HWC RGB float32

            # === 4. GPU Blending ===
            x_s, y_s, x_e, y_e = crop_box # Use the clamped box loaded from prep

            # --- Clamp crop box coords based on BODY dimensions ---
            x_s_clamped_body, y_s_clamped_body = max(0, x_s), max(0, y_s)
            x_e_clamped_body, y_e_clamped_body = min(w_body, x_e), min(h_body, y_e)
            if x_s_clamped_body >= x_e_clamped_body or y_s_clamped_body >= y_e_clamped_body:
                 print(f"Warn(GPU): BODY slice zero dim after clamping crop_box {crop_box} to body dims {(h_body, w_body)}. Skip {i}.")
                 return i, None

            # --- Slice body and create face_large using body-clamped coords ---
            body_slice_gpu = ori_frame_gpu[y_s_clamped_body:y_e_clamped_body, x_s_clamped_body:x_e_clamped_body]
            face_large_gpu = body_slice_gpu.clone(); h_large, w_large = face_large_gpu.shape[:2]
            if h_large <= 0 or w_large <= 0: # Double check face_large dims
                 print(f"Warn(GPU): face_large_gpu has zero dim {(h_large, w_large)}. Skip {i}.")
                 return i, None

            # --- Paste face (coords relative to face_large/body_slice) ---
            y_start_paste = y - y_s_clamped_body; y_end_paste = y1 - y_s_clamped_body
            x_start_paste = x - x_s_clamped_body; x_end_paste = x1 - x_s_clamped_body
            y_start_paste, y_end_paste = max(0, y_start_paste), min(h_large, y_end_paste)
            x_start_paste, x_end_paste = max(0, x_start_paste), min(w_large, x_end_paste)
            if y_start_paste >= y_end_paste or x_start_paste >= x_end_paste: return i, None
            slice_h = y_end_paste - y_start_paste; slice_w = x_end_paste - x_start_paste
            if resized_face_gpu.shape[0] != slice_h or resized_face_gpu.shape[1] != slice_w:
                try: face_nchw = resized_face_gpu.permute(2, 0, 1).unsqueeze(0); face_resized_again_nchw = F.interpolate(face_nchw, size=(slice_h, slice_w), mode='bilinear', align_corners=False); resized_face_gpu = face_resized_again_nchw.squeeze(0).permute(1, 2, 0)
                except Exception as e_resize_again: print(f"Err resize face: {e_resize_again}"); return i, None
            face_large_gpu[y_start_paste:y_end_paste, x_start_paste:x_end_paste] = resized_face_gpu

            # ===>>> FIX: Resize the *entire* loaded mask to match face_large dims <<<===
            mask_region_gpu = mask_gpu # Start with the full mask tensor [H_mask, W_mask]
            if mask_region_gpu.shape[0] != h_large or mask_region_gpu.shape[1] != w_large:
                 # print(f"Debug(GPU_Worker): Resizing full mask {mask_region_gpu.shape} to face_large {(h_large, w_large)}")
                 try:
                     # Reshape to NCHW (N=1, C=1) for interpolate
                     mask_for_resize = mask_region_gpu.view(1, 1, *mask_region_gpu.shape)
                     mask_region_gpu = F.interpolate(mask_for_resize, size=(h_large, w_large), mode='bilinear', align_corners=False).squeeze() # Back to HW [h_large, w_large]
                 except Exception as e_mask_resize:
                     print(f"Error(GPU_Worker): Failed resizing full mask: {e_mask_resize}"); return i, None
            # ===>>> END FIX <<<===

            mask_expanded_gpu = mask_region_gpu.unsqueeze(-1) # Add channel dim -> [h_large, w_large, 1]

            # Blend on GPU
            blended_region_gpu = face_large_gpu * mask_expanded_gpu + \
                                 body_slice_gpu * (1.0 - mask_expanded_gpu)
            combine_frame_gpu = ori_frame_gpu # Operate in-place on the full frame copy
            combine_frame_gpu[y_s_clamped_body:y_e_clamped_body, x_s_clamped_body:x_e_clamped_body] = blended_region_gpu # Use body-clamped coords

            # === 5. GPU Resize 2 ===
            combine_frame_nchw = combine_frame_gpu.permute(2, 0, 1).unsqueeze(0)
            final_resized_nchw = F.interpolate(combine_frame_nchw, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # === 6. Download and Convert Back ===
            final_resized_hwc_gpu = final_resized_nchw.squeeze(0).permute(1, 2, 0)
            final_np_rgb_uint8 = (torch.clamp(final_resized_hwc_gpu * 255.0, 0, 255)).cpu().byte().numpy()
            final_np_bgr_uint8 = cv2.cvtColor(final_np_rgb_uint8, cv2.COLOR_RGB2BGR)

            return i, final_np_bgr_uint8
    except Exception as e_worker: print(f"GPU Worker Err frame {i}:"); traceback.print_exc(); return i, None


# --- Main Avatar Class ---
@torch.no_grad()
class Avatar:
    # --- __init__, init, _reload_prepared_data, prepare_material (Keep from previous response) ---
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        print(f"Initializing Avatar: {avatar_id}")
        self.avatar_id = avatar_id; self.video_path = video_path; self.bbox_shift = bbox_shift
        self.batch_size = batch_size; self.preparation = preparation
        self.avatar_path = f"./results/avatars/{avatar_id}"; self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"; self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"; self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"; self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = { "avatar_id": avatar_id, "video_path": video_path, "bbox_shift": bbox_shift }
        self.idx = 0; self.input_latent_list_cycle = None; self.coord_list_cycle = None
        self.frame_list_cycle = None; self.mask_coords_list_cycle = None; self.mask_list_cycle = None
        self.init(); print(f"Avatar initialization complete for {avatar_id}.")

    def init(self):
        # (Keep init method from previous response)
        if self.preparation:
            if os.path.exists(self.avatar_path):
                try:
                    resp = input(f"'{self.avatar_id}' exists. Re-create? (y/n): ")
                    if resp.lower() == "y": print(f"Removing: {self.avatar_path}"); shutil.rmtree(self.avatar_path); self.prepare_material()
                    else: print("Loading existing data..."); self._reload_prepared_data()
                except: print("Error during input. Loading..."); self._reload_prepared_data()
            else: print(f"Path {self.avatar_path} not found. Preparing..."); self.prepare_material()
        else:
             req = [self.coords_path, self.latents_out_path, self.mask_coords_path, self.mask_out_path, self.full_imgs_path]
             if not all(os.path.exists(p) for p in req): print(f"Error: Data missing in {self.avatar_path}. Prep=False."); print("Missing:", [p for p in req if not os.path.exists(p)]); sys.exit(1)
             else: print("Loading existing data..."); self._reload_prepared_data() # Add consistency check back if needed

    def _reload_prepared_data(self):
        """Helper to reload prepared data into instance variables."""
        print("Reloading prepared data..."); all_ok = True
        try:
            # --- Load data saved as files ---
            print(f" Loading latents: {self.latents_out_path}")
            self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location='cpu')
            # If latents were saved as a stacked tensor, convert back to list
            if isinstance(self.input_latent_list_cycle, torch.Tensor):
                self.input_latent_list_cycle = list(self.input_latent_list_cycle)
            print(f" Latents OK ({len(self.input_latent_list_cycle)} items)")

            print(f" Loading coords: {self.coords_path}")
            with open(self.coords_path, 'rb') as f: self.coord_list_cycle = pickle.load(f)
            print(f" Coords OK ({len(self.coord_list_cycle)} items)")

            print(f" Loading mask coords: {self.mask_coords_path}")
            with open(self.mask_coords_path, 'rb') as f: self.mask_coords_list_cycle = pickle.load(f)
            print(f" MaskCoords OK ({len(self.mask_coords_list_cycle)} items)")

            # --- Load base frames and masks from image files ---
            print(f" Loading base frames from: {self.full_imgs_path}")
            input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jp][pn]g')))
            if not input_img_list: raise FileNotFoundError(f"No base images found in {self.full_imgs_path}")
            # Sort numerically based on filename (00000000.png, etc.)
            try:
                 input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except ValueError: print("Warning: Could not sort base images numerically.")
            # Read the base frames (these are the original N frames)
            base_frames = read_imgs(input_img_list)
            print(f" Base Frames OK ({len(base_frames)} items)")
            if not base_frames: raise ValueError("Failed to read base frames.")

            print(f" Loading masks from: {self.mask_out_path}")
            input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jp][pn]g')))
            if not input_mask_list: raise FileNotFoundError(f"No masks found in {self.mask_out_path}")
            # Sort numerically based on cycle index in filename (00000000.png to 00001099.png)
            try:
                 input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except ValueError: print("Warning: Could not sort masks numerically.")
            # Read the cycle masks (should be 2N)
            self.mask_list_cycle = read_imgs(input_mask_list)
            print(f" Masks OK ({len(self.mask_list_cycle)} items)")
            if not self.mask_list_cycle: raise ValueError("Failed to read masks.")

            # === FIX: Reconstruct frame_list_cycle ===
            print("Reconstructing frame cycle...")
            self.frame_list_cycle = base_frames + base_frames[::-1]
            print(f" Frame Cycle OK ({len(self.frame_list_cycle)} items)")
            # === END FIX ===

            # --- Validation ---
            print("Validating loaded data lengths...")
            data_lists = {
                "Coords": self.coord_list_cycle, "Frames": self.frame_list_cycle,
                "Masks": self.mask_list_cycle, "MaskCoords": self.mask_coords_list_cycle,
                "Latents": self.input_latent_list_cycle
            }
            list_lengths = {}
            ok = True
            for name, data_list in data_lists.items():
                 if data_list is None or not isinstance(data_list, list) or len(data_list) == 0:
                      print(f"Error: List '{name}' invalid after loading/reconstruction."); ok = False; break
                 list_lengths[name] = len(data_list)

            # Check if all lengths are the same
            if ok and len(set(list_lengths.values())) > 1:
                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 print("Error: Lengths of reloaded data lists do not match!")
                 print(f" Lengths: {list_lengths}")
                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 ok = False

            if not ok: raise ValueError("Failed load/validate data due to length mismatch or missing data.")

            print(f"Successfully reloaded and validated {list_lengths.get('Frames', 0)} frames/etc.")

        except Exception as e:
            print(f"Error reloading prepared data: {e}")
            traceback.print_exc()
            # Re-raise as SystemExit to be caught by __main__
            raise SystemExit(f"Failed reload: {e}")
    def prepare_material(self):
        """Prepares all necessary data for the avatar."""
        print(f"--- Preparing material for {self.avatar_id} ---"); osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
        with open(self.avatar_info_path, "w") as f: json.dump(self.avatar_info, f)

        # --- Step 1 & 2: Get Frames & Landmarks ---
        # (Keep code from previous response)
        if os.path.isfile(self.video_path): video2imgs(self.video_path, self.full_imgs_path, ext='.png')
        elif os.path.isdir(self.video_path):
            print(f"Copying frames from: {self.video_path}"); img_ext = ('.png', '.jpg', '.jpeg'); files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(img_ext)])
            if not files: print(f"Error: No images in {self.video_path}"); sys.exit(1)
            for i, fname in enumerate(tqdm(files, desc="Copying")): shutil.copyfile(os.path.join(self.video_path, fname), os.path.join(self.full_imgs_path, f"{i:08d}.png"))
            print(f"Copied {len(files)} frames.")
        else: print(f"Error: video_path '{self.video_path}' invalid."); sys.exit(1)
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))
        try: input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError: print("Warning: Could not sort image list numerically.")
        if not input_img_list: print(f"Error: No PNGs found in {self.full_imgs_path}"); sys.exit(1)
        print("Extracting landmarks..."); coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift); print(f" Found {len(coord_list)} coords, {len(frame_list)} frames.")

        # --- Step 3: VAE Encoding ---
        # (Keep VAE loop from previous response)
        print("Encoding latents..."); input_latent_list, valid_coords, valid_frames, valid_indices = [], [], [], []
        coord_placeholder_val = coord_placeholder # Assign tuple
        global vae; assert 'vae' in globals() and vae is not None, "VAE not loaded"
        for i, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list), desc="VAE Encoding")):
            crop = None; resized = None; latents = None; y1i, y2i, x1i, x2i = -1, -1, -1, -1
            try:
                 if bbox is None or bbox == coord_placeholder_val or frame is None: continue
                 x1c, y1c, x2c, y2c = bbox
                 y1i, y2i, x1i, x2i = int(y1c), int(y2c), int(x1c), int(x2c); hf, wf = frame.shape[:2]
                 y1i, y2i = max(0, y1i), min(hf, y2i); x1i, x2i = max(0, x1i), min(wf, x2i)
                 if x1i >= x2i or y1i >= y2i: continue
                 crop = frame[y1i:y2i, x1i:x2i]
                 if crop.size == 0: continue # Check after assignment
                 resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                 latents = vae.get_latents_for_unet(resized)
                 input_latent_list.append(latents); valid_coords.append(bbox); valid_frames.append(frame); valid_indices.append(i)
            except Exception as e:
                 print(f"--- VAE Error Details (Frame {i}) ---"); print(f" BBox: {bbox}"); print(f" Frame Shape: {frame.shape if frame is not None else 'None'}")
                 print(f" Coords (int): {(x1i, y1i, x2i, y2i)}"); print(f" Crop exists: {crop is not None}"); print(f" Exception Type: {type(e)}"); print(f" Exception Args: {e.args}")
                 traceback.print_exc(); print(f"--- End Error Details ---"); continue
        if not input_latent_list: print("Error: No valid latents generated."); sys.exit(1)
        print(f"Generated {len(input_latent_list)} latents.")

        # --- Step 4: Create Cycles ---
        frame_cycle = valid_frames + valid_frames[::-1]; coord_cycle = valid_coords + valid_coords[::-1]; latent_cycle = input_latent_list + input_latent_list[::-1]
        cycle_len = len(frame_cycle); print(f"Cycle length: {cycle_len}.")

        # --- Step 5: Mask Generation ---
        print("Generating masks..."); mask_coords_cycle, mask_cycle = [], []; global get_image_prepare_material; assert 'get_image_prepare_material' in globals(), "Func not found"
        proc_idx = set(); temp_mask = {}
        for i, frame in enumerate(tqdm(frame_cycle, desc="Masks")):
            bbox = coord_cycle[i]
            mask, crop_box = None, None
            try:
                 # Call the function
                 mask, crop_box = get_image_prepare_material(frame, bbox)

                 # Check the return values
                 if mask is None or crop_box is None:
                      # print(f"Debug Frame {i}: Skipping because mask ({mask is None}) or crop_box ({crop_box is None}) is None.") # Reduce noise
                      continue # Skip if function failed

                 # If function returned valid data, try to write and add
                 # print(f"Debug Frame {i}: Proceeding to write/add.") # Reduce noise

                 # ===>>> NESTED TRY/EXCEPT FOR IMWRITE <<<===
                 mask_write_success = False
                 try:
                     save_path = f"{self.mask_out_path}/{str(i).zfill(8)}.png"
                     # print(f"Debug Frame {i}: Attempting imwrite to {save_path}") # Optional debug
                     cv2.imwrite(save_path, mask)
                     mask_write_success = True # Mark as successful if no exception
                     # print(f"Debug Frame {i}: imwrite successful.") # Optional debug
                 except Exception as e_write:
                     print(f"!!!!! ERROR cv2.imwrite failed for frame {i} !!!!!")
                     print(f" Path: {save_path}")
                     print(f" Mask type: {type(mask)}, Mask shape: {getattr(mask, 'shape', 'N/A')}, Mask dtype: {getattr(mask, 'dtype', 'N/A')}")
                     traceback.print_exc()
                     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                     continue # Skip adding this frame if write fails
                 # ===>>> END NESTED TRY/EXCEPT <<<===

                 # Only add if imwrite succeeded
                 if mask_write_success:
                     temp_mask[i] = (mask, crop_box)
                     proc_idx.add(i)
                     # print(f"Debug Frame {i}: Added to processed_indices.") # Optional debug

            except Exception as e: # Catch errors from get_image_prepare_material itself if any slip through
                 print(f"--- Mask Gen Outer Loop Error (Frame {i}) ---")
                 print(f" BBox: {bbox}")
                 print(f" Exception Type: {type(e)}")
                 traceback.print_exc()
                 print(f"--- End Outer Loop Error Details ---")
                 continue # Skip frame on error

        # --- Step 6 & 7 (Keep from previous response) ---
        # (Filter Cycles, Final Save)
        print(f"Filtering cycles ({len(proc_idx)} masks)..."); fin_frame, fin_coord, fin_latent, fin_mask, fin_mask_coord = [], [], [], [], []
        for i in range(cycle_len):
             if i in proc_idx:
                  frame_img = frame_cycle[i]; mpath = os.path.join(self.mask_out_path, f"{i:08d}.png")
                  if os.path.exists(mpath): # Check if mask file was actually written
                       m_img = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
                       if frame_img is not None and m_img is not None:
                           fin_frame.append(frame_img); fin_coord.append(coord_cycle[i]); fin_latent.append(latent_cycle[i])
                           m_data, cbox_data = temp_mask[i]; fin_mask.append(m_data); fin_mask_coord.append(cbox_data)
                       # else: print(f"Warn: Read fail {i}") # Reduce noise
                  # else: print(f"Warn: Mask file missing {i}") # Reduce noise
        print("Saving prepared data..."); fin_len = len(fin_frame)
        if not all(len(lst) == fin_len for lst in [fin_coord, fin_latent, fin_mask, fin_mask_coord]): print("Error: Final lengths mismatch!"); sys.exit(1)
        if fin_len == 0: print("Error: No frames survived prep."); sys.exit(1)
        with open(self.coords_path, 'wb') as f: pickle.dump(fin_coord, f)
        with open(self.mask_coords_path, 'wb') as f: pickle.dump(fin_mask_coord, f)
        if isinstance(fin_latent[0], torch.Tensor): fin_latent = torch.stack(fin_latent)
        torch.save(fin_latent, self.latents_out_path); print("--- Prep Complete ---"); self._reload_prepared_data()

    # ========================================================================
    # === process_frames method - Calls GPU worker ===========================
    # ========================================================================
    def process_frames(self, res_frame_queue, video_len, gst_video_pipeline, gst_audio_pipeline, skip_save_images, debug=False):
        # (Keep process_frames method from previous response - calls process_single_frame_gpu)
        print(f"Target video length (batches): {video_len}")
        num_workers = max(1, os.cpu_count() - 2) # Use multiple cores
        print(f"--- Starting process_frames with {num_workers} workers (GPU Processing) ---")
        global device # Ensure device is accessible

        if not hasattr(self, 'coord_list_cycle') or not self.coord_list_cycle or len(self.coord_list_cycle) == 0:
             print("Error: Avatar reference data not loaded/empty before process_frames."); return

        batch_counter = 0; total_frames_sent = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            while True:
                current_loop_start_idx = self.idx
                if batch_counter >= video_len: print(f"Processed target batches ({batch_counter}/{video_len})."); break
                try: batch = res_frame_queue.get(block=True, timeout=10)
                except queue.Empty: print(f"Queue empty timeout ({batch_counter}/{video_len} batches)."); continue # Simplified

                if batch is None: print("Received None sentinel."); break
                if not isinstance(batch, tuple) or len(batch) != 2: print(f"Invalid batch type: {type(batch)}."); continue

                recon_tensor_batch, audio_chunk = batch
                if not isinstance(recon_tensor_batch, torch.Tensor): print(f"Error: Received non-Tensor frames: {type(recon_tensor_batch)}."); batch_counter += 1; continue
                if recon_tensor_batch.ndim != 4 or recon_tensor_batch.device.type != device.type: print(f"Error: Invalid GPU tensor. Dim:{recon_tensor_batch.ndim} Dev:{recon_tensor_batch.device}."); batch_counter += 1; continue

                num_frames_in_batch = recon_tensor_batch.shape[0]
                if num_frames_in_batch == 0: print("Empty tensor batch."); batch_counter += 1; continue

                args_list = []
                # recon_tensor_batch assumed NCHW from inference method
                for i in range(num_frames_in_batch):
                    res_frame_gpu_nchw = recon_tensor_batch[i] # Get i-th frame tensor [C, H, W]
                    worker_args = (self, i, current_loop_start_idx, res_frame_gpu_nchw, gst_video_pipeline.width, gst_video_pipeline.height, device)
                    args_list.append(worker_args)

                results_with_indices = []
                try: results_with_indices = list(executor.map(process_single_frame_gpu, args_list)) # Use GPU worker
                except Exception as e_map: print(f"Error executor.map batch {batch_counter}: {e_map}"); batch_counter += 1; self.idx += num_frames_in_batch; continue

                prepared_frames_dict = {}; successful_frames_in_batch = 0
                for result_tuple in results_with_indices:
                     if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                          i_res, result_frame = result_tuple
                          if result_frame is not None and isinstance(result_frame, np.ndarray): prepared_frames_dict[i_res] = result_frame; successful_frames_in_batch += 1

                prepared_frames = [prepared_frames_dict[i_res] for i_res in range(num_frames_in_batch) if i_res in prepared_frames_dict]

                frames_actually_sent_this_batch = 0
                if prepared_frames and audio_chunk is not None:
                    video_send_success = True
                    try:
                        for frame in prepared_frames:
                             if not gst_video_pipeline.send_frame(frame): print("Error sending video, stopping batch send."); video_send_success = False; break
                             else: frames_actually_sent_this_batch += 1
                    except Exception as e: print(f"Error sending video: {e}"); video_send_success = False
                    if video_send_success:
                         try:
                              if isinstance(audio_chunk, np.ndarray) and audio_chunk.dtype == np.int16:
                                   if not gst_audio_pipeline.send_audio(audio_chunk): print("Error sending audio.")
                              else: print(f"Invalid audio type/dtype")
                         except Exception as e: print(f"Error sending audio: {e}")
                elif not prepared_frames and audio_chunk is not None: print("Warn: No frames prepared, audio may desync.")

                total_frames_sent += frames_actually_sent_this_batch
                batch_counter += 1
                self.idx += num_frames_in_batch

        print(f"Processing loop finished. Batches: {batch_counter}. Frames Sent: {total_frames_sent}")


    # ========================================================================
    # === inference method FINAL VERSION with ALL fixes ======================
    # ========================================================================
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        res_frame_queue = queue.Queue(maxsize=max(10, (os.cpu_count() or 2)*2))
        gst_video_pipeline = None; gst_audio_pipeline = None
        process_thread = None; start_time = time.time(); frame_count = 0
        global device, audio_processor, vae, unet, pe, timesteps

        try: # Outer try block
            print(f"Starting inference for {audio_path}")
            try: # GStreamer Init
                gst_video_pipeline = GStreamerPipeline(fps=fps)
                gst_audio_pipeline = GStreamerAudio()
                if gst_video_pipeline.process is None or gst_audio_pipeline.process is None:
                     raise RuntimeError("GStreamer failed to initialize.")
            except Exception as e_gst: print(f"Fatal GStreamer Error: {e_gst}"); return

            print("Processing audio..."); total_iters = 0
            try: # Audio Processing
                whisper_feature = audio_processor.audio2feat(audio_path)
                whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
                video_len = len(whisper_chunks); total_iters = video_len
                if total_iters == 0: raise ValueError("No audio features.")
                audio_reader = FFmpegAudioReader(audio_path); audio_data = audio_reader.read_full_audio()
                if audio_data is None: raise ValueError("Audio read failed.")
                print(f"Audio/Feature Chunks (total_iters): {total_iters}, Input Batch Size: {self.batch_size}")
                audio_chunks = split_audio(audio_data, total_iters)
                if len(audio_chunks) != total_iters:
                     print(f"Warn: Audio chunks ({len(audio_chunks)}) != iters ({total_iters}). Using {len(audio_chunks)}.")
                     total_iters = len(audio_chunks) # Adjust
                if total_iters == 0: raise ValueError("No audio chunks.")
            except Exception as e_audio: print(f"Fatal Audio Error: {e_audio}"); raise

            self.idx = 0 # Reset index
            process_thread = threading.Thread(target=self.process_frames,
                                              args=(res_frame_queue, total_iters, gst_video_pipeline, gst_audio_pipeline, skip_save_images))
            process_thread.daemon = True; process_thread.start()

            print("Starting main inference loop...")
            gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)

            for i, batch_data in enumerate(tqdm(gen, total=total_iters)):
                if batch_data is None or len(batch_data) != 2: continue
                whisper_batch, latent_batch = batch_data
                if whisper_batch is None or latent_batch is None: continue

                try:
                    # --- Prepare Batch ---
                    audio_feature_batch = torch.from_numpy(whisper_batch).to(device, dtype=unet.model.dtype, non_blocking=True)
                    audio_feature_batch = pe(audio_feature_batch)
                    if not isinstance(latent_batch, torch.Tensor): latent_batch = torch.stack(latent_batch)
                    latent_batch = latent_batch.to(device, dtype=unet.model.dtype)

                    # --- Inference ---
                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents) # << GET VAE OUTPUT ('recon')

                    # =======================================================
                    # === VAE Output Handling (for GPU Worker) ==============
                    # =======================================================
                    recon_tensor_batch_nchw = None # Variable to hold final tensor for queue

                    # --- CORRECTED: Check 'recon' variable first ---
                    if isinstance(recon, torch.Tensor):
                        recon_tensor_batch = recon
                        if recon_tensor_batch.ndim != 4: print(f"Error: VAE Tensor wrong dims: {recon.shape}. Skip."); continue
                        if recon_tensor_batch.device.type != device.type: recon_tensor_batch = recon_tensor_batch.to(device)
                        if recon_tensor_batch.dtype != torch.float32: recon_tensor_batch = recon_tensor_batch.float()
                        if recon_tensor_batch.shape[1]!=3 and recon_tensor_batch.shape[3]==3: recon_tensor_batch = recon_tensor_batch.permute(0,3,1,2).contiguous()
                        elif recon_tensor_batch.shape[1]!=3: print(f"Error: VAE Tensor wrong channels: {recon.shape}. Skip."); continue
                        recon_tensor_batch_nchw = recon_tensor_batch

                    elif isinstance(recon, np.ndarray): # EXPECTED CASE
                        if recon.ndim == 4:
                            frames_np = recon # Assume NHWC uint8 BGR
                            if frames_np.shape[1]==3 or frames_np.shape[1]==1: frames_np = frames_np.transpose((0,2,3,1))
                            if frames_np.dtype != np.uint8: print(f"Warn: VAE NumPy dtype {frames_np.dtype}, expected uint8."); frames_np=frames_np.astype(np.uint8)
                            if frames_np.shape[-1] != 3: print(f"Error: VAE NumPy wrong channels: {frames_np.shape}. Skip."); continue
                            # Convert batch: NumPy BGR uint8 NHWC -> Tensor RGB float32 NCHW GPU
                            frames_np_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_np]
                            frames_np_rgb_batch = np.stack(frames_np_rgb, axis=0)
                            recon_tensor_batch_nchw = torch.from_numpy(frames_np_rgb_batch).permute(0, 3, 1, 2).to(device, non_blocking=True).float().div_(255.0)
                        else: print(f"Error: VAE NumPy wrong dims: {recon.shape}. Skip."); continue

                    elif isinstance(recon, list) and all(isinstance(f, np.ndarray) for f in recon):
                        print(f"Warning: VAE returned list of NumPy (unexpected). Count: {len(recon)}")
                        try: # Convert list[HWC BGR uint8] -> Tensor NCHW RGB float32 GPU
                            frames_np_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in recon]
                            frames_np_rgb_batch = np.stack(frames_np_rgb, axis=0)
                            recon_tensor_batch_nchw = torch.from_numpy(frames_np_rgb_batch).permute(0, 3, 1, 2).to(device, non_blocking=True).float().div_(255.0)
                        except Exception as e_list_conv: print(f"Error converting list output: {e_list_conv}"); continue
                    else:
                        print(f"VAE output unexpected type: {type(recon)}. Skipping {i}."); continue
                    # =======================================================
                    # === END VAE Output Handling ===========================
                    # =======================================================

                    # --- Check and Queue the GPU Tensor Batch ---
                    if recon_tensor_batch_nchw is None or recon_tensor_batch_nchw.ndim != 4:
                         print(f"Warn: Failed to create valid GPU tensor batch iter {i}. Skipping put.")
                         continue

                    current_batch_size = recon_tensor_batch_nchw.shape[0]
                    frame_count += current_batch_size
                    if i < len(audio_chunks):
                        try: # Put the NCHW GPU tensor batch
                            res_frame_queue.put((recon_tensor_batch_nchw, audio_chunks[i]), block=True, timeout=5)
                        except queue.Full: print(f"Warn: Queue full iter {i}. Skipping."); frame_count -= current_batch_size; continue
                    else: print(f"Warn: Audio index {i} OOB. Skipping put.")

                except Exception as e_loop: print(f"Err iter {i}:"); traceback.print_exc(); continue
            # --- End of Main Loop ---

        except Exception as e_inf:
             print(f"\nFATAL ERROR inference: {e_inf}"); traceback.print_exc()
             # === CORRECTED Syntax Error in Exception Handler ===
             if process_thread is not None and process_thread.is_alive():
                 try:
                     print("Attempting to signal worker thread with None due to error...")
                     res_frame_queue.put(None, block=False) # Try non-blocking put
                 except queue.Full:
                     print("Queue full when trying to send None sentinel during error.")
                 except Exception as e_put_err:
                     print(f"Exception sending None sentinel during error: {e_put_err}")
             # === END CORRECTION ===

        finally:
             # --- ENSURE FINAL CLEANUP AND FPS PRINT ---
             print("\n--- Starting Final Cleanup ---")
             try: print("Signaling worker (sending None)..."); res_frame_queue.put(None, block=False)
             except: pass # Ignore queue errors on cleanup
             if process_thread is not None:
                 print("Waiting for worker thread..."); process_thread.join(timeout=30)
                 if process_thread.is_alive(): print("⚠️ Worker thread timeout.")
                 else: print("✅ Worker thread joined.")
             else: print("Worker thread not started.")
             print("Stopping GStreamer...");
             # Check if pipelines were initialized before stopping
             if gst_video_pipeline: gst_video_pipeline.stop()
             if gst_audio_pipeline: gst_audio_pipeline.stop()
             print("GStreamer stop called.")
             print("Calculating FPS..."); total_elapsed_time = time.time() - start_time
             avg_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0
             print("\n========================================")
             print(f" Total elapsed time: {total_elapsed_time:.2f} s")
             print(f" Total frame count generated by VAE: {frame_count}")
             print(f" Final calculated average FPS: {avg_fps:.2f}")
             print("========================================")
             print(">>> Inference method finished.")


# --- Main Execution Block ---
# (Keep __main__ block from previous response)
if __name__ == "__main__":
    ''' Real-time streaming script for MuseTalk. '''
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(); parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--fps", type=int, default=25); parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--skip_save_images", action="store_true", help="Legacy flag")
    args = parser.parse_args(); print(f"Arguments: {args}")
    try: inference_config = OmegaConf.load(args.inference_config); print("Loaded config:"); print(OmegaConf.to_yaml(inference_config))
    except Exception as e_conf: print(f"Error loading config '{args.inference_config}': {e_conf}"); sys.exit(1)

    for avatar_id in inference_config:
        print(f"\n===== Processing Avatar: {avatar_id} =====")
        try:
            cfg = inference_config[avatar_id]; prep = cfg.get("preparation", False); vpath = cfg.get("video_path")
            bbox_s = cfg.get("bbox_shift", 0); aclips = cfg.get("audio_clips")
            if not vpath or not aclips: print(f"Error: Missing config for '{avatar_id}'."); continue
            avatar = Avatar(avatar_id, vpath, bbox_s, args.batch_size, prep)
            for anum, apath in aclips.items():
                print(f"\n--- Inferring Audio: {anum} ({apath}) ---")
                if not os.path.exists(apath): print(f"Warn: Audio not found: {apath}."); continue
                avatar.inference(apath, anum, args.fps, args.skip_save_images)
                print(f"--- Finished Audio: {anum} ---")
        except SystemExit as e: print(f"Exiting: {e}"); break
        except Exception as e_avatar: print(f"\n!!!!! Error avatar '{avatar_id}' !!!!!"); traceback.print_exc(); print(f"!!!!! Skipping !!!!!"); continue
    print("\n===== All Avatars Processed =====")

