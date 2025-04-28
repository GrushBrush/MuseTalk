# -*- coding: utf-8 -*-
import ffmpeg
import argparse
import os
import concurrent.futures # For ThreadPoolExecutor
import threading
import queue
from omegaconf import OmegaConf
import subprocess
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
import traceback          # For detailed error printing in threads
import time
from PIL import Image
import tempfile
import logging # Added for better logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')


# --- Watchdog and DotEnv Imports ---
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

# --- MuseTalk Specific Imports (Ensure these paths are correct) ---
try:
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    # from musetalk.utils.blending import get_image_prepare_material # Logic now likely in worker or prepare
    from musetalk.utils.utils import load_all_model
except ImportError as e:
    print(f"Error importing MuseTalk utilities: {e}")
    print("Please ensure the MuseTalk library is correctly installed and accessible.")
    sys.exit(1)

import shutil

# --- Configuration Loading ---
load_dotenv()
WATCHED_WAV_FILE_PATH = os.getenv("WATCHED_WAV_FILE_PATH")
AVATAR_CONFIG_PATH = os.getenv("AVATAR_CONFIG_PATH", "configs/inference/realtime.yaml")
AVATAR_ID_TO_USE = os.getenv("AVATAR_ID_TO_USE")
try:
    TARGET_FPS = int(os.getenv("TARGET_FPS", "25"))
except ValueError:
    print("Warning: TARGET_FPS in .env is not a valid integer. Using default 25.")
    TARGET_FPS = 25

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Global Lock for Inference ---
inference_lock = threading.Lock()
is_inferencing = False # Flag to track inference state

# --- PyTorch Device Setup ---
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logging.info("--- PyTorch Device Information ---")
if cuda_available:
    logging.info("✅ CUDA (GPU) detected by PyTorch.")
    try: gpu_name = torch.cuda.get_device_name(0); logging.info(f"✅ Using GPU: {gpu_name}")
    except Exception as e: logging.warning(f"⚠️ Could not retrieve GPU name: {e}")
    logging.info(f"✅ Selected device: {device}")
else:
    logging.info("❌ CUDA (GPU) not available or not detected by PyTorch.")
    logging.info(f"✅ Selected device: {device}")
logging.info("-------------------------------")

# --- Load Models ---
logging.info("Loading models...")
try:
    audio_processor, vae, unet, pe = load_all_model()
    logging.info("Models loaded.")
except Exception as e:
    logging.error(f"Fatal error loading models: {e}")
    sys.exit(1)

# --- Set Precision ---
logging.info("Setting model precision to half (FP16) on sub-modules...")
try:
    # (Keep the precision setting logic from your original script)
    if hasattr(pe, 'to'): pe = pe.to(device)
    if hasattr(pe, 'half'): pe = pe.half()
    else: logging.warning("Warning: 'pe' object doesn't have .half() method.")
    if hasattr(vae, 'vae'):
        if hasattr(vae.vae, 'to'): vae.vae = vae.vae.to(device)
        if hasattr(vae.vae, 'half'): vae.vae = vae.vae.half()
        else: logging.warning("Warning: 'vae.vae' object doesn't have .half() method.")
    else: logging.warning("Warning: Cannot access 'vae.vae'.")
    if hasattr(unet, 'model'):
        if hasattr(unet.model, 'to'): unet.model = unet.model.to(device)
        if hasattr(unet.model, 'half'): unet.model = unet.model.half()
        else: logging.warning("Warning: 'unet.model' object doesn't have .half() method.")
    else: logging.warning("Warning: Cannot access 'unet.model'.")
    logging.info("Precision set (check warnings above).")
except Exception as e_prec:
    logging.warning(f"Warning: Error setting model precision/device: {e_prec}")
timesteps = torch.tensor([0], device=device)

# --- FFmpegAudioReader Class (Keep as is from your script) ---
class FFmpegAudioReader:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        try:
            logging.info(f"Probing audio file: {audio_file}")
            probe = ffmpeg.probe(audio_file)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            if audio_stream is None: raise RuntimeError(f"No audio stream found in {audio_file}")
            self.sample_rate = int(audio_stream.get('sample_rate', '48000'))
            self.channels = int(audio_stream.get('channels', '2'))
            logging.info(f"Detected Sample Rate: {self.sample_rate}, Channels: {self.channels}")
        except ffmpeg.Error as e:
            logging.error(f"❌ Error probing audio file {audio_file}: {e.stderr.decode(errors='ignore') if e.stderr else 'Unknown FFmpeg error'}")
            raise
        except Exception as e: logging.error(f"❌ Unexpected error during FFmpeg probe: {e}"); raise

    def read_full_audio(self):
        logging.info("Reading and converting audio with FFmpeg...")
        target_sr = 48000; target_ac = 2; target_format = "s16le"
        try:
            process = subprocess.Popen(["ffmpeg", "-i", self.audio_file,"-f", target_format,"-ac", str(target_ac),"-ar", str(target_sr),"-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_data, stderr_data = process.communicate()
            retcode = process.poll(); stderr_str = stderr_data.decode(errors='ignore')
            if retcode != 0: logging.error(f"❌ FFmpeg process failed code {retcode}\nFFmpeg stderr:\n{stderr_str}"); return None
            if not raw_data: logging.error(f"❌ Failed read audio (FFmpeg no data)!\nFFmpeg stderr:\n{stderr_str}"); return None
            audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, target_ac)
            logging.info(f"✅ Read/converted audio, {len(audio_data)} samples at {target_sr}Hz, {target_ac}ch")
            return audio_data
        except FileNotFoundError: logging.error("❌ Error: ffmpeg command not found."); return None
        except Exception as e: logging.error(f"❌ FFmpeg read error: {e}"); return None

# --- GStreamerPipeline Class (Keep as is from your script) ---
class GStreamerPipeline:
    def __init__(self, width=720, height=1280, fps=25, host="127.0.0.1", port=5000):
        self.width = width; self.height = height; self.fps = fps; self.host = host; self.port = port; self.process = None
        pipeline_str_fdsrc = (f"fdsrc fd=0 ! videoparse format=bgr width={self.width} height={self.height} framerate={self.fps}/1 ! "
                              "queue ! videoconvert ! video/x-raw,format=NV12 ! queue ! cudaupload ! "
                              "nvh265enc bitrate=80000 bframes=0 preset=default gop-size=25 ! "
                              "h265parse ! rtph265pay config-interval=1 ! "
                              f"udpsink host={self.host} port={self.port} sync=false")
        logging.info("Starting GStreamer video pipeline...")
        try:
            self.process = subprocess.Popen(f"gst-launch-1.0 -v {pipeline_str_fdsrc}", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0)
            logging.info(f"✅ GStreamer video pipeline started (Res: {self.width}x{self.height} @ {self.fps}fps)")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_VID_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_VID_ERR"), daemon=True).start()
        except Exception as e: logging.error(f"❌ Failed start GStreamer video pipeline: {e}"); self.process = None
    def _log_stream(self, stream, prefix):
        try:
            for line in iter(stream.readline, b''): logging.debug(f"[{prefix}]: {line.decode(errors='ignore').strip()}")
        except Exception as e: logging.warning(f"Error GStreamer log thread ({prefix}): {e}")
        finally: stream.close()
    def send_frame(self, frame):
        if not self.process or self.process.stdin.closed: return False
        try:
            if frame.shape[0] != self.height or frame.shape[1] != self.width: frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
            self.process.stdin.write(frame.tobytes()); self.process.stdin.flush(); return True
        except BrokenPipeError: logging.error("❌ Error pushing video frame: Broken pipe."); self.process = None; return False
        except Exception as e: logging.error(f"❌ Error pushing video frame: {e}"); return False
    def stop(self):
        logging.info("Stopping GStreamer video pipeline...")
        if self.process:
            proc = self.process; self.process = None
            try:
                if not proc.stdin.closed: proc.stdin.close()
            except Exception as e: logging.warning(f"Error closing video stdin: {e}")
            try:
                proc.wait(timeout=5); logging.info("✅ GStreamer video process terminated.")
            except subprocess.TimeoutExpired: logging.warning("⚠️ GStreamer video kill..."); proc.kill(); proc.wait(); logging.info("✅ GStreamer video process killed.")
            except Exception as e: logging.error(f"Error waiting for video process: {e}")
        else: logging.info("Video pipeline already stopped.")

# --- GStreamerAudio Class (Keep as is from your script) ---
class GStreamerAudio:
    def __init__(self, host="127.0.0.1", port=5001):
        self.host = host; self.port = port; self.process = None
        pipeline_str = ("fdsrc fd=0 do-timestamp=true ! audio/x-raw,format=S16LE,channels=2,rate=48000,layout=interleaved ! "
                        "queue max-size-time=500000000 ! " # ADDED: Queue with 500ms buffer (time in ns)
                        "opusenc name=enc bitrate=64000 complexity=4 frame-size=20 ! rtpopuspay name=pay pt=97 ! "
                        f"udpsink host={self.host} port={self.port} sync=false async=false") # Keep sync=true for now
        logging.info("Starting GStreamer audio pipeline...")
        try:
            self.process = subprocess.Popen(f"gst-launch-1.0 -v {pipeline_str}", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0)
            logging.info("✅ GStreamer audio pipeline started.")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_AUD_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_AUD_ERR"), daemon=True).start()
        except Exception as e: logging.error(f"❌ Failed start GStreamer audio pipeline: {e}"); self.process = None
    def _log_stream(self, stream, prefix):
        try:
            for line in iter(stream.readline, b''): logging.debug(f"[{prefix}]: {line.decode(errors='ignore').strip()}")
        except Exception as e: logging.warning(f"Error GStreamer log thread ({prefix}): {e}")
        finally: stream.close()
    def send_audio(self, audio_data):
        if not self.process or self.process.stdin.closed: return False
        try:
            if audio_data.dtype != np.int16: audio_data = audio_data.astype(np.int16)
            self.process.stdin.write(audio_data.tobytes()); self.process.stdin.flush(); return True
        except BrokenPipeError: logging.error("❌ Error pushing audio chunk: Broken pipe."); self.process = None; return False
        except Exception as e: logging.error(f"❌ Error pushing audio chunk: {e}"); return False
    def stop(self):
        logging.info("Stopping GStreamer audio pipeline...")
        if self.process:
            proc = self.process; self.process = None
            try:
                if not proc.stdin.closed: proc.stdin.close()
            except Exception as e: logging.warning(f"Error closing audio stdin: {e}")
            try:
                proc.wait(timeout=5); logging.info("✅ GStreamer audio process terminated.")
            except subprocess.TimeoutExpired: logging.warning("⚠️ GStreamer audio kill..."); proc.kill(); proc.wait(); logging.info("✅ GStreamer audio process killed.")
            except Exception as e: logging.error(f"Error waiting for audio process: {e}")
        else: logging.info("Audio pipeline already stopped.")


# --- Misc Helper Functions (Keep as is: video2imgs, osmakedirs, process_single_frame_parallel) ---
def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    logging.info(f"Extracting frames from {vid_path} to {save_path}...")
    cap = cv2.VideoCapture(vid_path); count = 0; frame_idx = 0
    if not cap.isOpened(): logging.error(f"Error: Could not open video file: {vid_path}"); return
    while True:
        ret, frame = cap.read()
        if not ret: logging.info("Video ended or read error."); break
        if frame_idx % 1 == 0:
             filename = f"{count:08d}{ext}"; filepath = os.path.join(save_path, filename)
             try: cv2.imwrite(filepath, frame); count += 1
             except Exception as e: logging.error(f"Error writing frame {count}: {e}"); break
             if count >= cut_frame: logging.info(f"Reached cut_frame limit: {cut_frame}"); break
        frame_idx += 1
    cap.release(); logging.info(f"Finished extracting {count} frames.")

def osmakedirs(path_list):
    for path in path_list:
        try: os.makedirs(path, exist_ok=True)
        except Exception as e: logging.error(f"Error creating directory {path}: {e}")

# --- IMPORTANT: Ensure this uses the exact logic from your working script ---
def process_single_frame_parallel(args):
    self, i, start_idx, res_frame, gst_video_pipeline_width, gst_video_pipeline_height = args
    try:
        current_list_len = len(self.coord_list_cycle)
        if current_list_len == 0: logging.error("Error(worker): Avatar reference lists empty!"); return i, None
        current_idx = (start_idx + i) % current_list_len
        if current_idx >= current_list_len: logging.warning(f"Warning(worker): Index {current_idx} out of bounds. Skipping frame {i}."); return i, None

        bbox = self.coord_list_cycle[current_idx]; ori_frame_ref = self.frame_list_cycle[current_idx]
        mask = self.mask_list_cycle[current_idx]; mask_crop_box = self.mask_coords_list_cycle[current_idx]
        if ori_frame_ref is None or not isinstance(ori_frame_ref, np.ndarray): logging.error(f"Error(worker): Invalid ori_frame_ref index {current_idx}. Skip frame {i}."); return i, None
        ori_frame = ori_frame_ref.copy()
        x, y, x1, y1 = bbox

        res_frame_resized = None
        if x1 > x and y1 > y:
            if isinstance(res_frame, np.ndarray): res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x1 - x, y1 - y), interpolation=cv2.INTER_LINEAR)
            else: logging.error(f"Error(worker): res_frame not NumPy array frame {i}. Skip."); return i, None
        else: logging.warning(f"Warning(worker): Invalid bbox dims {bbox}. Skip frame {i}."); return i, None
        if res_frame_resized is None: logging.warning(f"Warning(worker): First resize failed frame {i}."); return i, None

        combine_frame = None; body = ori_frame; face = res_frame_resized
        mask_array = mask; crop_box = mask_crop_box
        if mask_array is None: logging.error(f"Error(worker): mask_array None index {current_idx}. Skip frame {i}."); return i, None
        if crop_box is None or len(crop_box) != 4: logging.error(f"Error(worker): crop_box invalid index {current_idx}. Skip frame {i}."); return i, None
        x_s, y_s, x_e, y_e = crop_box
        if y_e <= y_s or x_e <= x_s: logging.warning(f"Warning(worker): Invalid crop_box dims {crop_box}. Skip frame {i}."); return i, None
        h_body, w_body = body.shape[:2]; y_s, y_e = max(0, y_s), min(h_body, y_e); x_s, x_e = max(0, x_s), min(w_body, x_e)
        if y_e <= y_s or x_e <= x_s: logging.warning(f"Warning(worker): Clamped crop_box zero size. Skip frame {i}."); return i, None
        face_large = body[y_s:y_e, x_s:x_e].copy()
        y_start_paste = y - y_s; y_end_paste = y1 - y_s; x_start_paste = x - x_s; x_end_paste = x1 - x_s
        h_large, w_large = face_large.shape[:2]; y_start_paste, y_end_paste = max(0, y_start_paste), min(h_large, y_end_paste); x_start_paste, x_end_paste = max(0, x_start_paste), min(w_large, x_end_paste)
        if y_start_paste >= y_end_paste or x_start_paste >= x_end_paste: logging.warning(f"Warning(worker): Invalid paste slice zero size. Skip frame {i}."); return i, None
        slice_h = y_end_paste - y_start_paste; slice_w = x_end_paste - x_start_paste
        if face.shape[0] != slice_h or face.shape[1] != slice_w:
            try: face = cv2.resize(face, (slice_w, slice_h), interpolation=cv2.INTER_LINEAR)
            except Exception as e_resize2: logging.error(f"Error(worker): Failed resize face to slice: {e_resize2}. Skip frame {i}."); return i, None
        face_large[y_start_paste:y_end_paste, x_start_paste:x_end_paste] = face
        mask_image_blend = mask_array
        if len(mask_image_blend.shape) == 3: mask_image_blend = cv2.cvtColor(mask_image_blend, cv2.COLOR_BGR2GRAY)
        if mask_image_blend.shape[0] != face_large.shape[0] or mask_image_blend.shape[1] != face_large.shape[1]: mask_image_blend = cv2.resize(mask_image_blend, (face_large.shape[1], face_large.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_image_blend = (mask_image_blend / 255.0).astype(np.float32)
        try:
            mask_expanded = mask_image_blend[..., np.newaxis]
            if mask_expanded.shape[0]!=face_large.shape[0] or mask_expanded.shape[1]!=face_large.shape[1]: raise ValueError(f"Shape mismatch: mask {mask_expanded.shape}, face_large {face_large.shape}")
            body_slice = body[y_s:y_e, x_s:x_e]
            if body_slice.shape != face_large.shape: raise ValueError(f"Shape mismatch: body_slice {body_slice.shape}, face_large {face_large.shape}")
            blended_region = face_large.astype(np.float32) * mask_expanded + body_slice.astype(np.float32) * (1.0 - mask_expanded)
            body[y_s:y_e, x_s:x_e] = blended_region.astype(body.dtype); combine_frame = body
        except Exception as e_blend: logging.error(f"Error(worker): Blending failed frame {i}: {e_blend}"); return i, None
        if combine_frame is not None:
            final_resized = cv2.resize(combine_frame, (gst_video_pipeline_width, gst_video_pipeline_height), interpolation=cv2.INTER_LINEAR)
            return i, final_resized
        else: logging.warning(f"Warning(worker): combine_frame None frame {i}."); return i, None
    except Exception as e_worker: logging.error(f"!!!!! Error(worker): Unexpected error frame {i} !!!!!"); traceback.print_exc(); return i, None

# --- Avatar Class (Keep as is, including __init__, init, _reload_prepared_data, prepare_material, process_frames, inference) ---
# Make sure all necessary global variables (models like vae, unet, pe, audio_processor) are accessible
# Ensure the `inference` method correctly uses the provided `audio_path` and `fps`
@torch.no_grad()
class Avatar:
    # (Keep the __init__, init, _reload_prepared_data, prepare_material methods EXACTLY as in your working script)
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        logging.info(f"Initializing Avatar: {avatar_id}")
        self.avatar_id = avatar_id; self.video_path = video_path; self.bbox_shift = bbox_shift
        self.batch_size = batch_size; self.preparation = preparation
        self.avatar_path = f"./results/avatars/{avatar_id}"; self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"; self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"; self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"; self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {"avatar_id": avatar_id, "video_path": video_path, "bbox_shift": bbox_shift}; self.idx = 0
        self.input_latent_list_cycle = None; self.coord_list_cycle = None; self.frame_list_cycle = None
        self.mask_coords_list_cycle = None; self.mask_list_cycle = None
        self.init()
        logging.info(f"Avatar initialization complete for {avatar_id}.")

    def init(self):
         if self.preparation:
             if os.path.exists(self.avatar_path):
                 try:
                      response = input(f"Avatar '{self.avatar_id}' exists. Re-create? (y/n): ")
                      if response.lower() == "y":
                          logging.info(f"Removing existing avatar data: {self.avatar_path}"); shutil.rmtree(self.avatar_path)
                          self.prepare_material()
                      else: logging.info("Attempting load existing data..."); self._reload_prepared_data()
                 except Exception as e_input: logging.warning(f"Error user input: {e_input}. Assuming 'n'."); logging.info("Attempting load existing data..."); self._reload_prepared_data()
             else: logging.info(f"Avatar path {self.avatar_path} not exist. Preparing..."); self.prepare_material()
         else:
              required_files = [self.coords_path, self.latents_out_path, self.mask_coords_path, self.mask_out_path, self.full_imgs_path]
              if not all(os.path.exists(p) for p in required_files):
                   logging.error(f"Error: Not all required data found in {self.avatar_path} and preparation=False."); logging.error(f" Missing: {[p for p in required_files if not os.path.exists(p)]}"); sys.exit(1)
              else: logging.info("Preparation=False. Loading existing data..."); self._reload_prepared_data()
              try:
                   if os.path.exists(self.avatar_info_path):
                        with open(self.avatar_info_path, "r") as f: avatar_info_disk = json.load(f)
                        if avatar_info_disk.get('bbox_shift') != self.avatar_info['bbox_shift']: logging.error(f"Error: bbox_shift changed since preparation. Current: {self.avatar_info['bbox_shift']}, Prepared: {avatar_info_disk.get('bbox_shift')}"); sys.exit(1)
                   else: logging.warning(f"Warning: {self.avatar_info_path} not found. Cannot check consistency.")
              except Exception as e_info: logging.warning(f"Warning: Could not check avatar info: {e_info}")

    def _reload_prepared_data(self):
         logging.info("Reloading prepared data into instance variables...")
         all_loaded = True
         try:
            logging.info(f" Loading latents: {self.latents_out_path}")
            self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location='cpu')
            if isinstance(self.input_latent_list_cycle, torch.Tensor): self.input_latent_list_cycle = list(self.input_latent_list_cycle)
            logging.info(f" Loading coords: {self.coords_path}")
            with open(self.coords_path, 'rb') as f: self.coord_list_cycle = pickle.load(f)
            logging.info(f" Loading mask coords: {self.mask_coords_path}")
            with open(self.mask_coords_path, 'rb') as f: self.mask_coords_list_cycle = pickle.load(f)
            logging.info(f" Loading frames from: {self.full_imgs_path}")
            input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jp][pn]g')))
            if not input_img_list: raise FileNotFoundError(f"No images found in {self.full_imgs_path}")
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.frame_list_cycle = read_imgs(input_img_list)
            logging.info(f" Loading masks from: {self.mask_out_path}")
            input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jp][pn]g')))
            if not input_mask_list: raise FileNotFoundError(f"No masks found in {self.mask_out_path}")
            input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.mask_list_cycle = read_imgs(input_mask_list)
            logging.info("Validating loaded data...")
            data_lists = {"Coords": self.coord_list_cycle,"Frames": self.frame_list_cycle,"Masks": self.mask_list_cycle,"MaskCoords": self.mask_coords_list_cycle,"Latents": self.input_latent_list_cycle}
            list_lengths = {}
            for name, data_list in data_lists.items():
                 if data_list is None or not isinstance(data_list, list) or len(data_list) == 0: logging.error(f"Error: Data list '{name}' invalid after load."); all_loaded = False; break
                 list_lengths[name] = len(data_list)
            if all_loaded and len(set(list_lengths.values())) > 1: logging.error(f"Error: Lengths of loaded data lists mismatch! Lengths: {list_lengths}"); all_loaded = False
            if not all_loaded: raise ValueError("Failed load/validate all required data.")
            logging.info(f"Successfully reloaded {list_lengths.get('Frames', 0)} items.")
         except FileNotFoundError as e: logging.error(f"Error reloading: File not found - {e}"); raise SystemExit(f"Missing prepared file: {e}")
         except Exception as e: logging.error(f"Error reloading prepared data: {e}"); traceback.print_exc(); raise SystemExit(f"Failed reload data: {e}")

    def prepare_material(self):
        logging.info(f"--- Preparing material for avatar: {self.avatar_id} ---")
        osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
        with open(self.avatar_info_path, "w") as f: json.dump(self.avatar_info, f)
        if os.path.isfile(self.video_path): video2imgs(self.video_path, self.full_imgs_path, ext='.png')
        elif os.path.isdir(self.video_path):
            img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(img_extensions)])
            if not files: logging.error(f"Error: No images found {self.video_path}"); sys.exit(1)
            for i, filename in enumerate(tqdm(files, desc="Copying frames")):
                new_filename = f"{i:08d}.png"
                try: shutil.copyfile(os.path.join(self.video_path, filename), os.path.join(self.full_imgs_path, new_filename))
                except Exception as e_copy: logging.warning(f"Error copying {filename}: {e_copy}"); continue
            logging.info(f"Copied {len(files)} frames.")
        else: logging.error(f"Error: video_path '{self.video_path}' invalid."); sys.exit(1)

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))
        if not input_img_list: logging.error(f"Error: No PNGs found {self.full_imgs_path}"); sys.exit(1)
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        logging.info("Extracting landmarks/bboxes...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        logging.info(f" Found {len(coord_list)} coords, {len(frame_list)} frames init.")
        logging.info("Encoding VAE latents...")
        input_latent_list, valid_coords, valid_frames, valid_indices = [], [], [], []
        coord_placeholder_val = coord_placeholder()
        global vae
        if 'vae' not in globals() or vae is None: logging.error("Error: VAE model not loaded."); sys.exit(1)
        for i, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list), desc="VAE Encode")):
            if bbox is None or bbox == coord_placeholder_val or frame is None: continue
            x1c, y1c, x2c, y2c = bbox; y1c, y2c = int(round(y1c)), int(round(y2c)); x1c, x2c = int(round(x1c)), int(round(x2c))
            hf, wf = frame.shape[:2]; y1c, y2c = max(0, y1c), min(hf, y2c); x1c, x2c = max(0, x1c), min(wf, x2c)
            if x1c >= x2c or y1c >= y2c: continue
            crop_frame = frame[y1c:y2c, x1c:x2c]
            if crop_frame.size == 0: continue
            try:
                 resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                 latents = vae.get_latents_for_unet(resized_crop_frame)
                 input_latent_list.append(latents); valid_coords.append(bbox); valid_frames.append(frame); valid_indices.append(i)
            except Exception as e: logging.warning(f"Error VAE frame {i}: {e}"); continue
        if not input_latent_list: logging.error("Error: No valid latents generated."); sys.exit(1)
        logging.info(f"Generated {len(input_latent_list)} valid latents.")
        logging.info("Creating data cycles...")
        frame_list_cycle_prep = valid_frames + valid_frames[::-1]; coord_list_cycle_prep = valid_coords + valid_coords[::-1]
        input_latent_list_cycle_prep = input_latent_list + input_latent_list[::-1]; num_cycle_frames = len(frame_list_cycle_prep)
        logging.info(f"Cycle length: {num_cycle_frames} frames.")

        logging.info("Generating masks...")
        mask_coords_list_cycle_prep, mask_list_cycle_prep = [], []
        # --- Mask Generation (Ensure get_image_prepare_material is defined/imported) ---
        # Assuming get_image_prepare_material exists and works as before
        global get_image_prepare_material # Check if it exists
        if 'get_image_prepare_material' not in globals(): logging.error("Error: Blending function 'get_image_prepare_material' missing."); sys.exit(1)
        processed_indices = set(); temp_mask_data = {}
        for i, frame in enumerate(tqdm(frame_list_cycle_prep, desc="Mask Gen")):
            face_box = coord_list_cycle_prep[i]
            try:
                 mask, crop_box = get_image_prepare_material(frame, face_box) # CALL THE FUNCTION
                 if mask is None or crop_box is None: logging.warning(f"Warning: Mask gen fail frame {i}. Skip."); continue
                 cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
                 temp_mask_data[i] = (mask, crop_box); processed_indices.add(i)
            except Exception as e: logging.warning(f"Error mask gen frame {i}: {e}"); continue
        # --- End Mask Generation ---

        logging.info(f"Filtering cycles based on {len(processed_indices)} successful masks...")
        final_frame_list_cycle, final_coord_list_cycle, final_input_latent_list_cycle = [], [], []
        final_mask_list_cycle, final_mask_coords_list_cycle = [], []
        for i in range(num_cycle_frames):
             if i in processed_indices:
                  frame_path = os.path.join(self.full_imgs_path, f"{i:08d}.png"); mask_path = os.path.join(self.mask_out_path, f"{i:08d}.png")
                  if os.path.exists(mask_path) and os.path.exists(frame_path):
                       frame_img = cv2.imread(frame_path); mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                       if frame_img is not None and mask_img is not None:
                           final_frame_list_cycle.append(frame_img); final_coord_list_cycle.append(coord_list_cycle_prep[i])
                           final_input_latent_list_cycle.append(input_latent_list_cycle_prep[i]); mask_data, crop_box_data = temp_mask_data[i]
                           final_mask_list_cycle.append(mask_data); final_mask_coords_list_cycle.append(crop_box_data)
                       else: logging.warning(f"Warning: Failed read frame/mask {i}, skip.")
                  else: logging.warning(f"Warning: Missing frame/mask file index {i}, skip.")
        logging.info("Saving final prepared data...")
        final_len = len(final_frame_list_cycle)
        if not all(len(lst) == final_len for lst in [final_coord_list_cycle, final_input_latent_list_cycle, final_mask_list_cycle, final_mask_coords_list_cycle]): logging.error(f"Error: Final lengths mismatch after filter!"); sys.exit(1)
        if final_len == 0: logging.error("Error: No frames survived prepare filter."); sys.exit(1)
        with open(self.coords_path, 'wb') as f: pickle.dump(final_coord_list_cycle, f)
        with open(self.mask_coords_path, 'wb') as f: pickle.dump(final_mask_coords_list_cycle, f)
        if isinstance(final_input_latent_list_cycle[0], torch.Tensor): final_latents_tensor = torch.stack(final_input_latent_list_cycle); torch.save(final_latents_tensor, self.latents_out_path)
        else: torch.save(final_input_latent_list_cycle, self.latents_out_path)
        logging.info(f"--- Material preparation complete. Final cycle: {final_len} frames. ---")
        self._reload_prepared_data()

    # (Keep process_frames EXACTLY as in your working script)
    def process_frames(self, res_frame_queue, video_len, gst_video_pipeline, gst_audio_pipeline, skip_save_images, debug=False):
         num_workers = max(1, os.cpu_count() - 2)
         logging.info(f"--- Start process_frames with {num_workers} workers ---")
         if not hasattr(self, 'coord_list_cycle') or not self.coord_list_cycle: logging.error("Error: Avatar ref data not loaded."); return
         batch_counter = 0; total_frames_sent = 0; last_audio_send_time = None; time_diffs = []
         with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
             while True:
                 current_loop_start_idx = self.idx
                 if batch_counter >= video_len: logging.info(f"Processed target batches ({batch_counter}/{video_len}). Finish."); break
                 try: batch = res_frame_queue.get(block=True, timeout=10)
                 except queue.Empty: logging.info(f"Queue empty timeout ({batch_counter}/{video_len} batches)."); continue
                 if batch is None: logging.info("Received None sentinel. Finish."); break
                 if not isinstance(batch, tuple) or len(batch) != 2: logging.warning(f"Invalid batch type: {type(batch)}. Skip."); continue
                 frames, audio_chunk = batch
                 if not isinstance(frames, list) or len(frames) == 0: logging.warning(f"Invalid frames list. Skip."); batch_counter += 1; continue
                 if not all(isinstance(f, np.ndarray) for f in frames): logging.warning(f"Not all frames NumPy. Skip."); batch_counter += 1; continue
                 num_frames_in_batch = len(frames)
                 args_list = [(self, i, current_loop_start_idx, frame, gst_video_pipeline.width, gst_video_pipeline.height) for i, frame in enumerate(frames)]
                 results_with_indices = []
                 try: results_with_indices = list(executor.map(process_single_frame_parallel, args_list))
                 except Exception as e_map: logging.error(f"Error executor.map batch {batch_counter}: {e_map}"); batch_counter += 1; self.idx += num_frames_in_batch; continue
                 prepared_frames_dict = {}; successful_frames_in_batch = 0
                 for result_tuple in results_with_indices:
                     if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                         i, result_frame = result_tuple
                         if result_frame is not None: prepared_frames_dict[i] = result_frame; successful_frames_in_batch += 1
                     else: logging.warning(f"Warning: Worker returned bad format: {result_tuple}")
                 prepared_frames = [prepared_frames_dict[i] for i in range(num_frames_in_batch) if i in prepared_frames_dict]
                 frames_actually_sent_this_batch = 0
                 if prepared_frames and audio_chunk is not None:
                     video_send_success = True
                     try:
                         for frame in prepared_frames:
                             if gst_video_pipeline.send_frame(frame): frames_actually_sent_this_batch += 1
                             else: logging.error("Error sending video frame, stop sends."); video_send_success = False; break
                     except Exception as e: logging.error(f"Error sending video frame: {e}"); video_send_success = False
                     if video_send_success:
                         try:
                             if isinstance(audio_chunk, np.ndarray) and audio_chunk.dtype == np.int16:
                                 current_time = time.time()
                                 if last_audio_send_time is not None:
                                     time_diff = current_time - last_audio_send_time
                                     if time_diff > 0.001: logging.debug(f"DEBUG: Time since last audio send: {time_diff:.4f} s"); time_diffs.append(time_diff)
                                 if not gst_audio_pipeline.send_audio(audio_chunk): logging.error("Error sending audio chunk.")
                                 else: last_audio_send_time = current_time
                             else: logging.warning(f"Invalid audio chunk type/dtype: {type(audio_chunk)}, {getattr(audio_chunk, 'dtype', 'N/A')}")
                         except Exception as e: logging.error(f"Error sending audio chunk: {e}")
                 elif not prepared_frames and audio_chunk is not None: logging.warning("Warning: No frames prepared, audio may desync.")
                 total_frames_sent += frames_actually_sent_this_batch
                 batch_counter += 1; self.idx += num_frames_in_batch
         logging.info(f"Process loop finished after {batch_counter} batches. Total frames sent: {total_frames_sent}")
         try:
             if time_diffs: logging.debug(f"DEBUG: Audio send stats: Avg={np.mean(time_diffs):.4f}s, Std={np.std(time_diffs):.4f}s, Min={np.min(time_diffs):.4f}s, Max={np.max(time_diffs):.4f}s")
         except Exception as e_stats: logging.warning(f"DEBUG: Error calc stats: {e_stats}")

    # (Keep inference EXACTLY as in your working script)
    # ========================================================================
    # === FINAL inference method (WITH CHUNK SIZE LOGS ADDED) ================
    # ========================================================================
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        res_frame_queue = queue.Queue()
        gst_video_pipeline = None; gst_audio_pipeline = None; process_thread = None
        start_time = time.time(); frame_count = 0; target_sr = 48000
        try:
            logging.info(f"Starting inference for {audio_path}")
            try:
                if not hasattr(self, 'batch_size'): raise AttributeError("Avatar 'batch_size' missing.")
                gst_video_pipeline = GStreamerPipeline(fps=fps)
                gst_audio_pipeline = GStreamerAudio()
                if gst_video_pipeline.process is None or gst_audio_pipeline.process is None: raise RuntimeError("GStreamer failed init.")
            except Exception as e_gst: logging.error(f"Fatal GStreamer init error: {e_gst}"); return

            logging.info("Processing audio...")
            try:
                whisper_feature = audio_processor.audio2feat(audio_path)
                whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
                video_len = len(whisper_chunks) # number of whisper chunks/iterations

                audio_reader = FFmpegAudioReader(audio_path)
                audio_data = audio_reader.read_full_audio() # Get the FULL audio data
                if audio_data is None: raise ValueError("Failed read audio.")
                logging.info(f"Full audio length: {len(audio_data)} samples")

                total_iters = video_len
                logging.info(f"Audio/Feature Chunks: {total_iters}, Input Batch Size: {self.batch_size}")

                # Audio is no longer pre-split
            except Exception as e_audio: logging.error(f"Fatal audio processing error: {e_audio}"); raise

            self.idx = 0
            # --- Pass total_iters to process_frames thread ---
            process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, total_iters, gst_video_pipeline, gst_audio_pipeline, skip_save_images))
            process_thread.daemon = True; process_thread.start()

            logging.info("Starting main inference loop...")
            global unet, vae, pe, timesteps, device
            gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)

            # --- Track audio samples sent ---
            audio_samples_sent = 0
            # ---

            for i, batch_data in enumerate(tqdm(gen, total=total_iters)):
                if batch_data is None or len(batch_data) != 2: continue
                whisper_batch, latent_batch = batch_data
                if whisper_batch is None or latent_batch is None: continue

                try:
                    # --- Prepare Batch ---
                    audio_feature_batch = torch.from_numpy(whisper_batch).to(device=device, dtype=unet.model.dtype, non_blocking=True); audio_feature_batch = pe(audio_feature_batch)
                    if not isinstance(latent_batch, torch.Tensor): latent_batch = torch.stack(latent_batch)
                    latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

                    # --- Inference ---
                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents) # NumPy uint8 NHWC BGR

                    # --- Process VAE Output ---
                    recon_list = []
                    if isinstance(recon, np.ndarray) and recon.ndim == 4 and recon.shape[-1] == 3:
                        recon_list = [f.copy() for f in recon] # Assumes BGR
                    else:
                        logging.warning(f"VAE output unexpected shape/type: {type(recon)}, {getattr(recon, 'shape', 'N/A')}"); continue

                    # --- Check video frames ---
                    if not recon_list or not all(isinstance(f, np.ndarray) for f in recon_list):
                        logging.warning(f"Warning: recon_list empty/invalid iter {i}. Skip put."); continue

                    num_vid_frames = len(recon_list); frame_count += num_vid_frames

                    # --- Calculate and Slice Correct Audio Chunk ---
                    current_audio_chunk = None
                    if fps > 0:
                        expected_audio_samples = int(round(num_vid_frames / fps * target_sr)) # Calculate expected samples first
                        start_sample = audio_samples_sent
                        end_sample = start_sample + expected_audio_samples # Use expected for ideal end
                        end_sample = min(end_sample, len(audio_data)) # Clamp to actual data length

                        if start_sample < end_sample: # Check if there's audio left to slice
                           current_audio_chunk = audio_data[start_sample:end_sample]
                           audio_samples_sent += len(current_audio_chunk) # Update based on actual slice length
                        else:
                            logging.warning(f"Warning: No more audio data to slice at iter {i} (start_sample={start_sample})")
                            current_audio_chunk = np.array([], dtype=np.int16) # Send empty chunk

                        # --- ADDED: Logging for Chunk Size Comparison ---
                        audio_chunk_samples = len(current_audio_chunk)
                        logging.debug(f"DEBUG (inference): Iter {i}: VidFrames={num_vid_frames}, AudSamples={audio_chunk_samples}, ExpAud={expected_audio_samples}, Diff={audio_chunk_samples-expected_audio_samples}")
                        # --- END ADDED ---
                    else:
                        logging.warning("Warning: fps <= 0, cannot calculate audio chunk size."); continue

                    # --- Queue the data ---
                    if current_audio_chunk is not None:
                         res_frame_queue.put((recon_list, current_audio_chunk))
                    # ---

                except Exception as e_loop:
                    # Use f-string for cleaner formatting
                    logging.error(f"\n!!!!! Error main inference loop iter {i} !!!!!")
                    # Use logging.exception to include traceback automatically if needed,
                    # or keep traceback.print_exc() if preferred.
                    # logging.exception("Error details:")
                    traceback.print_exc(); continue # Continue to next iteration

        except Exception as e_inf:
            logging.error(f"\n!!!!! FATAL ERROR inference setup/loop: {e_inf} !!!!!")
            traceback.print_exc()
            # Signal processing thread to terminate early if it started (moved inside finally is safer)

        finally:
            logging.info("\n--- Starting Final Cleanup ---")
            # Signal End to Processing Thread
            # Ensure process_thread exists before putting None in queue in case of early exit
            if 'process_thread' in locals() and process_thread is not None and process_thread.is_alive():
                try:
                    logging.info("Signaling process_frames finish...")
                    res_frame_queue.put(None)
                except Exception as e_put_none:
                    logging.warning(f"Note: Exception putting None sentinel: {e_put_none}")

            # Wait for Processing Thread
            if 'process_thread' in locals() and process_thread is not None:
                logging.info("Waiting process_frames thread join...")
                process_thread.join(timeout=30) # Wait up to 30s
                if process_thread.is_alive(): logging.warning("⚠️ Warning: process_frames thread did not finish.")
                else: logging.info("✅ process_frames thread joined.")
            else:
                logging.info("process_frames thread not started or already finished.")

            # Stop GStreamer
            logging.info("Stopping GStreamer pipelines...")
            if gst_video_pipeline: gst_video_pipeline.stop()
            if gst_audio_pipeline: gst_audio_pipeline.stop()
            logging.info("GStreamer stop commands issued.")

            # Calculate and Print FPS
            logging.info("Calculating final FPS...")
            total_elapsed_time = time.time() - start_time;
            avg_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0
            logging.info("\n========================================")
            logging.info(f" Total elapsed time: {total_elapsed_time:.2f} s")
            logging.info(f" Total VAE frame count: {frame_count}")
            logging.info(f" Final avg FPS (VAE output): {avg_fps:.2f}")
            logging.info("========================================")
            logging.info(">>> Inference method finished.")


# --- Watchdog Event Handler ---
class WavFileHandler(FileSystemEventHandler):
    """Handles filesystem events for the target WAV file."""
    def __init__(self, target_file_path, avatar_instance, fps):
        self.target_file_path = os.path.abspath(target_file_path)
        self.avatar = avatar_instance
        self.fps = fps
        self.last_processed_time = 0
        self.debounce_time = 0.5 # Seconds to wait after modification before processing
        logging.info(f"Handler initialized. Watching for: {self.target_file_path}")
        logging.info(f"Using Avatar ID: {self.avatar.avatar_id}, FPS: {self.fps}")

    def process_wav(self):
        global is_inferencing # Use global flag

        if not inference_lock.acquire(blocking=False):
            logging.warning("Inference already in progress. Skipping trigger.")
            return

        is_inferencing = True # Set flag
        logging.info("--- Inference Lock Acquired ---")
        try:
            # Check file existence and size again before processing
            if not os.path.exists(self.target_file_path):
                logging.warning(f"File {self.target_file_path} disappeared before processing.")
                return
            if os.path.getsize(self.target_file_path) < 1024: # Basic sanity check for size
                 logging.warning(f"File {self.target_file_path} seems too small. Skipping processing.")
                 return

            logging.info(f"Starting inference for: {self.target_file_path}")
            # --- Call the Avatar's inference method ---
            self.avatar.inference(
                audio_path=self.target_file_path,
                out_vid_name="watched_output", # Or generate a dynamic name
                fps=self.fps,
                skip_save_images=True # Assuming we don't need to save intermediate images
            )
            # ------------------------------------------
            logging.info(f"Finished inference for: {self.target_file_path}")

        except Exception as e:
            logging.error(f"Error during inference processing for {self.target_file_path}:")
            logging.error(traceback.format_exc()) # Log full traceback
        finally:
            is_inferencing = False # Reset flag
            inference_lock.release()
            logging.info("--- Inference Lock Released ---")


    def on_modified(self, event):
        """Called when a file or directory is modified."""
        if not event.is_directory and os.path.abspath(event.src_path) == self.target_file_path:
            current_time = time.time()
            logging.info(f"Detected modification: {event.src_path}")

            # Debounce: Check if enough time has passed since the last modification event
            if current_time - getattr(self, '_last_event_time', 0) < self.debounce_time:
                # logging.debug("Debounce active, skipping immediate processing.") # Optional debug log
                setattr(self, '_last_event_time', current_time) # Update last event time
                # Schedule a check after debounce time if not already scheduled
                if not getattr(self, '_debounce_timer', None) or not self._debounce_timer.is_alive():
                     self._debounce_timer = threading.Timer(self.debounce_time, self.check_and_process, args=[current_time])
                     self._debounce_timer.start()
                     logging.debug(f"Scheduled processing check in {self.debounce_time}s.")
            else:
                # If debounce time has passed since last event, process immediately
                setattr(self, '_last_event_time', current_time)
                self.check_and_process(current_time)


    def check_and_process(self, event_time_to_match):
         """ Check if the latest event time matches the one that triggered this check, then process """
         # Ensure we process only based on the *last* modification within the debounce window
         if getattr(self, '_last_event_time', 0) == event_time_to_match:
             logging.info(f"Debounce period ended. Triggering processing for {self.target_file_path}")
             # Run processing in a separate thread to avoid blocking the observer/timer
             thread = threading.Thread(target=self.process_wav, daemon=True)
             thread.start()
         else:
              logging.debug("Skipping scheduled check - newer modification event occurred.")


# --- Main Execution (Watcher Setup) ---
if __name__ == "__main__":
    logging.info("Starting Realtime Stream Sync Watcher...")

    # --- Validate Configuration ---
    if not WATCHED_WAV_FILE_PATH:
        logging.error("❌ Error: WATCHED_WAV_FILE_PATH not set in .env file.")
        sys.exit(1)
    if not os.path.isabs(WATCHED_WAV_FILE_PATH):
         # Attempt to make it absolute based on current dir, but log a warning
         abs_path = os.path.abspath(WATCHED_WAV_FILE_PATH)
         logging.warning(f"⚠️ WATCHED_WAV_FILE_PATH is not absolute. Assuming path relative to script: {abs_path}")
         WATCHED_WAV_FILE_PATH = abs_path
         # Better: require absolute path in .env
         # logging.error("❌ Error: WATCHED_WAV_FILE_PATH must be an absolute path in .env file.")
         # sys.exit(1)

    if not AVATAR_CONFIG_PATH or not os.path.exists(AVATAR_CONFIG_PATH):
        logging.error(f"❌ Error: AVATAR_CONFIG_PATH '{AVATAR_CONFIG_PATH}' not found or not set in .env.")
        sys.exit(1)
    if not AVATAR_ID_TO_USE:
        logging.error("❌ Error: AVATAR_ID_TO_USE not set in .env file.")
        sys.exit(1)

    watch_directory = os.path.dirname(WATCHED_WAV_FILE_PATH)
    if not os.path.isdir(watch_directory):
        logging.error(f"❌ Error: Directory '{watch_directory}' derived from WATCHED_WAV_FILE_PATH does not exist.")
        sys.exit(1)

    # --- Load Avatar Config ---
    try:
        inference_config_all = OmegaConf.load(AVATAR_CONFIG_PATH)
        if AVATAR_ID_TO_USE not in inference_config_all:
            logging.error(f"❌ Error: Avatar ID '{AVATAR_ID_TO_USE}' not found in config file '{AVATAR_CONFIG_PATH}'.")
            logging.error(f"Available IDs: {list(inference_config_all.keys())}")
            sys.exit(1)
        avatar_config = inference_config_all[AVATAR_ID_TO_USE]
        logging.info(f"Loaded config for Avatar ID: {AVATAR_ID_TO_USE}")
    except Exception as e_conf:
        logging.error(f"❌ Error loading avatar config file '{AVATAR_CONFIG_PATH}': {e_conf}")
        sys.exit(1)

    # --- Instantiate the Avatar ---
    try:
        # Get necessary params from the specific avatar's config
        video_path = avatar_config.get("video_path")
        bbox_shift = avatar_config.get("bbox_shift", 0)
        preparation = avatar_config.get("preparation", False) # Default to False if not specified

        if not video_path:
             logging.error(f"❌ Error: 'video_path' not defined for avatar '{AVATAR_ID_TO_USE}' in config.")
             sys.exit(1)

        # Determine batch_size (example: fixed or from config/args if needed later)
        batch_size = 4 # Or load from config if available: avatar_config.get("batch_size", 4)

        logging.info("Instantiating Avatar object...")
        # Pass batch_size to Avatar constructor if needed
        main_avatar = Avatar(
            avatar_id=AVATAR_ID_TO_USE,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=batch_size,
            preparation=preparation
        )
        logging.info("Avatar object instantiated successfully.")

    except SystemExit: # Catch sys.exit calls during Avatar init
        logging.error("❌ Exiting due to error during Avatar initialization.")
        sys.exit(1)
    except Exception as e_avatar_init:
        logging.error(f"❌ Error instantiating Avatar '{AVATAR_ID_TO_USE}':")
        logging.error(traceback.format_exc())
        sys.exit(1)

    # --- Setup Watchdog ---
    event_handler = WavFileHandler(WATCHED_WAV_FILE_PATH, main_avatar, TARGET_FPS)
    observer = Observer()
    observer.schedule(event_handler, watch_directory, recursive=False)

    # --- Start Observer ---
    observer.start()
    logging.info(f"👀 Watcher started for directory: {watch_directory}")
    logging.info(f"👂 Listening for changes to: {WATCHED_WAV_FILE_PATH}")
    logging.info("🚀 Ready to process detected WAV file changes.")
    logging.info("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("🛑 KeyboardInterrupt received. Stopping observer...")
    except Exception as e:
         logging.error(f"An unexpected error occurred in main loop: {e}")
    finally:
        observer.stop()
        observer.join()
        logging.info("✅ Watcher stopped cleanly.")