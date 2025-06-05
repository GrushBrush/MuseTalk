# -*- coding: utf-8 -*-
import ffmpeg
import argparse
import os
import concurrent.futures # For ThreadPoolExecutor
import threading
import queue # <<<<<<<<<<<<<<<< ENSURE queue is imported
import io
from omegaconf import OmegaConf
import subprocess
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm # Ensure tqdm is imported if used
import copy
import json
import traceback      # For detailed error printing in threads
import time
from PIL import Image # Ensure PIL is imported if used
import tempfile
import logging

# --- MODIFICATION: Add platform-specific imports for pipes ---
if sys.platform == "win32":
    import win32pipe
    import win32file
    import pywintypes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MODIFICATION: Remove Watchdog and use STREAM_PIPE_PATH ---
from dotenv import load_dotenv

# --- MuseTalk Specific Imports (Ensure these paths are correct) ---
try:
    from musetalk.utils.utils import get_file_type, get_video_fps, datagen
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    from musetalk.utils.blending import get_image_prepare_material # Assuming this is from MuseTalk
    from musetalk.utils.utils import load_all_model # Assuming this is from MuseTalk
except ImportError as e:
    print(f"Error importing MuseTalk utilities: {e}")
    print("Please ensure the MuseTalk library is correctly installed and accessible in your Python environment.")
    sys.exit(1)

import shutil

# --- Configuration Loading ---
load_dotenv()
STREAM_PIPE_PATH = os.getenv("STREAM_PIPE_PATH")
AVATAR_CONFIG_PATH = os.getenv("AVATAR_CONFIG_PATH", "configs/inference/realtime.yaml")
AVATAR_ID_TO_USE = os.getenv("AVATAR_ID_TO_USE")
try:
    TARGET_FPS = int(os.getenv("TARGET_FPS", "25"))
except ValueError:
    print("Warning: TARGET_FPS in .env is not a valid integer. Using default 25.")
    TARGET_FPS = 25

# --- Global Variables ---
inference_lock = threading.Lock()
# <<<<<<<<<<<<<<<< NEW: Define the shared queue globally <<<<<<<<<<<<<<<<
# maxsize controls how many audio segments can buffer if inference is slow.
# Adjust based on typical audio segment processing time and memory.
opus_input_queue = queue.Queue(maxsize=10)


# --- PyTorch Device Setup ---
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logging.info("--- PyTorch Device Information ---")
if cuda_available:
    logging.info("✅ CUDA (GPU) detected by PyTorch.")
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"✅ Using GPU: {gpu_name}")
    except Exception as e:
        logging.warning(f"⚠️ Could not retrieve GPU name: {e}")
    logging.info(f"✅ Selected device: {device}")
else:
    logging.info("❌ CUDA (GPU) not available or not detected by PyTorch.")
    logging.info(f"✅ Selected device: {device}")
logging.info("-------------------------------")

# --- Load Models ---
logging.info("Loading models...")
try:
    # Ensure load_all_model correctly returns these components
    audio_processor, vae, unet, pe = load_all_model()
    logging.info("Models loaded.")
except Exception as e:
    logging.error(f"Fatal error loading models: {e}")
    logging.error(traceback.format_exc())
    sys.exit(1)

# --- Set Precision ---
logging.info("Setting model precision to half (FP16) on sub-modules (if applicable)...")
try:
    if hasattr(pe, 'to'): pe = pe.to(device)
    if hasattr(pe, 'half'): pe = pe.half() # Apply half precision if method exists
    else: logging.debug("PE module does not have .half() method or is not a PyTorch module.")

    if hasattr(vae, 'vae') and hasattr(vae.vae, 'to'): # Accessing nested vae
        vae.vae = vae.vae.to(device)
        if hasattr(vae.vae, 'half'): vae.vae = vae.vae.half()
        else: logging.debug("VAE.vae module does not have .half() method.")
    elif hasattr(vae, 'to'): # If vae itself is the module
        vae = vae.to(device)
        if hasattr(vae, 'half'): vae = vae.half()
        else: logging.debug("VAE module does not have .half() method.")
    else: logging.warning("VAE model structure not as expected for precision setting.")

    if hasattr(unet, 'model') and hasattr(unet.model, 'to'): # Accessing nested model
        unet.model = unet.model.to(device)
        if hasattr(unet.model, 'half'): unet.model = unet.model.half()
        else: logging.debug("UNET.model module does not have .half() method.")
    elif hasattr(unet, 'to'): # If unet itself is the module
        unet = unet.to(device)
        if hasattr(unet, 'half'): unet = unet.half()
        else: logging.debug("UNET module does not have .half() method.")
    else: logging.warning("UNET model structure not as expected for precision setting.")

    logging.info("Precision setting attempt complete.")
except Exception as e_prec:
    logging.warning(f"Warning: Error setting model precision/device: {e_prec}")
timesteps = torch.tensor([0], device=device) # Assuming timesteps is used by UNET

# --- FFmpegAudioReader Class ---
class FFmpegAudioReader:
    def __init__(self, audio_file_path_or_bytesio):
        self.audio_input = audio_file_path_or_bytesio # Can be path or BytesIO
        self.is_file_path = isinstance(audio_file_path_or_bytesio, str)
        try:
            input_source = self.audio_input if self.is_file_path else 'pipe:0'
            logging.info(f"Probing audio source: {'file' if self.is_file_path else 'pipe'}")

            probe_input_args = {}
            if not self.is_file_path: # If it's BytesIO, pass data via stdin
                # For probing BytesIO, we might need to write it to a temp file first if ffmpeg.probe can't handle it directly
                # Or, assume fixed parameters if probing is too complex for in-memory data directly with ffmpeg.probe
                # For simplicity, let's assume we know the format or make an attempt
                logging.warning("Probing in-memory data directly with ffmpeg.probe can be tricky; assuming Opus format for piped data.")
                # Alternatively, save to temp file for probing if necessary, then delete.
                # For now, we'll proceed assuming it's readable by ffmpeg.
                pass


            probe = ffmpeg.probe(input_source, **probe_input_args)

            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            if audio_stream is None:
                raise RuntimeError(f"No audio stream found in {'file' if self.is_file_path else 'piped data'}")

            # Default to target_sr if not found, but it should be.
            self.sample_rate = int(audio_stream.get('sample_rate', '48000'))
            self.channels = int(audio_stream.get('channels', '2')) # Default to 2 if not found
            logging.info(f"Detected Sample Rate: {self.sample_rate}, Channels: {self.channels} from source.")

        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode(errors='ignore') if e.stderr else 'Unknown FFmpeg error during probe'
            logging.error(f"❌ Error probing audio source: {stderr_output}")
            raise
        except Exception as e:
            logging.error(f"❌ Unexpected error during FFmpeg probe initialization: {e}")
            raise

    def read_full_audio(self):
        logging.info(f"Reading and converting audio to PCM s16le with FFmpeg...")
        target_sr = 48000
        target_ac = 2 # MuseTalk often expects stereo
        target_format = "s16le" # Signed 16-bit little-endian PCM

        try:
            input_args = {'ar': str(target_sr), 'ac': str(target_ac), 'f': target_format}
            process_input = self.audio_input

            # If input is BytesIO, we need to pass its content to ffmpeg's stdin
            ffmpeg_process = ffmpeg.input(process_input, format='opus' if not self.is_file_path else None) # Assume opus if piped
            ffmpeg_process = ffmpeg_process.output('pipe:', **input_args)
            
            stdout_data, stderr_data = ffmpeg_process.run(capture_stdout=True, capture_stderr=True, input=self.audio_input.getvalue() if not self.is_file_path else None)

            if stderr_data:
                logging.debug(f"FFmpeg conversion stderr: {stderr_data.decode(errors='ignore')}")

            if not stdout_data:
                logging.error(f"❌ Failed to read/convert audio: FFmpeg produced no PCM data!")
                return None

            # Reshape based on target channels
            audio_data = np.frombuffer(stdout_data, dtype=np.int16).reshape(-1, target_ac)
            logging.info(f"✅ Read/converted audio: {len(audio_data)} samples at {target_sr}Hz, {target_ac}ch.")
            return audio_data
        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode(errors='ignore') if e.stderr else 'Unknown FFmpeg error during conversion'
            logging.error(f"❌ FFmpeg error during audio read/conversion: {stderr_output}")
            return None
        except Exception as e:
            logging.error(f"❌ Unexpected error during FFmpeg audio read/conversion: {e}")
            logging.error(traceback.format_exc())
            return None


# --- GStreamer Classes ---
class GStreamerPipeline:
    def __init__(self, width=720, height=1280, fps=25, host="127.0.0.1", port=5000):
        self.width = width; self.height = height; self.fps = fps; self.host = host; self.port = port; self.process = None
        # Adjusted pipeline for potentially better compatibility and performance
        pipeline_str_fdsrc = (
            f"fdsrc fd=0 ! videoparse format=bgr width={self.width} height={self.height} framerate={self.fps}/1 ! "
            "queue leaky=downstream max-size-buffers=5 ! videoconvert ! " # Added leaky queue
            f"video/x-raw,format=NV12,width={self.width},height={self.height},framerate={self.fps}/1 ! queue max-size-buffers=5 ! "
            f"nvh265enc preset=low-latency-hq rc-mode=cbr bitrate=8000 gop-size=30 ! " # bitrate=8000 for 8 Mbps
            "h265parse ! rtph265pay config-interval=1 ! "
            f"udpsink host={self.host} port={self.port} sync=false async=false" # async=false might be important
        )
        logging.info(f"Attempting to start GStreamer video pipeline: {pipeline_str_fdsrc}")
        try:
            self.process = subprocess.Popen(f"gst-launch-1.0 -v {pipeline_str_fdsrc}", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0)
            logging.info(f"✅ GStreamer process launched (PID: {self.process.pid}) Pipeline initialization pending for {self.width}x{self.height} @ {self.fps}fps to {self.host}:{self.port}")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_VID_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_VID_ERR"), daemon=True).start()
        except Exception as e:
            logging.error(f"❌ Failed to start GStreamer video pipeline: {e}")
            self.process = None

    def _log_stream(self, stream, prefix):
        try:
            for line_bytes in iter(stream.readline, b''):
                line = line_bytes.decode(errors='ignore').strip()
                if line: logging.info(f"[{prefix}]: {line}")
        except Exception as e:
            logging.warning(f"Error in GStreamer log thread ({prefix}): {e}")
        finally:
            try: stream.close()
            except Exception: pass

    def send_frame(self, frame):
        if not self.process or self.process.stdin is None or self.process.stdin.closed:
            # logging.warning("GStreamer video pipeline not available or stdin closed.")
            return False
        try:
            # Ensure frame is C-contiguous
            if not frame.flags['C_CONTIGUOUS']: frame = np.ascontiguousarray(frame, dtype=np.uint8)
            # Resize if necessary (should match pipeline dimensions)
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if frame.dtype != np.uint8: frame = frame.astype(np.uint8)

            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            return True
        except BrokenPipeError:
            logging.error("❌ GStreamer video pipeline: Broken pipe while sending frame. Pipeline might have crashed.")
            self.stop() # Attempt to clean up
            return False
        except Exception as e:
            logging.error(f"❌ Error pushing video frame to GStreamer: {e}")
            return False

    def stop(self):
        if self.process:
            logging.info(f"Stopping GStreamer video pipeline (PID: {self.process.pid})...")
            proc_to_stop = self.process
            self.process = None # Mark as stopped immediately
            if proc_to_stop.stdin and not proc_to_stop.stdin.closed:
                try: proc_to_stop.stdin.close()
                except Exception as e_stdin: logging.warning(f"Error closing GStreamer video stdin: {e_stdin}")
            
            # Terminate gracefully first
            proc_to_stop.terminate()
            try:
                proc_to_stop.wait(timeout=2) # Wait a bit for graceful termination
                logging.info(f"✅ GStreamer video process (PID: {proc_to_stop.pid}) terminated.")
            except subprocess.TimeoutExpired:
                logging.warning(f"⚠️ GStreamer video process (PID: {proc_to_stop.pid}) did not terminate gracefully, killing...")
                proc_to_stop.kill()
                try: proc_to_stop.wait(timeout=2)
                except subprocess.TimeoutExpired: logging.error(f"GStreamer video process (PID: {proc_to_stop.pid}) failed to die even after kill.")
                else: logging.info(f"✅ GStreamer video process (PID: {proc_to_stop.pid}) killed.")
            except Exception as e_wait:
                logging.error(f"Error waiting for GStreamer video process (PID: {proc_to_stop.pid}): {e_wait}")
        else:
            logging.info("GStreamer video pipeline already stopped or not started.")


class GStreamerAudio:
    def __init__(self, host="127.0.0.1", port=5001, sample_rate=48000, channels=2):
        self.host = host; self.port = port; self.process = None
        self.sample_rate = sample_rate; self.channels = channels
        # Using opusenc for encoding
        pipeline_str = (
           f"fdsrc fd=0 do-timestamp=true ! audio/x-raw,format=S16LE,channels={self.channels},rate={self.sample_rate},layout=interleaved ! "
            "queue leaky=downstream max-size-buffers=30 max-size-time=1500000000 ! audioconvert ! audioresample ! " # 1.5 second buffer, leaky
            "opusenc bitrate=64000 complexity=2 frame-size=20 ! rtpopuspay pt=97 ! "
            f"udpsink host={self.host} port={self.port} sync=false async=false"
        )
        logging.info(f"Attempting to start GStreamer audio pipeline: {pipeline_str}")
        try:
            self.process = subprocess.Popen(f"gst-launch-1.0 -v {pipeline_str}", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, bufsize=0)
            logging.info(f"✅ GStreamer audio pipeline potentially started (PID: {self.process.pid}) for {self.sample_rate}Hz, {self.channels}ch to {self.host}:{self.port}")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_AUD_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_AUD_ERR"), daemon=True).start()
        except Exception as e:
            logging.error(f"❌ Failed to start GStreamer audio pipeline: {e}")
            self.process = None
            
    def _log_stream(self, stream, prefix):
        try:
            for line_bytes in iter(stream.readline, b''):
                line = line_bytes.decode(errors='ignore').strip()
                if line: logging.debug(f"[{prefix}]: {line}")
        except Exception as e:
            logging.warning(f"Error in GStreamer log thread ({prefix}): {e}")
        finally:
            try: stream.close()
            except Exception: pass

    def send_audio(self, audio_data_pcm): # Expects PCM s16le data
        if not self.process or self.process.stdin is None or self.process.stdin.closed:
            # logging.warning("GStreamer audio pipeline not available or stdin closed.")
            return False
        try:
            if audio_data_pcm.dtype != np.int16:
                audio_data_pcm = audio_data_pcm.astype(np.int16)
            # Ensure data is C-contiguous
            if not audio_data_pcm.flags['C_CONTIGUOUS']:
                audio_data_pcm = np.ascontiguousarray(audio_data_pcm, dtype=np.int16)

            self.process.stdin.write(audio_data_pcm.tobytes())
            self.process.stdin.flush()
            return True
        except BrokenPipeError:
            logging.error("❌ GStreamer audio pipeline: Broken pipe while sending audio. Pipeline might have crashed.")
            self.stop() # Attempt to clean up
            return False
        except Exception as e:
            logging.error(f"❌ Error pushing audio chunk to GStreamer: {e}")
            return False

    def stop(self):
        if self.process:
            logging.info(f"Stopping GStreamer audio pipeline (PID: {self.process.pid})...")
            proc_to_stop = self.process
            self.process = None # Mark as stopped
            if proc_to_stop.stdin and not proc_to_stop.stdin.closed:
                try: proc_to_stop.stdin.close()
                except Exception as e_stdin: logging.warning(f"Error closing GStreamer audio stdin: {e_stdin}")
            
            proc_to_stop.terminate()
            try:
                proc_to_stop.wait(timeout=2)
                logging.info(f"✅ GStreamer audio process (PID: {proc_to_stop.pid}) terminated.")
            except subprocess.TimeoutExpired:
                logging.warning(f"⚠️ GStreamer audio process (PID: {proc_to_stop.pid}) did not terminate gracefully, killing...")
                proc_to_stop.kill()
                try: proc_to_stop.wait(timeout=2)
                except subprocess.TimeoutExpired: logging.error(f"GStreamer audio process (PID: {proc_to_stop.pid}) failed to die even after kill.")
                else: logging.info(f"✅ GStreamer audio process (PID: {proc_to_stop.pid}) killed.")
            except Exception as e_wait:
                logging.error(f"Error waiting for GStreamer audio process (PID: {proc_to_stop.pid}): {e_wait}")
        else:
            logging.info("GStreamer audio pipeline already stopped or not started.")

# --- Helper Functions ---
def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    logging.info(f"Extracting frames from {vid_path} to {save_path}...")
    cap = cv2.VideoCapture(vid_path); count = 0; frame_idx = 0
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file: {vid_path}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Video ended or read error.")
            break
        if frame_idx % 1 == 0: # Process every frame
            filename = f"{str(count).zfill(8)}{ext}" # Ensure consistent naming
            filepath = os.path.join(save_path, filename)
            try:
                cv2.imwrite(filepath, frame)
                count += 1
            except Exception as e:
                logging.error(f"Error writing frame {count}: {e}")
                break # Stop if writing fails
            if count >= cut_frame:
                logging.info(f"Reached cut_frame limit: {cut_frame}")
                break
        frame_idx += 1
    cap.release()
    logging.info(f"Finished extracting {count} frames to {save_path}.")

def osmakedirs(path_list):
    for path in path_list:
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directory {path}: {e}")

# --- Avatar Class and Methods --- EDS
# Ensure these imports are at the top of your file:
# import os, shutil, pickle, json, glob, logging, traceback, cv2, numpy as np, torch, io, tempfile, time, queue, threading, concurrent.futures
# from tqdm import tqdm
# from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
# from musetalk.utils.blending import get_image_prepare_material
# Global variables like vae, unet, pe, device, audio_processor should be accessible

class Avatar: # NO @torch.no_grad() here
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        logging.info(f"Initializing Avatar: {avatar_id}")
        self.avatar_id = str(avatar_id) 
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.preparation = preparation

        self.avatar_base_path = os.path.join("./results/avatars", self.avatar_id)
        self.full_imgs_path = os.path.join(self.avatar_base_path, "full_imgs")
        self.coords_path = os.path.join(self.avatar_base_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_base_path, "latents.pt")
        self.mask_out_path = os.path.join(self.avatar_base_path, "mask") # Using "mask" as per your directory
        self.mask_coords_path = os.path.join(self.avatar_base_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_base_path, "avatar_info.json")

        self.avatar_info_runtime = {"avatar_id": self.avatar_id, "video_path": self.video_path, "bbox_shift": self.bbox_shift}
        self.idx = 0

        self.input_latent_list_cycle = []
        self.coord_list_cycle = []
        self.frame_list_cycle = []
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        self.init_avatar_data()
        logging.info(f"Avatar initialization complete for {self.avatar_id}.")

    def init_avatar_data(self):
        if self.preparation:
            if os.path.exists(self.avatar_base_path):
                try:
                    response = input(f"Avatar '{self.avatar_id}' data exists at '{self.avatar_base_path}'. Re-create all material? (y/n): ").strip().lower()
                    if response == "y":
                        logging.info(f"User chose to re-create. Removing: {self.avatar_base_path}")
                        if os.path.isdir(self.avatar_base_path): shutil.rmtree(self.avatar_base_path)
                        self._prepare_material_core()
                    else:
                        logging.info("User chose not to re-create. Loading existing data...")
                        self._reload_prepared_data()
                except Exception as e_input:
                    logging.warning(f"Input error for re-creation: {e_input}. Assuming 'n'.")
                    self._reload_prepared_data()
            else:
                logging.info(f"Avatar path {self.avatar_base_path} not exist. Preparing material...")
                self._prepare_material_core()
        else:
            required_paths = [self.coords_path, self.latents_out_path, self.mask_coords_path, self.mask_out_path, self.full_imgs_path]
            missing_paths = [p for p in required_paths if not os.path.exists(p)]
            if missing_paths:
                logging.error(f"Error: Not all required data found in '{self.avatar_base_path}' (preparation=False). Missing: {missing_paths}")
                sys.exit(1)
            else:
                logging.info("Preparation=False. Loading existing data...")
                self._reload_prepared_data()
                try:
                    if os.path.exists(self.avatar_info_path):
                        with open(self.avatar_info_path, "r") as f: avatar_info_disk = json.load(f)
                        if avatar_info_disk.get('bbox_shift') != self.avatar_info_runtime['bbox_shift']:
                            logging.error(f"Bbox_shift mismatch. Current: {self.avatar_info_runtime['bbox_shift']}, Prepared: {avatar_info_disk.get('bbox_shift')}. Re-prepare or use original.")
                            sys.exit(1)
                    else: logging.warning(f"Avatar info file '{self.avatar_info_path}' not found for consistency check.")
                except Exception as e_info: logging.warning(f"Could not check avatar info consistency: {e_info}")

    @torch.no_grad()
    def _reload_prepared_data(self):
        logging.info(f"Reloading prepared data from: {self.avatar_base_path}")
        try:
            loaded_latents = torch.load(self.latents_out_path, map_location='cpu')
            self.input_latent_list_cycle = list(loaded_latents) if isinstance(loaded_latents, torch.Tensor) else loaded_latents
            with open(self.coords_path, 'rb') as f: self.coord_list_cycle = pickle.load(f)
            with open(self.mask_coords_path, 'rb') as f: self.mask_coords_list_cycle = pickle.load(f)

            num_items = len(self.coord_list_cycle)
            if num_items == 0: raise ValueError("No coordinates loaded.")

            frame_files = [os.path.join(self.full_imgs_path, f"{str(i).zfill(8)}.png") for i in range(num_items)]
            self.frame_list_cycle = read_imgs(frame_files)
            mask_files = [os.path.join(self.mask_out_path, f"{str(i).zfill(8)}.png") for i in range(num_items)]
            self.mask_list_cycle = read_imgs(mask_files)

            data_map = {"Latents":self.input_latent_list_cycle, "Coords":self.coord_list_cycle, "Frames":self.frame_list_cycle, "Masks":self.mask_list_cycle, "MaskCoords":self.mask_coords_list_cycle}
            list_lengths = {name: len(data_list) for name, data_list in data_map.items() if isinstance(data_list, list)} # Check if list before len
            
            mismatched_or_empty = False
            for name, data_list in data_map.items():
                if not isinstance(data_list, list) or not data_list: # Check if it's not a list or is an empty list
                    logging.error(f"Data list '{name}' is empty or invalid type after loading.")
                    mismatched_or_empty = True; break
            if not mismatched_or_empty and len(set(list_lengths.values())) > 1:
                logging.error(f"Data lists lengths mismatch: {list_lengths}")
                mismatched_or_empty = True
            if mismatched_or_empty: raise ValueError("Failed to load/validate all data. Lists empty/mismatched.")
            logging.info(f"Reloaded and validated {num_items} items for avatar cycles.")
        except Exception as e:
            logging.error(f"Error reloading prepared data for {self.avatar_id}: {e}"); traceback.print_exc()
            raise SystemExit(f"Exiting: Failed reload for {self.avatar_id}. Consider re-preparation.")

    @torch.no_grad()
    def _prepare_material_core(self):
        logging.info(f"--- Preparing new material for avatar: {self.avatar_id} ---")
        osmakedirs([self.avatar_base_path, self.full_imgs_path, self.mask_out_path])
        with open(self.avatar_info_path, "w") as f: json.dump(self.avatar_info_runtime, f, indent=4)

        if os.path.isfile(self.video_path): video2imgs(self.video_path, self.full_imgs_path, ext='.png')
        elif os.path.isdir(self.video_path):
            source_files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff','.webp'))])
            if not source_files: logging.error(f"No images in source dir: {self.video_path}"); sys.exit(1)
            for i, filename in enumerate(tqdm(source_files, desc="Copying frames")):
                try:
                    img = cv2.imread(os.path.join(self.video_path, filename))
                    if img is None: logging.warning(f"Skip {filename}: not readable."); continue
                    cv2.imwrite(os.path.join(self.full_imgs_path, f"{str(i).zfill(8)}.png"), img)
                except Exception as e: logging.warning(f"Error processing {filename}: {e}")
        else: logging.error(f"Invalid video_path: {self.video_path}"); sys.exit(1)

        source_images = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if not source_images: logging.error(f"No PNGs in {self.full_imgs_path}"); sys.exit(1)

        coords, frames = get_landmark_and_bbox(source_images, self.bbox_shift)
        logging.info(f"Initial landmarks/bboxes: {len(coords)} items.")

        valid_l, valid_c, valid_f = [], [], []
        coord_ph_val = coord_placeholder()
        global vae # Ensure accessible
        for i, (b, fr) in enumerate(tqdm(zip(coords, frames), total=len(coords), desc="VAE Encoding")):
            if b is None or np.array_equal(b, coord_ph_val) or fr is None: continue
            x1c,y1c,x2c,y2c = b; y1c,y2c=int(round(y1c)),int(round(y2c)); x1c,x2c=int(round(x1c)),int(round(x2c))
            hf,wf=fr.shape[:2]; y1c,y2c=max(0,y1c),min(hf,y2c); x1c,x2c=max(0,x1c),min(wf,x2c)
            if x1c>=x2c or y1c>=y2c: continue
            crop = fr[y1c:y2c,x1c:x2c];
            if crop.size==0: continue
            try:
                resized = cv2.resize(crop,(256,256),interpolation=cv2.INTER_LANCZOS4)
                lat = vae.get_latents_for_unet(resized).cpu()
                valid_l.append(lat); valid_c.append(b); valid_f.append(fr)
            except Exception as e: logging.warning(f"VAE error frame {i}: {e}")
        if not valid_l: logging.error("No valid latents after VAE."); sys.exit(1)
        
        tmp_l, tmp_c, tmp_f = valid_l+valid_l[::-1], valid_c+valid_c[::-1], valid_f+valid_f[::-1]
        num_tmp = len(tmp_f)

        self.input_latent_list_cycle, self.coord_list_cycle, self.frame_list_cycle = [],[],[]
        self.mask_list_cycle, self.mask_coords_list_cycle = [],[]
        
        # Clean and remake directories for final filtered output
        for p in [self.full_imgs_path, self.mask_out_path]:
            if os.path.exists(p): shutil.rmtree(p)
            osmakedirs([p])
        
        global get_image_prepare_material # Ensure accessible
        for i, (fr_d, bb_d, lat_d) in enumerate(tqdm(zip(tmp_f, tmp_c, tmp_l), total=num_tmp, desc="Final Prep & Masking")):
            try:
                mask_out, crop_box_out = get_image_prepare_material(fr_d, bb_d)
                if mask_out is None or crop_box_out is None: logging.warning(f"Mask/crop fail item {i}, skip."); continue
                
                idx_str = str(len(self.frame_list_cycle)).zfill(8) # Use current length for sequential naming
                cv2.imwrite(os.path.join(self.full_imgs_path, f"{idx_str}.png"), fr_d)
                cv2.imwrite(os.path.join(self.mask_out_path, f"{idx_str}.png"), mask_out)

                self.frame_list_cycle.append(fr_d); self.coord_list_cycle.append(bb_d)
                self.input_latent_list_cycle.append(lat_d); self.mask_list_cycle.append(mask_out)
                self.mask_coords_list_cycle.append(crop_box_out)
            except Exception as e: logging.warning(f"Error processing item {i} final prep: {e}")
        
        if not self.frame_list_cycle: logging.error("No items survived full prep."); sys.exit(1)
        
        with open(self.coords_path,'wb') as f: pickle.dump(self.coord_list_cycle,f)
        with open(self.mask_coords_path,'wb') as f: pickle.dump(self.mask_coords_list_cycle,f)
        torch.save(torch.stack(self.input_latent_list_cycle) if self.input_latent_list_cycle else torch.empty(0), self.latents_out_path) # Handle empty list for stack
        logging.info(f"--- Material prep complete. Final cycle items: {len(self.frame_list_cycle)} ---")

        

    # In realtime_stream_sync.py, inside the Avatar class

    @torch.no_grad()
    def inference(self, audio_path_or_bytesio, out_vid_name_unused, target_fps, skip_save_images_unused):
        vae_to_blend_q = queue.Queue(maxsize=max(2, self.batch_size)) 
        gst_video_pipeline = None
        gst_audio_pipeline = None
        frame_blending_and_sending_thread = None
        start_time_current_inference = time.time()
        total_vae_frames_generated_this_run = 0
        run_id = os.path.basename(audio_path_or_bytesio) if isinstance(audio_path_or_bytesio, str) else f"stream_{int(time.time())}"
        logging.info(f"🎬 Starting Avatar.inference for run ID: {run_id} (Target FPS: {target_fps})")

        try:
            if not (hasattr(self, 'batch_size') and self.batch_size > 0):
                raise AttributeError(f"[{run_id}] Avatar 'batch_size' invalid: {getattr(self, 'batch_size', 'Not Set')}")

            gst_output_width, gst_output_height = 720, 1280
            gst_video_pipeline = GStreamerPipeline(width=gst_output_width, height=gst_output_height, fps=target_fps)
            gst_audio_pipeline = GStreamerAudio(sample_rate=48000, channels=2) 
            if gst_video_pipeline.process is None or gst_audio_pipeline.process is None:
                raise RuntimeError(f"[{run_id}] GStreamer pipeline(s) init failed.")

            input_for_ffmpeg = io.BytesIO(audio_path_or_bytesio) if isinstance(audio_path_or_bytesio, bytes) else audio_path_or_bytesio
            
            path_for_features = None
            temp_file_for_features_path = None
            if isinstance(input_for_ffmpeg, io.BytesIO):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".opus") as tmp_f:
                    tmp_f.write(input_for_ffmpeg.getvalue()); temp_file_for_features_path = tmp_f.name
                path_for_features = temp_file_for_features_path
            else: path_for_features = input_for_ffmpeg
            
            global audio_processor
            whisper_feature = audio_processor.audio2feat(path_for_features)
            whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=target_fps)

            audio_reader = FFmpegAudioReader(input_for_ffmpeg)
            full_audio_pcm = audio_reader.read_full_audio()
            if temp_file_for_features_path and os.path.exists(temp_file_for_features_path):
                try: os.unlink(temp_file_for_features_path)
                except Exception as e_del_temp: logging.warning(f"Could not delete temp audio file {temp_file_for_features_path}: {e_del_temp}")
            
            if full_audio_pcm is None or full_audio_pcm.size == 0:
                raise ValueError(f"[{run_id}] PCM audio is empty after FFmpeg read.")
            
            actual_total_audio_samples = len(full_audio_pcm)

            # VVVVVVVVVVVVVVVVVVVV NEW A/V DURATION SYNC LOGIC VVVVVVVVVVVVVVVVVVVV
            actual_audio_duration_sec = actual_total_audio_samples / gst_audio_pipeline.sample_rate
            # Calculate the maximum number of video frames the audio can support
            max_possible_frames = int(actual_audio_duration_sec * target_fps)
            
            original_planned_frames = len(whisper_chunks)

            # If MuseTalk plans a video much longer than the audio, truncate the plan.
            # Allow a small buffer (e.g., 10%) in case of minor timing variations.
            if original_planned_frames > max_possible_frames * 1.1:
                logging.warning(f"[{run_id}] MuseTalk planned for {original_planned_frames} frames, but audio duration ({actual_audio_duration_sec:.2f}s) only supports ~{max_possible_frames} frames.")
                logging.warning(f"[{run_id}] Truncating video plan to {max_possible_frames} frames to match audio duration.")
                whisper_chunks = whisper_chunks[:max_possible_frames]
            
            num_frames_to_generate = len(whisper_chunks)
            if num_frames_to_generate == 0:
                logging.warning(f"[{run_id}] No frames to generate after A/V sync check. Skipping.")
                return
            
            # Correctly calculate the number of batches for the accurate progress bar
            num_vae_batches = (num_frames_to_generate + self.batch_size - 1) // self.batch_size
            logging.info(f"[{run_id}] PCM: {actual_total_audio_samples} samples. Video frames to generate: {num_frames_to_generate}. VAE Batches: {num_vae_batches}")
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            self.idx = 0
            frame_blending_and_sending_thread = threading.Thread(
                target=self.process_frames, 
                args=(vae_to_blend_q, num_vae_batches, gst_video_pipeline, gst_audio_pipeline, True, target_fps),
                daemon=True, name=f"FrameProcessor_{run_id}"
            )
            frame_blending_and_sending_thread.start()

            global unet, vae, pe, timesteps, device
            data_gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
            audio_samples_allocated_count = 0

            # Use the corrected num_vae_batches for the tqdm total
            for i, batch_data in enumerate(tqdm(data_gen, total=num_vae_batches, desc=f"VAE/UNET [{run_id}]")):
                if not frame_blending_and_sending_thread.is_alive(): logging.error(f"[{run_id}] FrameProc died. Halt VAE."); break
                if batch_data is None or len(batch_data)!=2: continue
                w_batch, l_batch = batch_data
                if w_batch is None or l_batch is None: continue

                audio_feat = pe(torch.from_numpy(w_batch).to(device, dtype=unet.model.dtype))
                latent_in = torch.stack(l_batch).to(device,dtype=unet.model.dtype) if not isinstance(l_batch,torch.Tensor) else l_batch.to(device,dtype=unet.model.dtype)
                
                pred_lats = unet.model(latent_in, timesteps, encoder_hidden_states=audio_feat).sample
                vae_out = vae.decode_latents(pred_lats)
                
                proc_frames = []
                if isinstance(vae_out, np.ndarray) and vae_out.ndim==4: proc_frames = [f.copy() for f in vae_out]
                elif isinstance(vae_out, list) and vae_out: proc_frames = [f for f in vae_out if isinstance(f, np.ndarray)]
                if not proc_frames: logging.warning(f"[{run_id}] No valid frames from VAE iter {i}. Skip."); continue
                
                total_vae_frames_generated_this_run += len(proc_frames)
                num_vid_f_batch = len(proc_frames)
                
                aud_chunk_pcm = np.array([], dtype=np.int16)
                num_aud_ch = gst_audio_pipeline.channels
                if target_fps > 0:
                    exp_aud_smp = int(round(num_vid_f_batch / target_fps * gst_audio_pipeline.sample_rate))
                    s_idx, e_idx = audio_samples_allocated_count, audio_samples_allocated_count + exp_aud_smp
                    act_e_idx = min(e_idx, actual_total_audio_samples)
                    if s_idx < act_e_idx:
                        aud_chunk_pcm = full_audio_pcm[s_idx:act_e_idx].copy()
                        if aud_chunk_pcm.ndim==1 and num_aud_ch==2: aud_chunk_pcm = np.ascontiguousarray(np.column_stack((aud_chunk_pcm,aud_chunk_pcm)))
                        elif aud_chunk_pcm.ndim==2 and aud_chunk_pcm.shape[1]!=num_aud_ch: aud_chunk_pcm = np.zeros((exp_aud_smp, num_aud_ch),dtype=np.int16)
                        audio_samples_allocated_count += (act_e_idx - s_idx)
                    else: 
                        silent_len = exp_aud_smp if exp_aud_smp > 0 else 1024
                        aud_chunk_pcm = np.zeros((silent_len, num_aud_ch),dtype=np.int16)
                        if s_idx < actual_total_audio_samples: audio_samples_allocated_count += len(aud_chunk_pcm)
                else: aud_chunk_pcm = np.zeros((1024, num_aud_ch), dtype=np.int16)
                
                try: vae_to_blend_q.put((proc_frames, aud_chunk_pcm), timeout=5.0)
                except queue.Full: logging.error(f"[{run_id}] VAE-to-Blend Q full iter {i}. Halting."); break
            
            if audio_samples_allocated_count < actual_total_audio_samples:
                logging.warning(f"[{run_id}] PCM left after VAE loop: {actual_total_audio_samples - audio_samples_allocated_count} samples.")

        except Exception as e:
            logging.error(f"\n!!!!! [{run_id}] CRITICAL ERROR in Avatar.inference: {e} !!!!!")
            logging.error(traceback.format_exc())
            
        finally:
            # ... (The finally block remains the same as the previous version with the colored logging) ...
            logging.info(f"\n--- [{run_id}] Final Cleanup for Avatar.inference ---")
            if 'vae_to_blend_q' in locals() and vae_to_blend_q is not None:
                try: vae_to_blend_q.put(None, block=False)
                except queue.Full: logging.warning(f"[{run_id}] VAE-to-Blend queue was full when trying to send None sentinel.")
            if frame_blending_and_sending_thread and frame_blending_and_sending_thread.is_alive():
                frame_blending_and_sending_thread.join(timeout=30.0)
            if gst_video_pipeline: gst_video_pipeline.stop()
            if gst_audio_pipeline: gst_audio_pipeline.stop()
            elapsed = time.time() - start_time_current_inference
            avg_fps = total_vae_frames_generated_this_run / elapsed if elapsed > 0 else 0
            YELLOW = '\033[93m'; RESET_COLOR = '\033[0m'
            summary_string = f"==== {run_id} Summary: Elapsed={elapsed:.2f}s | VAE Frames={total_vae_frames_generated_this_run} | Avg VAE FPS={avg_fps:.2f} (Target was {target_fps}) ===="
            logging.info(f"{YELLOW}{summary_string}{RESET_COLOR}")
            logging.info(f">>> Avatar.inference method finished for {run_id}.")

    def process_frames(self, res_frame_q, video_len_in_batches, gst_video_pipeline, gst_audio_pipeline, skip_save_images, target_fps):
        cpu_cores = os.cpu_count() or 4 
        num_workers = max(1, cpu_cores // 2, 4)

        # Ensure a run_id or similar identifier for logging if desired, e.g. from self.avatar_id
        run_id_logging_prefix = f"[{self.avatar_id} PrcFrames]" 
        logging.info(f"{run_id_logging_prefix} Started with {num_workers} workers for {video_len_in_batches} VAE batches.")

        required_lists = [
            self.coord_list_cycle, self.frame_list_cycle,
            self.mask_list_cycle, self.mask_coords_list_cycle
        ]
        if not all(lst and len(lst) > 0 for lst in required_lists):
            logging.error(f"{run_id_logging_prefix} Critical Error: Avatar reference data not loaded or empty.")
            if gst_video_pipeline: gst_video_pipeline.stop()
            if gst_audio_pipeline: gst_audio_pipeline.stop()
            return

        processed_vae_batch_count = 0
        total_frames_sent_to_gstreamer = 0
        expected_total_video_frames = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            while processed_vae_batch_count < video_len_in_batches:
                try:
                    batch_tuple = res_frame_q.get(block=True, timeout=10.0)
                except queue.Empty:
                    logging.warning(f"{run_id_logging_prefix} Timeout on res_frame_q. Processed {processed_vae_batch_count}/{video_len_in_batches}.")
                    if processed_vae_batch_count > 0 : break
                    else: continue
                
                if batch_tuple is None: 
                    logging.info(f"{run_id_logging_prefix} Got None sentinel. Ending process_frames.")
                    break
                
                if not isinstance(batch_tuple, tuple) or len(batch_tuple)!=2: 
                    logging.warning(f"{run_id_logging_prefix} Invalid data from res_frame_q. Skip."); res_frame_q.task_done(); continue
                
                vae_output_frames, audio_pcm_chunk_for_gstreamer = batch_tuple

                if not vae_output_frames or not isinstance(vae_output_frames, list):
                    logging.warning(f"{run_id_logging_prefix} Invalid VAE frames. Skip batch."); res_frame_q.task_done(); processed_vae_batch_count+=1; continue
                
                num_frames_in_this_vae_batch = len(vae_output_frames)
                expected_total_video_frames += num_frames_in_this_vae_batch
                
                args_for_map = [
                    (i, self.idx, vae_output_frames[i], gst_video_pipeline.width, gst_video_pipeline.height)
                    for i in range(num_frames_in_this_vae_batch)
                ]
                
                final_blended_frames_for_gstreamer = [None] * num_frames_in_this_vae_batch
                try:
                    for i_map, blended_frame_map in executor.map(self.process_single_frame_parallel, args_for_map): # Call as method
                        if blended_frame_map is not None: final_blended_frames_for_gstreamer[i_map] = blended_frame_map
                        else: logging.warning(f"{run_id_logging_prefix} Blending returned None for frame {i_map} of VAE batch {processed_vae_batch_count}.")
                except Exception as e_map_exc: 
                    logging.error(f"{run_id_logging_prefix} Error in parallel blending batch {processed_vae_batch_count}: {e_map_exc}"); traceback.print_exc()
                    res_frame_q.task_done(); processed_vae_batch_count+=1; self.idx = (self.idx + num_frames_in_this_vae_batch)%len(self.coord_list_cycle); continue
                
                valid_blended_frames = [f for f in final_blended_frames_for_gstreamer if f is not None]
                frames_sent_iter = 0
                if valid_blended_frames:
                    for frame_send in valid_blended_frames:
                        if gst_video_pipeline.send_frame(frame_send): frames_sent_iter += 1
                        else: logging.error(f"{run_id_logging_prefix} Failed send video to GStreamer. Stop video for this segment."); break
                    total_frames_sent_to_gstreamer += frames_sent_iter
                    if frames_sent_iter > 0 and audio_pcm_chunk_for_gstreamer is not None and audio_pcm_chunk_for_gstreamer.size > 0:
                        if not gst_audio_pipeline.send_audio(audio_pcm_chunk_for_gstreamer): logging.error(f"{run_id_logging_prefix} Failed send audio to GStreamer.")
                    elif frames_sent_iter == 0 and audio_pcm_chunk_for_gstreamer is not None and audio_pcm_chunk_for_gstreamer.size > 0: 
                        logging.warning(f"{run_id_logging_prefix} No video sent, audio chunk skipped.")
                else: 
                    logging.warning(f"{run_id_logging_prefix} No valid blended frames for batch {processed_vae_batch_count}.")
                    if audio_pcm_chunk_for_gstreamer is not None and audio_pcm_chunk_for_gstreamer.size > 0:
                         logging.warning(f"{run_id_logging_prefix} Audio chunk for this batch also skipped as no video was sent.")
                
                self.idx = (self.idx + num_frames_in_this_vae_batch) % len(self.coord_list_cycle)
                res_frame_q.task_done(); processed_vae_batch_count += 1
        logging.info(f"--- {run_id_logging_prefix} Avatar.process_frames finished. Processed {processed_vae_batch_count} VAE batches. Sent {total_frames_sent_to_gstreamer} frames to GStreamer. ---")

    # This method MUST be part of the Avatar class and correctly indented
    def process_single_frame_parallel(self, args_tuple):
        original_batch_idx, start_idx_in_cycle, vae_frame, gst_vid_width, gst_vid_height = args_tuple
        
        # Ensure cyclic lists are valid and populated before accessing
        if not all(hasattr(self, attr) and isinstance(getattr(self, attr), list) and getattr(self, attr) 
                   for attr in ['coord_list_cycle', 'frame_list_cycle', 'mask_list_cycle', 'mask_coords_list_cycle']):
            logging.error("process_single_frame_parallel: Avatar cyclic data lists are not properly initialized or are empty.")
            return original_batch_idx, None
            
        cycle_len = len(self.coord_list_cycle) # Assuming all cyclic lists are of the same length after init
        if cycle_len == 0:
            logging.error("process_single_frame_parallel: Avatar cyclic data is empty (length 0).")
            return original_batch_idx, None
            
        current_cycle_idx = (start_idx_in_cycle + original_batch_idx) % cycle_len

        try:
            bbox = self.coord_list_cycle[current_cycle_idx]
            ori_frame_ref = self.frame_list_cycle[current_cycle_idx]
            mask_array_ref = self.mask_list_cycle[current_cycle_idx]
            # mc_box = self.mask_coords_list_cycle[current_cycle_idx] # mask_crop_box from _prepare_material_core

            if any(x is None for x in [bbox, ori_frame_ref, mask_array_ref]): # Removed mc_box from here as it's not directly used in this simplified blend
                logging.warning(f"BlendWorker: Missing essential data for cycle_idx {current_cycle_idx}, batch frame {original_batch_idx}. Skipping.")
                return original_batch_idx, None

            bg_frame = ori_frame_ref.copy()
            face_gen = vae_frame.astype(np.uint8)
            x,y,x1,y1 = bbox
            face_w_bg, face_h_bg = int(x1-x), int(y1-y) # Ensure integer dimensions

            if face_w_bg <=0 or face_h_bg <=0:
                logging.warning(f"BlendWorker: Invalid bbox dimensions {bbox}. Skip frame {original_batch_idx}.")
                return original_batch_idx, None
            
            resized_face = cv2.resize(face_gen, (face_w_bg, face_h_bg), interpolation=cv2.INTER_LINEAR)
            
            mask = mask_array_ref.copy()
            if len(mask.shape)==3: mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # Ensure mask matches the size of the face region it's supposed to mask
            if mask.shape[0]!=face_h_bg or mask.shape[1]!=face_w_bg:
                mask = cv2.resize(mask, (face_w_bg, face_h_bg), interpolation=cv2.INTER_NEAREST)
            
            alpha_m = (mask / 255.0).astype(np.float32)
            # Ensure alpha_m is broadcastable for 3-channel images
            if alpha_m.ndim == 2: alpha_m = alpha_m[..., np.newaxis] 
            if alpha_m.shape[-1] == 1 and resized_face.shape[-1] == 3: alpha_m = np.repeat(alpha_m, 3, axis=2)


            # Ensure ROI slice is valid
            if y1 > bg_frame.shape[0] or x1 > bg_frame.shape[1] or y < 0 or x < 0:
                logging.warning(f"BlendWorker: Bbox {bbox} out of bounds for background frame shape {bg_frame.shape}. Skipping.")
                return original_batch_idx, None

            roi_bg = bg_frame[y:y1, x:x1] # Define roi_bg

            if roi_bg.shape != resized_face.shape or \
               (alpha_m.ndim == 3 and roi_bg.shape != alpha_m.shape) or \
               (alpha_m.ndim == 2 and roi_bg.shape[:2] != alpha_m.shape): # Check for 2D alpha as well
                logging.warning(f"Blend Shape Mismatch: ROI({roi_bg.shape}), Face({resized_face.shape}), Mask({alpha_m.shape}). Attempting simple paste.")
                try:
                    bg_frame[y:y1, x:x1] = resized_face
                except Exception as e_paste:
                    logging.error(f"BlendWorker: Error during simple paste fallback: {e_paste}. Skipping frame."); return original_batch_idx, None
            else:
                # Ensure alpha_m is correctly shaped for broadcasting or direct multiplication
                if alpha_m.shape != roi_bg.shape and alpha_m.ndim == 3 and alpha_m.shape[-1] == 1: # (H,W,1) needs to broadcast with (H,W,3)
                    alpha_m_broadcast = alpha_m 
                elif alpha_m.shape == roi_bg.shape : # (H,W,3)
                     alpha_m_broadcast = alpha_m
                else: # Fallback if shapes are still problematic after checks
                    logging.warning(f"Blend Shape Mismatch (alpha): ROI({roi_bg.shape}), Alpha({alpha_m.shape}). Simple paste.")
                    bg_frame[y:y1, x:x1] = resized_face
                    alpha_m_broadcast = None # Indicate that blending wasn't done with alpha

                if alpha_m_broadcast is not None:
                    blended_roi = np.uint8(resized_face * alpha_m_broadcast + roi_bg * (1.0 - alpha_m_broadcast))
                    bg_frame[y:y1, x:x1] = blended_roi
            
            final_out = cv2.resize(bg_frame, (gst_vid_width, gst_vid_height), interpolation=cv2.INTER_LINEAR)
            return original_batch_idx, final_out
        except Exception as e:
            logging.error(f"Error in BlendWorker (original_idx {original_batch_idx}, cycle_idx {current_cycle_idx}): {e}")
            logging.error(traceback.format_exc())
            return original_batch_idx, None

# --- Pipe Processing and Listener Functions ---
def process_pipe_stream(opus_data_segment, avatar_instance_ref, target_fps_ref):
    """
    Processes a single Opus data segment: saves to temp file, then calls avatar inference.
    This function is called by the inference_worker.
    """
    global inference_lock # Use the global lock
    tmp_audio_file_path = None

    if not inference_lock.acquire(blocking=False):
        logging.warning(f"Inference already in progress (lock held). Skipping received audio segment of {len(opus_data_segment)} bytes.")
        return # Skip if another inference is already running

    try:
        logging.info(f"--- Inference Lock Acquired by process_pipe_stream for segment {len(opus_data_segment)} bytes ---")
        
        # Wrap raw Opus bytes in BytesIO for FFmpegAudioReader if it expects a file-like object for piped data
        # Or save to a temp file if FFmpegAudioReader or audio_processor strictly need a path
        # Current FFmpegAudioReader is modified to accept BytesIO if not a path.

        # For MuseTalk's audio_processor.audio2feat which likely needs a path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".opus", mode='wb') as temp_opus_file_for_features:
            temp_opus_file_for_features.write(opus_data_segment)
            tmp_audio_file_path = temp_opus_file_for_features.name
        
        logging.info(f"Opus segment data ({len(opus_data_segment)} bytes) written to temporary file: {tmp_audio_file_path} for feature extraction.")

        if tmp_audio_file_path:
            # Avatar.inference will handle reading this temp file (which contains Opus data)
            # and then internally using FFmpegAudioReader to get PCM.
            avatar_instance_ref.inference(
                audio_path_or_bytesio=tmp_audio_file_path, # Pass the path to the temp Opus file
                out_vid_name_unused="realtime_gstreamer_output", # Name not really used for GStreamer output
                target_fps=target_fps_ref,
                skip_save_images_unused=True
            )
        else: # Should not happen if tempfile creation worked
            logging.error("Temporary Opus audio file path was not obtained. Skipping inference.")

    except Exception as e:
        logging.error(f"Error during inference processing for Opus segment (length {len(opus_data_segment)}):")
        logging.error(traceback.format_exc())
    finally:
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            try:
                os.unlink(tmp_audio_file_path)
                logging.info(f"Temporary Opus file {tmp_audio_file_path} deleted.")
            except Exception as e_del:
                logging.error(f"Error deleting temporary Opus file {tmp_audio_file_path}: {e_del}")
        
        inference_lock.release()
        logging.info(f"--- Inference Lock Released by process_pipe_stream (segment {len(opus_data_segment)} bytes) ---")


def main_pipe_listener(pipe_path, avatar_instance_unused, fps_unused, opus_input_q_ref):
    """
    Continuously listens to the named pipe for audio data and puts it into opus_input_q_ref.
    It will poll less aggressively when an inference task is already in progress.
    """
    logging.info(f"🚀 Starting pipe listener thread...")
    logging.info(f"👂 Listening for audio stream on pipe: {pipe_path}")

    while True:
        # VVVVVVVVVVVVVVVVVVVV NEW LOGIC VVVVVVVVVVVVVVVVVVVV
        # If an inference is busy, sleep for a longer, non-intensive interval
        # before trying to connect. This reduces CPU/GIL contention.
        if inference_lock.locked():
            time.sleep(0.5) # Sleep for 500ms and check the lock again
            continue # Go to the start of the while loop to re-check the lock
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # If the lock is NOT held, we proceed with aggressive polling to be responsive.
        pipe_handle = None
        connected_this_cycle = False
        try:
            opus_data_for_current_message = b''
            if sys.platform == "win32":
                pipe_name = r'\\.\pipe\\' + os.path.basename(pipe_path)
                # We can reduce attempts here since we check the lock first
                max_connect_attempts = 5
                connect_attempt_delay = 0.1 # Shorter delay between responsive polls

                for attempt in range(max_connect_attempts):
                    try:
                        # Since we only get here when inference isn't running, we can be more verbose
                        logging.debug(f"Attempting to connect to Windows pipe: {pipe_name} (Attempt {attempt + 1}/{max_connect_attempts})")
                        pipe_handle = win32file.CreateFile(
                            pipe_name,
                            win32file.GENERIC_READ,
                            win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE,
                            None, win32file.OPEN_EXISTING, 0, None
                        )
                        logging.info(f"Successfully connected to Windows pipe: {pipe_name}")
                        connected_this_cycle = True
                        break
                    except pywintypes.error as e:
                        if e.winerror == 2: # ERROR_FILE_NOT_FOUND
                            if attempt < max_connect_attempts - 1:
                                time.sleep(connect_attempt_delay)
                            else: # After all attempts fail, the outer loop will trigger a longer sleep.
                                logging.debug(f"Pipe not found after {max_connect_attempts} attempts. Will wait.")
                        else:
                            logging.warning(f"Windows API error on connect attempt: {e}")
                            time.sleep(1) # Wait a bit on other errors
                
                # If we are connected, read the data from the pipe
                if connected_this_cycle and pipe_handle:
                    logging.info(f"Windows pipe connected. Reading data...")
                    while True:
                        try:
                            hr, chunk = win32file.ReadFile(pipe_handle, 131072)
                            if len(chunk) > 0:
                                opus_data_for_current_message += chunk
                            if hr == 0 and len(chunk) == 0: # Graceful close
                                break
                        except pywintypes.error as e_read:
                            if e_read.winerror == 109: # Broken pipe is normal EOF here
                                logging.info(f"Broken pipe on {pipe_name} - writer closed connection.")
                            else: logging.warning(f"Error reading from pipe: {e_read}")
                            break
                    # The finally block will close the handle
            # ... (your existing Linux/macOS logic remains the same) ...
            
            # --- Data Queuing Logic ---
            if opus_data_for_current_message:
                logging.info(f"Pipe listener successfully read {len(opus_data_for_current_message)} bytes. Queuing for processing.")
                try:
                    # This put will block until the inference_worker's queue has space, which is fine.
                    opus_input_q_ref.put(opus_data_for_current_message) 
                except Exception as e_queue:
                    logging.error(f"Failed to put data into queue: {e_queue}")

        except Exception as e_outer:
            logging.error(f"Pipe listener loop error: {e_outer}")
            time.sleep(2) # Sleep on major errors before retrying
        finally:
            if pipe_handle:
                try: win32file.CloseHandle(pipe_handle)
                except Exception: pass

# <<<<<<<<<<<<<<<< NEW: Worker Function Definition <<<<<<<<<<<<<<<<
def inference_worker(avatar_instance_ref, target_fps_ref, opus_q_ref):
    """
    Worker thread that takes Opus data from a queue and processes it for inference.
    """
    logging.info("🚀 Inference worker thread started. Waiting for Opus data...")
    while True:
        try:
            # Get data from the shared queue, block with a timeout
            # Timeout allows this thread to periodically check for other conditions or be interrupted
            opus_data_segment = opus_q_ref.get(block=True, timeout=1.0)

            if opus_data_segment is None:  # Sentinel value for shutdown
                logging.info("Inference worker received None (shutdown sentinel). Exiting worker thread.")
                opus_q_ref.task_done() # Acknowledge the sentinel
                break # Exit the loop, thread will terminate

            logging.info(f"Inference worker picked up audio segment of {len(opus_data_segment)} bytes for processing.")
            
            # Call process_pipe_stream which contains the inference_lock and calls avatar.inference()
            process_pipe_stream(opus_data_segment, avatar_instance_ref, target_fps_ref)
            
            opus_q_ref.task_done() # Signal that this item from the queue has been processed

        except queue.Empty:
            # Timeout occurred on opus_q_ref.get(), queue is empty. Loop again to wait.
            # This is normal if no data is coming in.
            logging.debug("Inference worker: Opus input queue empty after timeout, continuing to wait...")
            continue
        except Exception as e:
            logging.error(f"Error in inference_worker's main loop: {e}")
            logging.error(traceback.format_exc())
            # Depending on the error, you might want to try to acknowledge the queue item if one was fetched
            # For now, log and continue. If errors are persistent, the worker might get stuck.
            # A robust system might re-queue a failed item or have a dead-letter queue.
            # Since task_done is after process_pipe_stream, if that errors, task_done isn't called.
            # Add a task_done in a finally block if an item was successfully dequeued.
            # However, if process_pipe_stream itself handles its errors well, this might be okay.
            # For simplicity now, just logging.
            time.sleep(1) # Brief pause after an error before retrying a get()

# ==================================================================================
# VVVVVVVVVVVVVVVVVV  MAIN EXECUTION BLOCK VVVVVVVVVVVVVVVVVVVVVV
# ==================================================================================
if __name__ == "__main__":
    logging.info("🎬 Starting Realtime Stream Sync (Pipe Reader & Inference Engine)...")

    # --- Validate Essential Configuration ---
    if not STREAM_PIPE_PATH:
        logging.critical("❌ CRITICAL ERROR: STREAM_PIPE_PATH environment variable not set.")
        sys.exit(1)
    if not AVATAR_ID_TO_USE:
        logging.critical("❌ CRITICAL ERROR: AVATAR_ID_TO_USE environment variable not set.")
        sys.exit(1)
    if not os.path.exists(AVATAR_CONFIG_PATH):
        logging.critical(f"❌ CRITICAL ERROR: AVATAR_CONFIG_PATH '{AVATAR_CONFIG_PATH}' not found.")
        sys.exit(1)
    
    logging.info(f"  Pipe Path       : {STREAM_PIPE_PATH}")
    logging.info(f"  Avatar Config   : {AVATAR_CONFIG_PATH}")
    logging.info(f"  Avatar ID       : {AVATAR_ID_TO_USE}")
    logging.info(f"  Target Output FPS: {TARGET_FPS}")

    main_avatar_instance = None # Initialize
    try:
        inference_config_all = OmegaConf.load(AVATAR_CONFIG_PATH)
        if AVATAR_ID_TO_USE not in inference_config_all:
            logging.critical(f"❌ CRITICAL ERROR: Avatar ID '{AVATAR_ID_TO_USE}' not found in config file '{AVATAR_CONFIG_PATH}'.")
            logging.info(f"Available Avatar IDs in config: {list(inference_config_all.keys())}")
            sys.exit(1)
        
        avatar_specific_config = inference_config_all[AVATAR_ID_TO_USE]
        logging.info(f"Loaded configuration for Avatar ID: {AVATAR_ID_TO_USE}")

        # Extract parameters for Avatar class instantiation
        video_path_from_config = avatar_specific_config.get("video_path")
        bbox_shift_from_config = avatar_specific_config.get("bbox_shift", [0,0,0,0]) # Ensure default is a list
        preparation_mode = avatar_specific_config.get("preparation", False)
        batch_size_from_config = avatar_specific_config.get("batch_size", 4) # Default if not in config

        if not video_path_from_config:
            logging.critical(f"❌ CRITICAL ERROR: 'video_path' not defined for avatar '{AVATAR_ID_TO_USE}' in config.")
            sys.exit(1)

        logging.info("Instantiating Avatar object...")
        # VVVVVV ADD THESE DIAGNOSTIC PRINT STATEMENTS VVVVVV
        try:
            print(f"DEBUG: Avatar class: {Avatar}")
            print(f"DEBUG: Avatar.__init__ method: {Avatar.__init__}")
            print(f"DEBUG: Expected __init__ arg count (incl. self): {Avatar.__init__.__code__.co_argcount}")
            print(f"DEBUG: Expected __init__ arg names: {Avatar.__init__.__code__.co_varnames}")
        except Exception as e_diag:
            print(f"DEBUG: Error during diagnostics: {e_diag}")
        # ^^^^^^ END OF DIAGNOSTIC PRINT STATEMENTS ^^^^^^
        main_avatar_instance = Avatar(
            avatar_id=AVATAR_ID_TO_USE,
            video_path=video_path_from_config,
            bbox_shift=bbox_shift_from_config,
            batch_size=batch_size_from_config,
            preparation=preparation_mode
        )
        logging.info("✅ Avatar object instantiated successfully.")

    except SystemExit: # Catch sys.exit calls from Avatar init or config loading
        logging.critical("❌ Exiting due to critical error during Avatar initialization or configuration loading.")
        sys.exit(1) # Ensure exit
    except Exception as e_avatar_setup:
        logging.critical(f"❌ CRITICAL ERROR during Avatar setup for '{AVATAR_ID_TO_USE}':")
        logging.critical(traceback.format_exc())
        sys.exit(1)

    if main_avatar_instance is None: # Should have been caught by prior exceptions
        logging.critical("❌ CRITICAL ERROR: Avatar instance is None after setup attempt. Exiting.")
        sys.exit(1)

    # --- Start the Worker Threads ---
    logging.info("🚀 Initializing and starting worker threads...")

    # Inference Worker Thread (processes data from opus_input_queue)
    inference_processing_thread = threading.Thread(
        target=inference_worker, 
        args=(main_avatar_instance, TARGET_FPS, opus_input_queue), # Pass the global queue
        daemon=True, # Daemon threads will exit when the main program exits
        name="InferenceWorkerThread"
    )
    inference_processing_thread.start()
    logging.info("✅ Inference worker thread started.")

    # Pipe Listener Thread (reads from pipe and puts into opus_input_queue)
    pipe_listening_thread = threading.Thread(
        target=main_pipe_listener, 
        args=(STREAM_PIPE_PATH, main_avatar_instance, TARGET_FPS, opus_input_queue), # Pass the global queue
        daemon=True,
        name="PipeListenerThread"
    )
    pipe_listening_thread.start()
    logging.info("✅ Pipe listener thread started.")

    # --- Keep Main Thread Alive & Handle Graceful Shutdown ---
    logging.info("Application is running. Press Ctrl+C to initiate a graceful shutdown.")
    try:
        while True: 
            # Check if worker threads are alive; main thread primarily waits for KeyboardInterrupt
            if not pipe_listening_thread.is_alive():
                logging.warning("⚠️ Pipe listener thread has unexpectedly stopped. Inference may no longer receive data.")
                # Optionally, try to signal inference worker to stop if pipe listener dies.
                if inference_processing_thread.is_alive():
                     logging.info("Signaling inference worker to stop due to pipe listener failure.")
                     opus_input_queue.put(None) # Attempt to stop worker
                break # Exit main loop if pipe listener dies
            
            if not inference_processing_thread.is_alive() and pipe_listening_thread.is_alive():
                logging.warning("⚠️ Inference processing thread has unexpectedly stopped. Pipe listener might fill the queue.")
                # Depending on recovery strategy, you might want to stop the pipe_listener or restart inference_worker.
                # For now, main loop will continue if pipe_listener is alive.
                pass # Main loop continues, relying on Ctrl+C or pipe_listener dying to exit.

            time.sleep(1.0) # Keep main thread responsive to interrupt

    except KeyboardInterrupt:
        logging.info("\n🛑 KeyboardInterrupt received in main thread. Initiating graceful shutdown...")
    except Exception as e_main_loop:
        logging.error(f"Unexpected error in main execution loop: {e_main_loop}")
        logging.error(traceback.format_exc())
    finally:
        logging.info("--- Shutdown sequence started ---")

        # 1. Signal the inference_worker to stop by putting None (sentinel) in its input queue.
        #    This allows it to finish its current task if any.
        if inference_processing_thread.is_alive():
            logging.info("Signaling inference worker thread to stop by placing None in its queue...")
            try:
                opus_input_queue.put(None, timeout=2.0) # Timeout for putting sentinel
            except queue.Full:
                logging.warning("Opus input queue was full when trying to send shutdown sentinel to inference worker. It might not shut down cleanly if stuck on queue.put().")
            except Exception as e_q_shutdown:
                logging.error(f"Error putting shutdown sentinel to inference queue: {e_q_shutdown}")

        # 2. Wait for the pipe listener thread to finish.
        #    It's a daemon, so it would die with main, but joining is cleaner.
        #    It might be blocked on a pipe read.
        if pipe_listening_thread.is_alive():
            logging.info("Waiting for pipe listener thread to join (max 5s)...")
            pipe_listening_thread.join(timeout=5.0)
            if pipe_listening_thread.is_alive():
                logging.warning("⚠️ Pipe listener thread did not join in time. It might be blocked on a pipe operation.")
            else:
                logging.info("✅ Pipe listener thread joined or was already finished.")
        
        # 3. Wait for the inference worker thread to finish.
        if inference_processing_thread.is_alive():
            logging.info("Waiting for inference worker thread to join (max 30s, allowing current task to finish)...")
            inference_processing_thread.join(timeout=30.0) 
            if inference_processing_thread.is_alive():
                logging.warning("⚠️ Inference worker thread did not join in time. It might be stuck on a long inference task or did not receive/process the sentinel.")
            else:
                logging.info("✅ Inference worker thread joined or was already finished.")
        
        # Ensure GStreamer processes are cleaned up (though Avatar.inference's finally should do it for active ones)
        # This is a fallback if GStreamer instances were created but not cleaned by Avatar.inference ending.
        if main_avatar_instance:
            # You might need to add a method to Avatar to access its GStreamer instances if they are per-inference.
            # For now, this is a placeholder if GStreamer objects were global or long-lived.
            logging.info("Ensuring any global GStreamer resources are released (if applicable).")
            # if hasattr(main_avatar_instance, 'gst_video_pipeline') and main_avatar_instance.gst_video_pipeline:
            # main_avatar_instance.gst_video_pipeline.stop()
            # if hasattr(main_avatar_instance, 'gst_audio_pipeline') and main_avatar_instance.gst_audio_pipeline:
            # main_avatar_instance.gst_audio_pipeline.stop()

        logging.info("✅ Application shutdown sequence complete.")