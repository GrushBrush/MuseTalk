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
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
# from musetalk.utils.blending import get_image_blending # Logic now in worker
from musetalk.utils.utils import load_all_model
import shutil
import time
from PIL import Image
import tempfile

# --- PyTorch Device Setup ---
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

print("--- PyTorch Device Information ---")
if cuda_available:
    print("✅ CUDA (GPU) detected by PyTorch.")
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {gpu_name}")
    except Exception as e:
        print(f"⚠️ Could not retrieve GPU name: {e}")
    print(f"✅ Selected device: {device}")
else:
    print("❌ CUDA (GPU) not available or not detected by PyTorch.")
    print(f"✅ Selected device: {device}")
print("-------------------------------")

# --- Load Models ---
print("Loading models...")
# These might be wrapper objects, not nn.Module directly
audio_processor, vae, unet, pe = load_all_model()
print("Models loaded.")

# --- Set Precision (and potentially move sub-modules if needed) ---
# Reverting to logic based on original code - applying .half() to sub-attributes
# Assuming internal methods handle device placement or relevant tensors are moved.
print("Setting model precision to half (FP16) on sub-modules...")
try:
    # Handle PE (Positional Encoding) - original code applied .half() directly
    # Let's try moving it first if possible, then apply half
    if hasattr(pe, 'to'):
        pe = pe.to(device)
    if hasattr(pe, 'half'):
        pe = pe.half()
    else:
        print("Warning: 'pe' object doesn't have .half() method.")

    # Handle VAE - original code accessed vae.vae
    if hasattr(vae, 'vae'):
        if hasattr(vae.vae, 'to'):
            vae.vae = vae.vae.to(device) # Move sub-module
        if hasattr(vae.vae, 'half'):
            vae.vae = vae.vae.half() # Set precision on sub-module
        else:
            print("Warning: 'vae.vae' object doesn't have .half() method.")
    else:
        print("Warning: Cannot access 'vae.vae'. VAE structure might differ.")
        # Fallback: Try applying to vae itself if possible (less likely based on error)
        # if hasattr(vae, 'to'): vae = vae.to(device)
        # if hasattr(vae, 'half'): vae = vae.half()

    # Handle UNet - original code accessed unet.model
    if hasattr(unet, 'model'):
        if hasattr(unet.model, 'to'):
            unet.model = unet.model.to(device) # Move sub-module
        if hasattr(unet.model, 'half'):
            unet.model = unet.model.half() # Set precision on sub-module
        else:
            print("Warning: 'unet.model' object doesn't have .half() method.")
    else:
        print("Warning: Cannot access 'unet.model'. UNet structure might differ.")
        # Fallback: Try applying to unet itself if possible
        # if hasattr(unet, 'to'): unet = unet.to(device)
        # if hasattr(unet, 'half'): unet = unet.half()

    print("Precision set (check warnings above).")
except Exception as e_prec:
     print(f"Warning: Error setting model precision/device on sub-modules: {e_prec}")
     print("Continuing, but models might be on wrong device or have wrong precision.")

# Timesteps tensor on the correct device
timesteps = torch.tensor([0], device=device)


# --- GStreamer Receiver Function (Optional) ---
def start_gstreamer_receiver():
    receiver_cmd = (
        "gst-launch-1.0 -v udpsrc port=5000 caps=\"application/x-matroska\" ! "
        "matroskademux name=d d.video_0 ! queue ! decodebin ! autovideosink "
        "d.audio_0 ! queue ! decodebin ! autoaudiosink"
    )
    print("Starting GStreamer receiver (run separately if needed)...")
    return subprocess.Popen(receiver_cmd, shell=True)

# --- FFmpeg Audio Reader Class ---
class FFmpegAudioReader:
    """ Use FFmpeg to read an entire audio file and convert it to PCM s16le, 48kHz, Stereo """
    def __init__(self, audio_file):
        self.audio_file = audio_file
        try:
            print(f"Probing audio file: {audio_file}")
            probe = ffmpeg.probe(audio_file)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            if audio_stream is None:
                raise RuntimeError(f"No audio stream found in {audio_file}")
            self.sample_rate = int(audio_stream.get('sample_rate', '48000'))
            self.channels = int(audio_stream.get('channels', '2'))
            print(f"Detected Sample Rate: {self.sample_rate}, Channels: {self.channels}")
        except ffmpeg.Error as e:
            print(f"❌ Error probing audio file {audio_file}: {e.stderr.decode(errors='ignore') if e.stderr else 'Unknown FFmpeg error'}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error during FFmpeg probe: {e}")
            raise


    def read_full_audio(self):
        """ Use FFmpeg to read the full audio file, resample/reformat as needed """
        print("Reading and converting audio with FFmpeg...")
        target_sr = 48000
        target_ac = 2
        target_format = "s16le"

        try:
            process = subprocess.Popen(
                ["ffmpeg", "-i", self.audio_file,
                 "-f", target_format,
                 "-ac", str(target_ac),
                 "-ar", str(target_sr),
                 "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            raw_data, stderr_data = process.communicate()
            retcode = process.poll()
            stderr_str = stderr_data.decode(errors='ignore')
            if retcode != 0:
                print(f"❌ FFmpeg process failed with code {retcode}")
                print(f"FFmpeg stderr:\n{stderr_str}")
                return None

            if not raw_data:
                print("❌ Failed to read audio file (FFmpeg produced no data)!")
                print(f"FFmpeg stderr:\n{stderr_str}")
                return None

            audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, target_ac)
            print(f"✅ Read and converted full audio, {len(audio_data)} samples at {target_sr} Hz, {target_ac} channels")
            return audio_data

        except FileNotFoundError:
             print("❌ Error: ffmpeg command not found. Is FFmpeg installed and in your system's PATH?")
             return None
        except Exception as e:
             print(f"❌ Unexpected error during FFmpeg audio read: {e}")
             try:
                  if stderr_data: print(f"FFmpeg stderr:\n{stderr_data.decode(errors='ignore')}")
             except: pass
             return None


# --- Audio Splitting Function ---
def split_audio(audio_data, num_chunks):
    """ Split audio data (NumPy array) into num_chunks parts """
    if audio_data is None or num_chunks <= 0:
        print("❌ Cannot split invalid audio data or chunk count.")
        return []
    if num_chunks == 1:
        return [audio_data]

    total_samples = len(audio_data)
    chunks = np.array_split(audio_data, num_chunks, axis=0) # Use numpy's array_split

    # Validation print
    total_samples_in_chunks = sum(len(c) for c in chunks)
    print(f"✅ Audio split into {len(chunks)} chunks using np.array_split. Total samples: {total_samples_in_chunks} (Expected: {total_samples})")
    if total_samples_in_chunks != total_samples:
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
         print("Error: Mismatch in total samples after splitting audio!")
         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return chunks


# --- GStreamer Pipeline Classes ---
class GStreamerPipeline:
    # (Keep the GStreamerPipeline class code from the previous correct response - no changes needed here)
    def __init__(self, width=720, height=1280, fps=25, host="127.0.0.1", port=5000):
        self.width = width
        self.height = height
        self.fps = fps
        self.host = host
        self.port = port
        self.process = None # Initialize process to None

        pipeline_str_fdsrc = (
            f"fdsrc fd=0 ! "
    f"videoparse format=bgr width={self.width} height={self.height} framerate={self.fps}/1 ! "
    "queue ! videoconvert ! video/x-raw,format=NV12 ! "
    "queue ! cudaupload ! "
    # --- Using your old working encoder parameters ---
    "nvh265enc bitrate=8000000 bframes=0 preset=default gop-size=25 ! " 
    # --- End using old parameters ---
    "h265parse ! " # Keep h265parse, it's generally good practice
    "rtph265pay config-interval=1 ! "
    # --- Using your old working sink parameters ---
    f"udpsink host={self.host} port={self.port} sync=true"
        )

        print("Starting GStreamer video pipeline...")
        try:
            self.process = subprocess.Popen(
                f"gst-launch-1.0 -v {pipeline_str_fdsrc}", # Add -v for verbose GStreamer output
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # Capture stdout
                stderr=subprocess.PIPE, # Capture stderr
                shell=True,
                bufsize=0 # Try unbuffered stdin
            )
            print(f"✅ GStreamer video pipeline started (Res: {self.width}x{self.height})")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_VID_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_VID_ERR"), daemon=True).start()
        except Exception as e:
            print(f"❌ Failed to start GStreamer video pipeline: {e}")
            self.process = None

    def _log_stream(self, stream, prefix):
        try:
            for line in iter(stream.readline, b''):
                 print(f"[{prefix}]: {line.decode(errors='ignore').strip()}")
        except Exception as e:
            print(f"Error in GStreamer log thread ({prefix}): {e}")
        finally:
            stream.close()

    def send_frame(self, frame):
        if not self.process or self.process.stdin.closed:
             return False
        try:
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                # This resize should ideally not happen if process_frames is correct
                print(f"⚠️ Resizing frame in send_frame from {frame.shape[1]}x{frame.shape[0]} to {self.width}x{self.height}")
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            return True
        except BrokenPipeError:
             print("❌ Error pushing video frame: Broken pipe.")
             self.process = None
             return False
        except Exception as e:
             print(f"❌ Error pushing video frame: {e}")
             return False

    def stop(self):
        print("Stopping GStreamer video pipeline...")
        if self.process:
            proc = self.process # Avoid race condition if self.process is set to None elsewhere
            self.process = None # Mark as stopping
            try:
                if not proc.stdin.closed:
                    proc.stdin.close()
            except Exception as e:
                 print(f"Error closing video stdin: {e}")
            try:
                proc.wait(timeout=5)
                print("✅ GStreamer video process terminated.")
            except subprocess.TimeoutExpired:
                print("⚠️ GStreamer video process did not terminate gracefully, killing...")
                proc.kill()
                proc.wait()
                print("✅ GStreamer video process killed.")
            except Exception as e:
                 print(f"Error waiting for video process: {e}")
        else:
            print("Video pipeline process was not running or already stopped.")


class GStreamerAudio:
    # (Keep the GStreamerAudio class code from the previous correct response - no changes needed here)
    def __init__(self, host="127.0.0.1", port=5001):
        self.host = host
        self.port = port
        self.process = None # Initialize

        pipeline_str = (
    "fdsrc fd=0 do-timestamp=true ! "
    "audio/x-raw,format=S16LE,channels=2,rate=48000,layout=interleaved ! "
    # "queue name=q_in ! " # Removed
    # "audioconvert ! " # Removed
    # "audioresample ! " # Removed
    # "queue name=q_before_enc ! " # Removed
    "opusenc name=enc bitrate=64000 complexity=4 frame-size=20 ! "
    "rtpopuspay name=pay pt=97 ! "
    f"udpsink host={self.host} port={self.port} sync=false async=false"
)
        print("Starting GStreamer audio pipeline...")
        try:
            self.process = subprocess.Popen(
                f"gst-launch-1.0 -v {pipeline_str}", # Add -v for verbose GStreamer output
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # Capture stdout
                stderr=subprocess.PIPE, # Capture stderr
                shell=True,
                bufsize=0 # Try unbuffered stdin
            )
            print("✅ GStreamer audio pipeline started.")
            threading.Thread(target=self._log_stream, args=(self.process.stdout, "GST_AUD_OUT"), daemon=True).start()
            threading.Thread(target=self._log_stream, args=(self.process.stderr, "GST_AUD_ERR"), daemon=True).start()

        except Exception as e:
            print(f"❌ Failed to start GStreamer audio pipeline: {e}")
            self.process = None

    def _log_stream(self, stream, prefix):
        try:
            for line in iter(stream.readline, b''):
                 print(f"[{prefix}]: {line.decode(errors='ignore').strip()}")
        except Exception as e:
            print(f"Error in GStreamer log thread ({prefix}): {e}")
        finally:
            stream.close()

    def send_audio(self, audio_data):
        if not self.process or self.process.stdin.closed:
             return False
        try:
            if audio_data.dtype != np.int16:
                print(f"Warning: Audio data is not int16 ({audio_data.dtype}), attempting conversion.")
                audio_data = audio_data.astype(np.int16)
            self.process.stdin.write(audio_data.tobytes())
            self.process.stdin.flush()
            return True
        except BrokenPipeError:
             print("❌ Error pushing audio chunk: Broken pipe.")
             self.process = None
             return False
        except Exception as e:
             print(f"❌ Error pushing audio chunk: {e}")
             return False

    def stop(self):
        print("Stopping GStreamer audio pipeline...")
        if self.process:
            proc = self.process
            self.process = None # Mark as stopping
            try:
                if not proc.stdin.closed:
                    proc.stdin.close()
            except Exception as e:
                 print(f"Error closing audio stdin: {e}")
            try:
                proc.wait(timeout=5)
                print("✅ GStreamer audio process terminated.")
            except subprocess.TimeoutExpired:
                print("⚠️ GStreamer audio process did not terminate gracefully, killing...")
                proc.kill()
                proc.wait()
                print("✅ GStreamer audio process killed.")
            except Exception as e:
                 print(f"Error waiting for audio process: {e}")
        else:
            print("Audio pipeline process was not running or already stopped.")


# --- Misc Helper Functions ---
def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    """Extracts frames from video, ensures 8-digit naming."""
    print(f"Extracting frames from {vid_path} to {save_path}...")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {vid_path}")
        return
    count = 0
    frame_interval = 1 # Process every frame
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or read error.")
            break

        if frame_idx % frame_interval == 0:
             filename = f"{count:08d}{ext}"
             filepath = os.path.join(save_path, filename)
             try:
                 cv2.imwrite(filepath, frame)
                 count += 1
                 if count >= cut_frame:
                      print(f"Reached cut_frame limit: {cut_frame}")
                      break
             except Exception as e:
                  print(f"Error writing frame {count}: {e}")
                  break # Stop on write error

        frame_idx += 1

    cap.release()
    print(f"Finished extracting {count} frames.")


def osmakedirs(path_list):
    for path in path_list:
        try:
            os.makedirs(path, exist_ok=True) # exist_ok=True prevents error if dir exists
        except Exception as e:
             print(f"Error creating directory {path}: {e}")


# --- Parallel Frame Processing Helper Function ---
# (Keep the process_single_frame_parallel function from the previous correct response - no changes needed here)
def process_single_frame_parallel(args):
    """Worker function to process one frame in parallel."""
    # Unpack arguments
    self, i, start_idx, res_frame, gst_video_pipeline_width, gst_video_pipeline_height = args
    try:
        # Calculate index for this specific frame within the reference cycle
        current_list_len = len(self.coord_list_cycle)
        if current_list_len == 0:
            print("Error(worker): Avatar reference lists are empty!")
            return i, None # Return index and None for error

        current_idx = (start_idx + i) % current_list_len # Use start_idx + offset

        # --- Safely get data for current_idx ---
        if current_idx >= len(self.coord_list_cycle) or \
           current_idx >= len(self.frame_list_cycle) or \
           current_idx >= len(self.mask_list_cycle) or \
           current_idx >= len(self.mask_coords_list_cycle):
            print(f"Warning(worker): Index {current_idx} out of bounds (len={current_list_len}). Skipping frame {i}.")
            return i, None

        bbox = self.coord_list_cycle[current_idx]
        ori_frame_ref = self.frame_list_cycle[current_idx] # Get reference first
        mask = self.mask_list_cycle[current_idx]
        mask_crop_box = self.mask_coords_list_cycle[current_idx]

        # === Unpack bbox coordinates ===
        x, y, x1, y1 = bbox

        # --- Use .copy() instead of deepcopy ---
        if ori_frame_ref is None:
            print(f"Error(worker): ori_frame_ref is None for index {current_idx}. Skipping frame {i}.")
            return i, None
        # Ensure it's a numpy array before copying
        if not isinstance(ori_frame_ref, np.ndarray):
             print(f"Error(worker): ori_frame_ref is not NumPy array (type={type(ori_frame_ref)}) for index {current_idx}. Skipping frame {i}.")
             return i, None
        ori_frame = ori_frame_ref.copy() # Use shallow copy

        # --- First Resize ---
        res_frame_resized = None
        # Check if width (x1-x) and height (y1-y) are positive
        if x1 > x and y1 > y:
            if isinstance(res_frame, np.ndarray):
                 # Use correct dimensions (width, height) for resizing
                 res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x1 - x, y1 - y), interpolation=cv2.INTER_LINEAR)
            else:
                 print(f"Error(worker): res_frame is not a NumPy array for frame {i}. Type: {type(res_frame)}. Skipping.")
                 return i, None # Skip if input isn't as expected
        else:
            print(f"Warning(worker): Invalid bounding box dimensions (width or height non-positive): {bbox}. Skipping frame {i}.")
            return i, None

        if res_frame_resized is None:
             print(f"Warning(worker): First resize failed or skipped for frame {i}.")
             return i, None # Exit if resize didn't happen

        # --- Blending Logic ---
        combine_frame = None
        body = ori_frame # Blending modifies ori_frame (the copy)
        face = res_frame_resized # Use the resized face
        # face_box = bbox # Not needed directly
        mask_array = mask
        crop_box = mask_crop_box

        # Ensure mask_array and crop_box are valid before proceeding
        if mask_array is None:
             print(f"Error(worker): mask_array is None for index {current_idx}. Skipping frame {i}.")
             return i, None
        if crop_box is None or len(crop_box) != 4:
             print(f"Error(worker): crop_box is invalid for index {current_idx}: {crop_box}. Skipping frame {i}.")
             return i, None

        x_s, y_s, x_e, y_e = crop_box
        if y_e <= y_s or x_e <= x_s:
             print(f"Warning(worker): Invalid crop_box dimensions: {crop_box}. Skipping frame {i}.")
             return i, None
        # Ensure crop_box indices are within body (ori_frame) bounds
        h_body, w_body = body.shape[:2]
        y_s, y_e = max(0, y_s), min(h_body, y_e)
        x_s, x_e = max(0, x_s), min(w_body, x_e)
        if y_e <= y_s or x_e <= x_s: # Check again after clamping
             print(f"Warning(worker): Clamped crop_box results in zero size: {(x_s,y_s,x_e,y_e)}. Skipping frame {i}.")
             return i, None

        face_large = body[y_s:y_e, x_s:x_e].copy()

        # Calculate target slice coordinates relative to face_large (top-left is 0,0)
        y_start_paste = y - y_s
        y_end_paste = y1 - y_s
        x_start_paste = x - x_s
        x_end_paste = x1 - x_s

        # Ensure calculated slice indices are valid within face_large bounds
        h_large, w_large = face_large.shape[:2]
        y_start_paste, y_end_paste = max(0, y_start_paste), min(h_large, y_end_paste)
        x_start_paste, x_end_paste = max(0, x_start_paste), min(w_large, x_end_paste)

        if y_start_paste >= y_end_paste or x_start_paste >= x_end_paste:
             print(f"Warning(worker): Invalid paste slice (zero size) after clamping ({y_start_paste}:{y_end_paste}, {x_start_paste}:{x_end_paste}) "
                   f"for face_large {face_large.shape}. Skipping frame {i}.")
             return i, None

        slice_h = y_end_paste - y_start_paste
        slice_w = x_end_paste - x_start_paste

        # Check face shape against calculated slice shape
        if face.shape[0] != slice_h or face.shape[1] != slice_w:
             print(f"Error(worker): Resized face shape {face.shape[:2]} mismatch with target slice shape ({slice_h}, {slice_w}). Attempting final resize on face.")
             try:
                  face = cv2.resize(face, (slice_w, slice_h), interpolation=cv2.INTER_LINEAR)
                  print(f"Info(worker): Face resized again to fit slice.")
             except Exception as e_resize2:
                  print(f"Error(worker): Failed to resize face to slice shape: {e_resize2}. Skipping frame {i}.")
                  return i, None

        # Paste the face onto the copied region
        face_large[y_start_paste:y_end_paste, x_start_paste:x_end_paste] = face

        # Prepare mask for blending
        mask_image_blend = mask_array
        if len(mask_image_blend.shape) == 3:
            mask_image_blend = cv2.cvtColor(mask_image_blend, cv2.COLOR_BGR2GRAY)
        # Ensure mask has same H,W as face_large for blending
        if mask_image_blend.shape[0] != face_large.shape[0] or mask_image_blend.shape[1] != face_large.shape[1]:
             print(f"Warning(worker): Mask shape {mask_image_blend.shape} mismatch with face_large {face_large.shape[:2]}. Resizing mask.")
             mask_image_blend = cv2.resize(mask_image_blend, (face_large.shape[1], face_large.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_image_blend = (mask_image_blend / 255.0).astype(np.float32)

        # Perform Blending (Using manual NumPy blend for compatibility)
        try:
            mask_expanded = mask_image_blend[..., np.newaxis]
            if mask_expanded.shape[0] != face_large.shape[0] or mask_expanded.shape[1] != face_large.shape[1]:
                 raise ValueError(f"Shape mismatch: mask {mask_expanded.shape}, face_large {face_large.shape}")

            body_slice = body[y_s:y_e, x_s:x_e]
            if body_slice.shape != face_large.shape:
                 raise ValueError(f"Shape mismatch: body_slice {body_slice.shape}, face_large {face_large.shape}")

            blended_region = face_large.astype(np.float32) * mask_expanded + \
                             body_slice.astype(np.float32) * (1.0 - mask_expanded)
            # Assign blended result back to the original body (which is a copy of ori_frame)
            body[y_s:y_e, x_s:x_e] = blended_region.astype(body.dtype)
            combine_frame = body
        except Exception as e_blend:
            print(f"Error(worker): Blending failed for frame {i}: {e_blend}")
            print(f"Shapes - face_large: {face_large.shape}, body_slice: {body[y_s:y_e, x_s:x_e].shape}, mask_expanded: {mask_expanded.shape if 'mask_expanded' in locals() else 'NotCreated'}")
            return i, None

        # --- Second Resize ---
        if combine_frame is not None:
            final_resized = cv2.resize(combine_frame, (gst_video_pipeline_width, gst_video_pipeline_height), interpolation=cv2.INTER_LINEAR)
            return i, final_resized
        else:
            print(f"Warning(worker): combine_frame is None after blending for frame {i}.")
            return i, None

    except Exception as e_worker:
        print(f"!!!!! Error(worker): Unexpected error processing frame {i} !!!!!")
        traceback.print_exc()
        return i, None


# --- Main Avatar Class ---
@torch.no_grad()
class Avatar:
    # Define slots for potential memory optimization (optional)
    # __slots__ = [...] # List all expected instance attributes

    # ========================================================================
    # === Corrected __init__, init, prepare_material Methods =================
    # ========================================================================
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        print(f"Initializing Avatar: {avatar_id}")
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size # Store batch_size
        self.preparation = preparation
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id, "video_path": video_path, "bbox_shift": bbox_shift
        }
        self.idx = 0 # Processing index

        # Initialize data lists to None
        self.input_latent_list_cycle = None
        self.coord_list_cycle = None
        self.frame_list_cycle = None
        self.mask_coords_list_cycle = None
        self.mask_list_cycle = None

        # init handles loading or preparing data
        self.init()
        print(f"Avatar initialization complete for {avatar_id}.")

    def init(self):
        """Handles preparation or loading of existing data."""
        if self.preparation:
            if os.path.exists(self.avatar_path):
                try:
                    response = input(f"Avatar '{self.avatar_id}' exists. Re-create? (y/n): ")
                    if response.lower() == "y":
                        print(f"Removing existing avatar data: {self.avatar_path}")
                        shutil.rmtree(self.avatar_path)
                        self.prepare_material() # Prepare calls _reload at end
                    else:
                        print("Attempting to load existing prepared data...")
                        self._reload_prepared_data()
                except Exception as e_input:
                     print(f"Error during user input: {e_input}. Assuming 'n'.")
                     print("Attempting to load existing prepared data...")
                     self._reload_prepared_data()
            else:
                 print(f"Avatar path {self.avatar_path} does not exist. Preparing...")
                 self.prepare_material()
        else:
             # Check essential files exist if not preparing
             required_files = [self.coords_path, self.latents_out_path, self.mask_coords_path, self.mask_out_path, self.full_imgs_path]
             if not all(os.path.exists(p) for p in required_files):
                  print(f"Error: Not all required avatar data found in {self.avatar_path} and preparation=False.")
                  print(" Missing:", [p for p in required_files if not os.path.exists(p)])
                  print(" Run with preparation=True first.")
                  sys.exit(1)
             else:
                  print("Preparation=False. Loading existing prepared data...")
                  self._reload_prepared_data()
                  # Optional consistency check
                  try:
                       if os.path.exists(self.avatar_info_path):
                            with open(self.avatar_info_path, "r") as f:
                                 avatar_info_disk = json.load(f)
                            if avatar_info_disk.get('bbox_shift') != self.avatar_info['bbox_shift']:
                                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                 print("Error: bbox_shift has changed since data was prepared.")
                                 print(f" Current: {self.avatar_info['bbox_shift']}, Prepared: {avatar_info_disk.get('bbox_shift')}")
                                 print(" Re-run with preparation=True.")
                                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                 sys.exit(1)
                       else:
                            print(f"Warning: {self.avatar_info_path} not found. Cannot check bbox_shift consistency.")
                  except Exception as e_info:
                       print(f"Warning: Could not read/check avatar info file: {e_info}")

    def _reload_prepared_data(self):
         """Helper to reload prepared data into instance variables."""
         print("Reloading prepared data into instance variables...")
         all_loaded = True
         try:
            print(f" Loading latents: {self.latents_out_path}")
            self.input_latent_list_cycle = torch.load(self.latents_out_path, map_location='cpu') # Load to CPU first
            if isinstance(self.input_latent_list_cycle, torch.Tensor): # If saved as stacked tensor
                 self.input_latent_list_cycle = list(self.input_latent_list_cycle) # Convert back to list
            print(f" Loading coords: {self.coords_path}")
            with open(self.coords_path, 'rb') as f: self.coord_list_cycle = pickle.load(f)
            print(f" Loading mask coords: {self.mask_coords_path}")
            with open(self.mask_coords_path, 'rb') as f: self.mask_coords_list_cycle = pickle.load(f)

            print(f" Loading frames from: {self.full_imgs_path}")
            input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jp][pn]g')))
            if not input_img_list: raise FileNotFoundError(f"No images found in {self.full_imgs_path}")
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.frame_list_cycle = read_imgs(input_img_list) # read_imgs should return list of BGR uint8 NumPy

            print(f" Loading masks from: {self.mask_out_path}")
            input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jp][pn]g')))
            if not input_mask_list: raise FileNotFoundError(f"No masks found in {self.mask_out_path}")
            input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.mask_list_cycle = read_imgs(input_mask_list) # read_imgs should return list of Grayscale uint8 NumPy

            # --- Validation ---
            print("Validating loaded data...")
            data_lists = {"Coords": self.coord_list_cycle,"Frames": self.frame_list_cycle,
                          "Masks": self.mask_list_cycle,"MaskCoords": self.mask_coords_list_cycle,
                          "Latents": self.input_latent_list_cycle}
            list_lengths = {}
            for name, data_list in data_lists.items():
                 if data_list is None or not isinstance(data_list, list) or len(data_list) == 0:
                      print(f"Error: Data list '{name}' is invalid (None, not list, or empty) after loading.")
                      all_loaded = False; break
                 list_lengths[name] = len(data_list)

            if all_loaded and len(set(list_lengths.values())) > 1:
                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 print("Error: Lengths of loaded data lists do not match!")
                 print(f" Lengths: {list_lengths}")
                 print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 all_loaded = False

            if not all_loaded: raise ValueError("Failed to load or validate all required prepared data.")

            print(f"Successfully reloaded {list_lengths.get('Frames', 0)} frames/masks/coords.")

         except FileNotFoundError as e:
              print(f"Error reloading prepared data: File not found - {e}")
              print(" You may need to run with preparation=True.")
              raise SystemExit(f"Missing prepared file: {e}") # Exit more cleanly
         except Exception as e:
              print(f"Error reloading prepared data: {e}")
              traceback.print_exc()
              raise SystemExit(f"Failed to reload data: {e}")


    def prepare_material(self):
        """Prepares all necessary data for the avatar."""
        # (Keep the prepare_material method code from the previous correct response - no changes needed here)
        # Note: Ensure the _reload_prepared_data() call remains at the end
        print("---------------------------------")
        print(f" Preparing material for avatar: {self.avatar_id}")
        print("---------------------------------")
        osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])

        with open(self.avatar_info_path, "w") as f: json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            print(f"Extracting frames from video: {self.video_path}")
            video2imgs(self.video_path, self.full_imgs_path, ext='.png')
        elif os.path.isdir(self.video_path):
            print(f"Copying frames from directory: {self.video_path}")
            img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            files = sorted([f for f in os.listdir(self.video_path) if f.lower().endswith(img_extensions)])
            if not files: print(f"Error: No image files found in directory {self.video_path}"); sys.exit(1)
            for i, filename in enumerate(tqdm(files, desc="Copying frames")):
                new_filename = f"{i:08d}.png"
                try: shutil.copyfile(os.path.join(self.video_path, filename), os.path.join(self.full_imgs_path, new_filename))
                except Exception as e_copy: print(f"Error copying file {filename}: {e_copy}"); continue
            print(f"Copied and renamed {len(files)} frames.")
        else: print(f"Error: video_path '{self.video_path}' is not a valid file or directory."); sys.exit(1)

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.png')))
        if not input_img_list: print(f"Error: No PNG files found in {self.full_imgs_path}"); sys.exit(1)
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        print("Extracting landmarks and bounding boxes...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        print(f" Found {len(coord_list)} coords, {len(frame_list)} frames initially.")

        print("Encoding VAE latents...")
        input_latent_list, valid_coords, valid_frames, valid_indices = [], [], [], []
        coord_placeholder_val = coord_placeholder()
        global vae
        if 'vae' not in globals() or vae is None: print("Error: VAE model not loaded."); sys.exit(1)

        for i, (bbox, frame) in enumerate(tqdm(zip(coord_list, frame_list), total=len(coord_list), desc="VAE Encoding")):
            if bbox is None or bbox == coord_placeholder_val or frame is None: continue
            x1_coord, y1_coord, x2_coord, y2_coord = bbox
            try:
                 y1_c, y2_c = int(round(y1_coord)), int(round(y2_coord))
                 x1_c, x2_c = int(round(x1_coord)), int(round(x2_coord))
                 h_f, w_f = frame.shape[:2]
                 y1_c, y2_c = max(0, y1_c), min(h_f, y2_c)
                 x1_c, x2_c = max(0, x1_c), min(w_f, x2_c)
                 if x1_c >= x2_c or y1_c >= y2_c: continue
                 crop_frame = frame[y1_c:y2_c, x1_c:x2_c]
                 if crop_frame.size == 0: continue
                 resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                 latents = vae.get_latents_for_unet(resized_crop_frame) # Assumes vae handles device/dtype
                 input_latent_list.append(latents)
                 valid_coords.append(bbox); valid_frames.append(frame); valid_indices.append(i)
            except Exception as e: print(f"Error VAE frame {i}: {e}"); continue

        if not input_latent_list: print("Error: No valid latents generated."); sys.exit(1)
        print(f"Generated {len(input_latent_list)} valid latents.")

        print("Creating data cycles...")
        frame_list_cycle_prep = valid_frames + valid_frames[::-1]
        coord_list_cycle_prep = valid_coords + valid_coords[::-1]
        input_latent_list_cycle_prep = input_latent_list + input_latent_list[::-1]
        num_cycle_frames = len(frame_list_cycle_prep)
        print(f"Cycle length: {num_cycle_frames} frames.")

        print("Generating masks...")
        mask_coords_list_cycle_prep, mask_list_cycle_prep = [], []
        global get_image_prepare_material
        if 'get_image_prepare_material' not in globals(): print("Error: get_image_prepare_material not found."); sys.exit(1)

        processed_indices = set()
        temp_mask_data = {}

        for i, frame in enumerate(tqdm(frame_list_cycle_prep, desc="Mask Generation")):
            face_box = coord_list_cycle_prep[i]
            try:
                 mask, crop_box = get_image_prepare_material(frame, face_box)
                 if mask is None or crop_box is None: print(f"Warning: Mask gen failed frame {i}. Skipping."); continue
                 cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
                 temp_mask_data[i] = (mask, crop_box); processed_indices.add(i)
            except Exception as e: print(f"Error mask gen frame {i}: {e}"); continue

        print(f"Filtering cycles based on {len(processed_indices)} successful masks...")
        final_frame_list_cycle, final_coord_list_cycle, final_input_latent_list_cycle = [], [], []
        final_mask_list_cycle, final_mask_coords_list_cycle = [], []

        # Rebuild based on successfully processed indices
        for i in range(num_cycle_frames):
             if i in processed_indices:
                  frame_path = os.path.join(self.full_imgs_path, f"{i:08d}.png")
                  mask_path = os.path.join(self.mask_out_path, f"{i:08d}.png")
                  if os.path.exists(mask_path) and os.path.exists(frame_path):
                       frame_img = cv2.imread(frame_path) # Read BGR
                       mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read Grayscale
                       if frame_img is not None and mask_img is not None:
                           final_frame_list_cycle.append(frame_img)
                           final_coord_list_cycle.append(coord_list_cycle_prep[i])
                           final_input_latent_list_cycle.append(input_latent_list_cycle_prep[i])
                           mask_data, crop_box_data = temp_mask_data[i]
                           final_mask_list_cycle.append(mask_data)
                           final_mask_coords_list_cycle.append(crop_box_data)
                       else: print(f"Warning: Failed read frame/mask {i}, skipping.")
                  else: print(f"Warning: Missing frame/mask file for index {i}, skipping.")
             # else: print(f"Debug: Index {i} not in processed_indices.") # Optional debug

        print("Saving final prepared data...")
        final_len = len(final_frame_list_cycle)
        if not all(len(lst) == final_len for lst in [final_coord_list_cycle, final_input_latent_list_cycle, final_mask_list_cycle, final_mask_coords_list_cycle]):
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print("Error: Final lengths of prepared data lists do not match after filtering!")
             print(f" Frames: {final_len}, Coords: {len(final_coord_list_cycle)}, Latents: {len(final_input_latent_list_cycle)}")
             print(f" Masks: {len(final_mask_list_cycle)}, MaskCoords: {len(final_mask_coords_list_cycle)}")
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             sys.exit(1)
        if final_len == 0: print("Error: No frames survived preparation filtering."); sys.exit(1)

        with open(self.coords_path, 'wb') as f: pickle.dump(final_coord_list_cycle, f)
        with open(self.mask_coords_path, 'wb') as f: pickle.dump(final_mask_coords_list_cycle, f)
        if isinstance(final_input_latent_list_cycle[0], torch.Tensor):
             final_latents_tensor = torch.stack(final_input_latent_list_cycle)
             torch.save(final_latents_tensor, self.latents_out_path)
        else: torch.save(final_input_latent_list_cycle, self.latents_out_path)

        print("---------------------------------")
        print(" Material preparation complete.")
        print(f" Final cycle length: {final_len} frames.")
        print("---------------------------------")
        self._reload_prepared_data() # Load the finalized data


    # ========================================================================
    # === PARALLELIZED process_frames ========================================
    # ========================================================================
    # ========================================================================
    # === PARALLELIZED process_frames (WITH TIMING LOGS) =====================
    # ========================================================================
    def process_frames(self, res_frame_queue, video_len, gst_video_pipeline, gst_audio_pipeline, skip_save_images, debug=False):
        print(f"Target video length (number of batches/chunks): {video_len}")
        num_workers = max(1, os.cpu_count() - 2)
        print(f"--- Starting process_frames with {num_workers} workers ---")

        if not hasattr(self, 'coord_list_cycle') or not self.coord_list_cycle or len(self.coord_list_cycle) == 0:
            print("Error: Avatar reference data not loaded or empty before process_frames.")
            return

        batch_counter = 0
        total_frames_sent = 0

        # --- ADDED: Initialization for timing diagnostics ---
        last_audio_send_time = None
        time_diffs = []
        # --- END ADDED ---

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            while True:
                current_loop_start_idx = self.idx
                if batch_counter >= video_len:
                    print(f"Processed target number of batches ({batch_counter}/{video_len}). Finishing.")
                    break
                try:
                    batch = res_frame_queue.get(block=True, timeout=10)
                except queue.Empty:
                    print(f"Queue empty timeout ({batch_counter}/{video_len} batches processed).")
                    if batch_counter >= video_len: print("Queue empty and target batches processed. Finishing."); break
                    else: print("Continuing to wait for batches..."); continue

                if batch is None: print("Received None (sentinel value) from queue. Finishing."); break
                if not isinstance(batch, tuple) or len(batch) != 2: print(f"Invalid batch type: {type(batch)}. Skipping."); continue

                frames, audio_chunk = batch
                if not isinstance(frames, list) or len(frames) == 0: print(f"Invalid frames list: {type(frames)}. Skipping."); batch_counter += 1; continue
                if not all(isinstance(f, np.ndarray) for f in frames): print(f"Not all frames are NumPy arrays. Skipping."); batch_counter += 1; continue

                num_frames_in_batch = len(frames)
                args_list = [(self, i, current_loop_start_idx, frame, gst_video_pipeline.width, gst_video_pipeline.height) for i, frame in enumerate(frames)]
                results_with_indices = []
                try:
                     results_with_indices = list(executor.map(process_single_frame_parallel, args_list))
                except Exception as e_map: print(f"Error during executor.map batch {batch_counter}: {e_map}"); batch_counter += 1; self.idx += num_frames_in_batch; continue

                prepared_frames_dict = {}
                successful_frames_in_batch = 0
                for result_tuple in results_with_indices:
                    if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                        i, result_frame = result_tuple
                        if result_frame is not None: prepared_frames_dict[i] = result_frame; successful_frames_in_batch += 1
                    else: print(f"Warning: Worker returned unexpected result format: {result_tuple}")

                prepared_frames = [prepared_frames_dict[i] for i in range(num_frames_in_batch) if i in prepared_frames_dict]

                frames_actually_sent_this_batch = 0
                if prepared_frames and audio_chunk is not None:
                    video_send_success = True
                    try:
                        for frame in prepared_frames:
                            if gst_video_pipeline.send_frame(frame): frames_actually_sent_this_batch += 1
                            else: print("Error sending video frame, stopping sends."); video_send_success = False; break
                    except Exception as e: print(f"Error sending video frame: {e}"); video_send_success = False

                    if video_send_success:
                        try:
                            if isinstance(audio_chunk, np.ndarray) and audio_chunk.dtype == np.int16:

                                # --- ADDED: Timing diagnostics ---
                                current_time = time.time() # Get current time
                                if last_audio_send_time is not None:
                                    time_diff = current_time - last_audio_send_time
                                    # Only print if time_diff is somewhat meaningful (e.g. > 0.001)
                                    if time_diff > 0.001:
                                        print(f"DEBUG (process_frames): Time since last audio send: {time_diff:.4f} s") # Log time difference
                                        time_diffs.append(time_diff) # Append to list
                                # --- END ADDED ---

                                if not gst_audio_pipeline.send_audio(audio_chunk):
                                     print("Error sending audio chunk.")
                                     # If send_audio fails, don't update last_audio_send_time
                                else:
                                     # --- ADDED: Update time only after successful send ---
                                     last_audio_send_time = current_time # Update last send time
                                     # --- END ADDED ---

                            else: print(f"Invalid audio chunk type/dtype: {type(audio_chunk)}, {getattr(audio_chunk, 'dtype', 'N/A')}")
                        except Exception as e: print(f"Error sending audio chunk: {e}")

                elif not prepared_frames and audio_chunk is not None: print("Warning: No frames prepared for batch, audio may desync.")

                total_frames_sent += frames_actually_sent_this_batch
                batch_counter += 1
                self.idx += num_frames_in_batch # Update index based on processed video frames

        print(f"Processing loop finished after {batch_counter} batches. Total frames sent: {total_frames_sent}")

        # --- ADDED: Optional final statistics ---
        try:
             if time_diffs: # Only calculate if list is not empty
                  print(f"DEBUG: Audio send interval stats: Avg={np.mean(time_diffs):.4f}s, StdDev={np.std(time_diffs):.4f}s, Min={np.min(time_diffs):.4f}s, Max={np.max(time_diffs):.4f}s")
        except NameError:
             print("DEBUG: Cannot calculate stats, numpy not imported or time_diffs empty?")
        except Exception as e_stats:
            print(f"DEBUG: Error calculating stats: {e_stats}")
        # --- END ADDED ---


    # ========================================================================
    # === FINAL inference method (WITH CORRECTED AUDIO SLICING & RECON FIX) ==
    # ========================================================================
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        res_frame_queue = queue.Queue()
        gst_video_pipeline = None
        gst_audio_pipeline = None
        process_thread = None
        start_time = time.time()
        frame_count = 0 # Counts frames GENERATED by VAE successfully

        # --- Define target_sr here or ensure it's accessible ---
        target_sr = 48000
        # ---

        try: # Outer try block for setup and inference loop
            print(f"Starting inference for {audio_path}")
            # --- Initialize GStreamer ---
            try:
                # Ensure self.batch_size is available (should be set in __init__)
                if not hasattr(self, 'batch_size'):
                     raise AttributeError("Avatar object missing 'batch_size' attribute.")

                gst_video_pipeline = GStreamerPipeline(fps=fps) # Pass FPS
                gst_audio_pipeline = GStreamerAudio()
                if gst_video_pipeline.process is None or gst_audio_pipeline.process is None:
                    raise RuntimeError("GStreamer pipeline failed to initialize.")
            except Exception as e_gst:
                print(f"Fatal Error initializing GStreamer pipelines: {e_gst}"); return

            # --- Audio Processing ---
            print("Processing audio...")
            try:
                whisper_feature = audio_processor.audio2feat(audio_path)
                whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
                video_len = len(whisper_chunks) # video_len is the number of whisper chunks/iterations

                audio_reader = FFmpegAudioReader(audio_path)
                audio_data = audio_reader.read_full_audio() # Get the FULL audio data
                if audio_data is None: raise ValueError("Failed to read audio data.")
                print(f"Full audio length: {len(audio_data)} samples")

                total_iters = video_len
                print(f"Audio/Feature Chunks (total_iters): {total_iters}, Input Batch Size: {self.batch_size}")

                # Audio is no longer pre-split
            except Exception as e_audio:
                print(f"Fatal Error during audio processing: {e_audio}"); raise

            # --- Setup Processing Thread ---
            self.idx = 0 # Reset index
            process_thread = threading.Thread(target=self.process_frames,
                                              # Pass total_iters (number of generator iterations)
                                              args=(res_frame_queue, total_iters, gst_video_pipeline, gst_audio_pipeline, skip_save_images))
            process_thread.daemon = True
            process_thread.start()

            # --- Main Inference Loop ---
            print("Starting main inference loop...")
            global unet, vae, pe, timesteps, device # Ensure models are accessible
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
                    audio_feature_batch = torch.from_numpy(whisper_batch).to(device=device, dtype=unet.model.dtype, non_blocking=True)
                    audio_feature_batch = pe(audio_feature_batch)

                    if not isinstance(latent_batch, torch.Tensor): latent_batch = torch.stack(latent_batch)
                    latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

                    # --- Inference ---
                    # *** THESE LINES MUST BE PRESENT AND UNCOMMENTED ***
                    pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = vae.decode_latents(pred_latents) # Returns NumPy uint8 NHWC
                    # **************************************************

                    # --- Process VAE Output ---
                    recon_list = []
                    if isinstance(recon, np.ndarray) and recon.ndim == 4 and recon.shape[-1] == 3:
                         # Assuming recon is NHWC BGR uint8 based on previous context
                         # Adjust if your vae.decode_latents returns RGB or a different format
                         recon_list = [f.copy() for f in recon]
                    else:
                        # Add appropriate handling if VAE output is different
                        print(f"VAE output unexpected shape/type: {type(recon)}, {getattr(recon, 'shape', 'N/A')}")
                        continue # Skip this batch if output is wrong

                    # --- Check video frames ---
                    if not recon_list or not all(isinstance(f, np.ndarray) for f in recon_list):
                        print(f"Warning: recon_list empty/invalid iter {i}. Skipping put.")
                        continue

                    num_vid_frames = len(recon_list)
                    frame_count += num_vid_frames # Increment usable frame count

                    # --- Calculate and Slice Correct Audio Chunk ---
                    current_audio_chunk = None
                    if fps > 0:
                        samples_needed_for_batch = int(round(num_vid_frames / fps * target_sr))
                        start_sample = audio_samples_sent
                        end_sample = start_sample + samples_needed_for_batch
                        end_sample = min(end_sample, len(audio_data)) # Ensure not past end

                        if start_sample < end_sample:
                           current_audio_chunk = audio_data[start_sample:end_sample]
                           audio_samples_sent += len(current_audio_chunk)
                        else:
                            print(f"Warning: No more audio data to slice at iter {i}")
                            current_audio_chunk = np.array([], dtype=np.int16)

                        # --- Logging for Chunk Size Comparison ---
                        audio_chunk_samples = len(current_audio_chunk)
                        expected_audio_samples = samples_needed_for_batch
                        print(f"DEBUG (inference): Iter {i}: VidFrames={num_vid_frames}, AudSamples={audio_chunk_samples}, ExpectedAudSamples={expected_audio_samples}, Diff={audio_chunk_samples-expected_audio_samples}")
                        # --- END Logging ---
                    else:
                        print("Warning: fps <= 0, cannot calculate audio chunk size.")
                        continue

                    # --- Queue the data ---
                    if current_audio_chunk is not None:
                         res_frame_queue.put((recon_list, current_audio_chunk))
                    # ---

                except Exception as e_loop:
                    print(f"\n!!!!! Error in main inference loop iteration {i} !!!!!")
                    traceback.print_exc(); continue # Use traceback to see where error occurred
            # --- End of Main Loop ---

        except Exception as e_inf:
            print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"FATAL ERROR during inference setup or loop: {e_inf}")
            traceback.print_exc()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if process_thread is not None and process_thread.is_alive():
                 try: res_frame_queue.put(None)
                 except: pass

        finally:
            # --- ENSURE FINAL CLEANUP AND FPS PRINT ---
            print("\n--- Starting Final Cleanup ---")
            # ... (Cleanup code: signal thread, join thread, stop GStreamer, print FPS - same as before) ...
             # Signal End to Processing Thread
            try:
                print("Signaling process_frames to finish (sending None)...")
                res_frame_queue.put(None)
            except Exception as e_put_none:
                print(f"Note: Exception putting None sentinel: {e_put_none}")

            # Wait for Processing Thread
            if process_thread is not None:
                print("Waiting for process_frames thread to join...")
                process_thread.join(timeout=30) # Wait up to 30s
                if process_thread.is_alive(): print("⚠️ Warning: process_frames thread did not finish.")
                else: print("✅ process_frames thread joined.")
            else: print("process_frames thread was not started or already finished.")

            # Stop GStreamer
            print("Stopping GStreamer pipelines...")
            if gst_video_pipeline: gst_video_pipeline.stop()
            if gst_audio_pipeline: gst_audio_pipeline.stop()
            print("GStreamer stop commands issued.")

            # Calculate and Print FPS
            print("Calculating final FPS...")
            total_elapsed_time = time.time() - start_time
            avg_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0
            print("\n========================================")
            print(f" Total elapsed time: {total_elapsed_time:.2f} s")
            print(f" Total frame count generated by VAE: {frame_count}") # Use frame_count
            print(f" Final calculated average FPS (based on VAE output): {avg_fps:.2f}")
            print("========================================")
            print(">>> Inference method finished.")


# --- Main Execution Block ---
# (Keep the __main__ block from the previous correct response - no changes needed here)
if __name__ == "__main__":
    '''
    Real-time streaming script for MuseTalk.
    '''
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4) # Default batch size
    parser.add_argument("--skip_save_images", action="store_true", help="Legacy flag, not used here")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    try:
        inference_config = OmegaConf.load(args.inference_config)
        print("Loaded inference config:")
        print(OmegaConf.to_yaml(inference_config))
    except Exception as e_conf:
        print(f"Error loading inference config '{args.inference_config}': {e_conf}")
        sys.exit(1)

    for avatar_id in inference_config:
        print(f"\n===== Processing Avatar: {avatar_id} =====")
        try:
            config_data = inference_config[avatar_id]
            data_preparation = config_data.get("preparation", False)
            video_path = config_data.get("video_path")
            bbox_shift = config_data.get("bbox_shift", 0)
            audio_clips = config_data.get("audio_clips")

            if not video_path or not audio_clips:
                print(f"Error: Missing 'video_path' or 'audio_clips' for avatar '{avatar_id}'.")
                continue

            avatar = Avatar(
                avatar_id=avatar_id, video_path=video_path, bbox_shift=bbox_shift,
                batch_size=args.batch_size, preparation=data_preparation
            )

            for audio_num, audio_path in audio_clips.items():
                print(f"\n--- Inferring using Audio: {audio_num} ({audio_path}) ---")
                if not os.path.exists(audio_path):
                     print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                     continue

                avatar.inference(
                    audio_path=audio_path, out_vid_name=audio_num,
                    fps=args.fps, skip_save_images=args.skip_save_images
                )
                print(f"--- Finished inference for Audio: {audio_num} ---")

        except SystemExit as e: # Catch sys.exit() calls from Avatar methods
             print(f"Exiting script due to error during processing avatar '{avatar_id}': {e}")
             break # Stop processing further avatars
        except Exception as e_avatar:
            print(f"\n!!!!! Error processing avatar '{avatar_id}' !!!!!")
            traceback.print_exc()
            print(f"!!!!! Skipping to next avatar if any !!!!!")
            continue

    print("\n===== All Avatars Processed =====")