import ffmpeg
import argparse
import os
# import pyaudio
# import threading
import queue
from omegaconf import OmegaConf
# import soundfile as sf
# import wave
import subprocess
import numpy as np
import cv2  # Used elsewhere (e.g., for image reading/writing)
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
import shutil
import time
from PIL import Image  # For resizing in our new stream part

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
pe = pe.half()
vae.vae = vae.vae.half()
unet.model = unet.model.half()

class GStreamerAudio:
    """
    Use GStreamer audio streaming via gst-launch-1.0 (using fdsrc) to push PCM audio.
    This replacement avoids using gi (which is not well supported on Windows).
    """
    def __init__(self):
        # Replace appsrc with fdsrc that reads from stdin.
        pipeline_str = (
            "fdsrc fd=0 ! "
            "audio/x-raw,format=S16LE,channels=2,rate=48000,layout=interleaved ! "
            "queue ! audioconvert ! queue ! audioresample ! "
            "queue ! opusenc ! queue ! rtpopuspay ! "
            "udpsink host=127.0.0.1 port=5001 sync=false"
        )
        # Launch gst-launch-1.0 with the pipeline string.
        # Using shell=True so that the entire pipeline is passed as a single command.
        self.process = subprocess.Popen(
            f"gst-launch-1.0 {pipeline_str}",
            stdin=subprocess.PIPE,
            shell=True
        )
        print("✅ GStreamer audio pipeline started using gst-launch-1.0")

    def send_audio(self, audio_data):
        """ Once, push complete PCM audio data to GStreamer via stdin """
        try:
            self.process.stdin.write(audio_data.tobytes())
            self.process.stdin.flush()
            print(f"✅ Pushed complete audio, {len(audio_data)} samples")
        except Exception as e:
            print("❌ Error pushing audio:", e)

    def stop(self):
        """ Stop the audio streaming """
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait()
                print("✅ GStreamer audio pipeline closed")
            except Exception as e:
                print("❌ Error stopping GStreamer audio pipeline:", e)

class FFmpegAudioReader:
    """ Use FFmpeg to read an entire audio file and convert it to PCM """
    def __init__(self, audio_file):
        self.audio_file = audio_file
        probe = ffmpeg.probe(audio_file)
        self.sample_rate = int(probe['streams'][0]['sample_rate'])
        self.channels = int(probe['streams'][0]['channels'])

    def read_full_audio(self):
        """ Use FFmpeg to read the full audio file """
        process = subprocess.Popen(
            ["ffmpeg", "-i", self.audio_file, "-f", "s16le", "-ac", "2", "-ar", "48000", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        raw_data = process.stdout.read()
        process.stdout.close()
        process.wait()

        if not raw_data:
            print("❌ Failed to read audio file!")
            return None

        audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
        print(f"✅ Read full audio, {len(audio_data)} samples")
        return audio_data

def split_audio(audio_data, num_chunks):
    """ Split audio data into num_chunks parts """
    if num_chunks <= 1:
        return [audio_data]

    chunk_size = len(audio_data) // num_chunks
    chunks = [audio_data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

    # If there are remaining samples, add them to the last chunk
    remainder = len(audio_data) % num_chunks
    if remainder > 0:
        chunks[-1] = np.vstack((chunks[-1], audio_data[-remainder:]))

    print(f"✅ Audio split into {num_chunks} chunks, each approx. {chunk_size} samples")
    return chunks

# ---------------------------------------------------------------------------
# New GStreamerPipeline (video streaming) without using cv2 for streaming.
# This version uses a direct pipe to gst-launch-1.0 with fdsrc and videoparse.
# ---------------------------------------------------------------------------
class GStreamerPipeline:
    def __init__(self, width=640, height=480, fps=25, host="127.0.0.1", port=5000):
        self.width = width
        self.height = height
        self.fps = fps
        self.host = host
        self.port = port

        # GStreamer pipeline string using fdsrc and videoparse to read raw RGB data from stdin.
        pipeline_str = (
            "fdsrc fd=0 ! "
            f"videoparse format=rgb width={self.width} height={self.height} framerate={self.fps}/1 ! "
            "videoconvert ! video/x-raw ! "
            "queue ! x264enc bitrate=8000 tune=zerolatency ! "
            "rtph264pay ! "
            f"udpsink host={self.host} port={self.port} sync=false"
        )
        self.process = subprocess.Popen(
            f"gst-launch-1.0 {pipeline_str}",
            stdin=subprocess.PIPE,
            shell=True
        )
        print("✅ GStreamer video pipeline started using direct pipe (no cv2)")

    def send_frame(self, frame):
        """
        Send a video frame to the pipeline.
        The frame must be a numpy array with shape (height, width, 3) in RGB format.
        If the frame dimensions do not match, it is resized using PIL.
        """
        try:
            # Resize frame using PIL if needed
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                img = Image.fromarray(frame)
                img = img.resize((self.width, self.height), Image.LANCZOS)
                frame = np.array(img)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Write the raw frame bytes to the pipeline's stdin.
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            print("✅ Frame pushed to GStreamer video pipe")
        except Exception as e:
            print("❌ Error pushing frame:", e)

    def stop(self):
        """ Stop the GStreamer video streaming pipeline. """
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait()
                print("✅ GStreamer video pipeline closed")
            except Exception as e:
                print("❌ Error stopping GStreamer video pipeline:", e)

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.{ext.strip('.')}", frame)
            count += 1
        else:
            break

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

@torch.no_grad() 
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.avatar_path = f"./results/avatars/{avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs" 
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift   
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()
        
    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n) ")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avatar: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else: 
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)
                
            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue) ")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avatar: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:  
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)
    
    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)
            
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        
        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # placeholder if the bbox is not sufficient 
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            
            face_box = self.coord_list_cycle[i]
            mask, crop_box = get_image_prepare_material(frame, face_box)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)
            
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
            
        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))
        
    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        print(video_len)
        while True:
            if self.idx >= video_len - 1:
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
      
            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                continue
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            if not skip_save_images:
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)
            self.idx = self.idx + 1
    
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)   
        print("start inference")

        gst_pipeline = GStreamerPipeline(width=640, height=480, fps=fps, host="127.0.0.1", port=5000)
        audio_sender = GStreamerAudio()
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)

        total_iters = int(np.ceil(float(len(whisper_chunks)) / self.batch_size))
        audio_reader = FFmpegAudioReader(audio_path)
        audio_data = audio_reader.read_full_audio()
        audio_chunks = split_audio(audio_data, total_iters)

        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
    
        frame_count = 0
        start_time = time.time()  # record start time of first frame inference

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            start = time.time()
            # Process audio features
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype, non_blocking=True)
            audio_feature_batch = pe(audio_feature_batch)
            print(f"⏱️ Processing audio features took: {time.time() - start:.4f} seconds")
            
            start = time.time()
            # Process latent_batch
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            print(f"⏱️ GPU IO took: {time.time() - start:.4f} seconds")
            
            start = time.time()
            # Run UNet to generate mouth animation
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            print(f"⏱️ GPU computation took: {time.time() - start:.4f} seconds")

            start = time.time()
            recon = vae.decode_latents(pred_latents)
            print(f"⏱️ Decoding took: {time.time() - start:.4f} seconds")

            start = time.time()
            audio_sender.send_audio(audio_chunks[i])
            # Stream frames to GStreamer sequentially
            for j, res_frame in enumerate(recon):
                frame_count += 1
                print("✅ Pushing video frame...")
                gst_pipeline.send_frame(res_frame)
            print(f"⏱️ Streaming took: {time.time() - start:.4f} seconds")
        ##############################################
        # Step 4: After finishing, calculate final FPS
        ##############################################
        total_elapsed_time = time.time() - start_time
        print(f"\nTotal elapsed time: {total_elapsed_time:.2f} s")
        print(f"\nTotal frame count: {frame_count}")
        avg_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0
        print(f"\nFinal calculated average FPS: {avg_fps:.2f}")
        gst_pipeline.stop()
        audio_sender.stop()
        
if __name__ == "__main__":
    '''
    This script simulates online chatting and applies necessary pre-processing (e.g., face detection and parsing) in advance.
    During online chatting, only UNet and the VAE decoder are used, making MuseTalk real-time.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", 
                        type=str, 
                        default="configs/inference/realtime.yaml")
    parser.add_argument("--fps", 
                        type=int, 
                        default=25)
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=4)
    parser.add_argument("--skip_save_images",
                        action="store_true",
                        help="Whether to skip saving images for better generation speed calculation")

    args = parser.parse_args()
    
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)
    
    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        bbox_shift = inference_config[avatar_id]["bbox_shift"]
        avatar = Avatar(
            avatar_id=avatar_id, 
            video_path=video_path, 
            bbox_shift=bbox_shift, 
            batch_size=args.batch_size,
            preparation=data_preparation)
        
        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print("Inferring using:", audio_path)
            avatar.inference(audio_path, 
                             audio_num, 
                             args.fps,
                             args.skip_save_images)

