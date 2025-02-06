from diffusers import StableDiffusionInpaintPipeline
from torchvision.transforms import ToTensor
from scipy.ndimage import binary_fill_holes
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import resize
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import cv2
import pickle
import gzip
import argparse
import json
import glob
import os
import re


def convert_timestamp_to_second(timestamp):
    timestamp = datetime.strptime(timestamp, "%H:%M:%S.%f")
    return timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second + timestamp.microsecond / 1_000_000
    
def extract_clip(video_path, output_path, resolution, start_sec, end_sec):    
    video = VideoFileClip(video_path)
    clip = video.subclip(start_sec, end_sec)
    clip = resize(clip, newsize=resolution)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    
def sd2_preprocess(image_batch, width=512, height=512):
    tensor_image_batch = []
    for image in image_batch:
        image = image.resize((width, height))
        image = ToTensor()(image)
        tensor_image_batch.append(image)
    return torch.stack(tensor_image_batch)
    
if __name__ == "__main__":
    # conda activate /nfs/code/miniconda3/envs/stablediffusion
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, default="msrvtt_videos")
    parser.add_argument("--benchmark_annotation_folder", type=str, default="msrvtt_personalization_annotation")
    parser.add_argument("--benchmark_data_folder", type=str, default="msrvtt_personalization_data")
    args = parser.parse_args()
    
    # parse input arguments
    assert os.path.exists(args.video_folder)
    assert os.path.exists(args.benchmark_annotation_folder)
    if os.path.exists(args.benchmark_data_folder):
        os.system("rm -rf %s"%args.benchmark_data_folder)
    os.makedirs(args.benchmark_data_folder)
    
    # read video list
    videos = sorted(glob.glob(os.path.join(args.benchmark_annotation_folder, "*.vid_info.json")))
    videos = [os.path.basename(video).split(".")[0] for video in videos]

    # initialize StableDiffusionInpaint model
    inpainting_model = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    inpainting_model = inpainting_model.to("cuda")
    inpainting_model.set_progress_bar_config(disable=True)

    # process video
    for video_name in tqdm(videos):
        
        # read input data
        video_info_file = os.path.join(args.benchmark_annotation_folder, video_name + ".vid_info.json")
        video_info = json.load(open(video_info_file))
        word_tags_masks_file = os.path.join(args.benchmark_annotation_folder, video_name + ".word_tags_masks.pkl.gz")
        word_tags_masks = pickle.load(gzip.open(word_tags_masks_file, "rb"))
        
        video_file = os.path.join(args.video_folder, video_info["video"])
        resolution = video_info["resolution"]
        start_sec = convert_timestamp_to_second(video_info["timestamp"][0])
        end_sec = convert_timestamp_to_second(video_info["timestamp"][1])
        caption = video_info["caption"]
        word_tags = video_info["word_tags"]
        
        # setup output filename and folder
        output_folder = os.path.join(args.benchmark_data_folder, video_name)
        os.makedirs(output_folder)
        output_video_file = os.path.join(output_folder, "video.mp4")
        output_frame1_file = os.path.join(output_folder, "frame1.jpg")
        output_frame2_file = os.path.join(output_folder, "frame2.jpg")
        output_frame3_file = os.path.join(output_folder, "frame3.jpg")
        output_caption_file = os.path.join(output_folder, "summary_text.txt")
        output_word_tags_file = os.path.join(output_folder, "word_tags.json")
        output_background_file = os.path.join(output_folder, "background.jpg")
        
        # split video
        extract_clip(video_file, output_video_file, resolution, start_sec, end_sec)
        
        # extract video frames
        video = VideoFileClip(output_video_file)
        Image.fromarray(video.get_frame(0.05 * (end_sec - start_sec))).save(output_frame1_file, format="JPEG")
        Image.fromarray(video.get_frame(0.50 * (end_sec - start_sec))).save(output_frame2_file, format="JPEG")
        Image.fromarray(video.get_frame(0.95 * (end_sec - start_sec))).save(output_frame3_file, format="JPEG")
        
        # save caption file
        with open(output_caption_file, "w") as f:
            f.write(caption)
        
        # save word tags file
        with open(output_word_tags_file, "w") as f:
            json_str = json.dumps(word_tags, indent=4)
            json_str = re.sub(
                r'\[\n\s+(.*?)\n\s+\]', lambda m: '[' + re.sub(r'\s*,\s*', ', ', m.group(1).replace('\n', '')) + ']',
                json_str,flags=re.DOTALL
            )
            f.write(json_str)
        
        # save word tags masks
        for word_tag, masks in word_tags_masks.items():
            for mask in masks:
                filename = mask["filename"]
                x0, y0, x1, y1 = mask["box_xyxy"]
                np_mask = np.array(mask["mask"])
                is_face_crop = mask["is_face_crop"]
                
                if filename == "frame1.jpg":
                    np_image = np.array(Image.open(output_frame1_file).convert("RGBA"))
                elif filename == "frame2.jpg":
                    np_image = np.array(Image.open(output_frame2_file).convert("RGBA"))
                elif filename == "frame3.jpg":
                    np_image = np.array(Image.open(output_frame3_file).convert("RGBA"))

                # cropping
                np_image = np_image[y0:y1, x0:x1]
                                
                # masking (mask is sometimes larger than cropped image; cropping mask to the same shape in such case)
                h, w, _ = np_image.shape
                np_mask = np_mask[:h, :w]
                np_image[...,-1] = np_image[...,-1] * np_mask
                
                # save image (note that JPEG does not support 4-channels image)
                if not is_face_crop:
                    output_image_file = os.path.join(output_folder, word_tag.replace(" ", "_").replace("/", "_") + "." + filename.replace(".jpg", ".png"))
                else:
                    output_image_file = os.path.join(output_folder, word_tag.replace(" ", "_").replace("/", "_") + ".face." + filename.replace(".jpg", ".png"))
                Image.fromarray(np_image, "RGBA").save(output_image_file)
                
        # prepare background image (inpainting from frame2.jpg)
        pil_image = Image.open(output_frame2_file).convert("RGB")
        w, h = pil_image.size

        # prepare foreground mask
        foreground_mask = np.zeros((h, w))
        for word_tag, masks in word_tags_masks.items():
            for mask in masks:
                if mask["filename"] == "frame2.jpg":
                    x0, y0, x1, y1 = mask["box_xyxy"]
                    mask = mask["mask"]
                    foreground_mask[y0:y1, x0:x1] = np.logical_or(foreground_mask[y0:y1, x0:x1], mask)

        # run dilation for foreground mask
        kernel = np.ones((4, 4), np.uint8)
        foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=3)
        foreground_mask = binary_fill_holes(foreground_mask).astype(float)
        pil_foreground_mask = Image.fromarray((foreground_mask * 255).astype(np.uint8))

        # run inpainting
        pil_inpainting_image = inpainting_model(
            prompt=["clean and empty background"],
            image=sd2_preprocess([pil_image]),
            mask_image=sd2_preprocess([pil_foreground_mask]),
            negative_prompt=["any human or any object, complex pattern and texture"],
            guidance_scale=12.0,
        ).images[0]

        # combine results with original image
        original_image = np.array(pil_image)
        foreground_mask = np.array(pil_foreground_mask).astype(float)[..., None] / 255.
        inpainting_image = np.array(pil_inpainting_image.resize((w,h)).convert("RGB"))
        combined_image = inpainting_image * foreground_mask + original_image * (1 - foreground_mask)

        # save inpainted background image
        cv2.imwrite(output_background_file, combined_image[...,::-1])
