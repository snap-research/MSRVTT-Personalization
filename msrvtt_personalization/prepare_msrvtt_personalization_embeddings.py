import sys
sys.path.append("../models/")
from arcface.backbones import get_model

import torch
import torchvision.transforms as transforms
import numpy as np
import clip
import cv2

from tqdm import tqdm
from PIL import Image
import json
import glob
import os
import gzip
import pickle
import argparse


def build_arcface_model(network, weight):
    net = get_model(network, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    return net

@torch.no_grad()
def extract_arcface_embeddings(image_list, model):
    def load_image(pil_image):
        img = np.array(pil_image)
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img.div_(255).sub_(0.5).div_(0.5)
        return img

    images = torch.stack([load_image(i) for i in image_list])
    if torch.cuda.is_available():
        images = images.to("cuda")
    features = model(images).cpu()
    return features

def pil_to_tensor(pil_image: Image.Image, square_mode: str=None, normalization_mode: str=None):
    width, height = pil_image.size
    if square_mode == "cropping": # center cropping
        square_size = min(width, height)
        square_image = transforms.functional.center_crop(pil_image, (square_size, square_size))
    elif square_mode == "padding": # pad to square
        if width == height:
            square_image = pil_image
        elif width > height:
            square_image = Image.new(pil_image.mode, (width, width), color=(255, 255, 255))
            square_image.paste(pil_image, (0, (width - height) // 2))
        else:
            square_image = Image.new(pil_image.mode, (height, height), color=(255, 255, 255))
            square_image.paste(pil_image, ((height - width) // 2, 0))
    else:
        raise AssertionError(f"Unsupported square mode: {square_mode}")

    if normalization_mode == "clip":
        avg = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif normalization_mode == "imagenet":
        avg = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise AssertionError(f"Unsupported normalization mode: {normalization_mode}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(avg, std),
    ])

    transformed_image = transform(square_image)

    return transformed_image

def add_white_background(pil_image_rgba: Image.Image):
    assert pil_image_rgba.mode == "RGBA"
    np_image = np.array(pil_image_rgba)
    background_mask = np_image[:, :, 3] == 0
    np_image[background_mask] = [255, 255, 255, 255]
    np_image = np_image[:, :, :3]
    image = Image.fromarray(np_image, mode="RGB")
    return image

    
if __name__ == "__main__":
    # conda activate /nfs/code/miniconda3/envs/stablediffusion
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_annotation_folder", type=str, default="msrvtt_personalization_annotation")
    parser.add_argument("--benchmark_data_folder", type=str, default="msrvtt_personalization_data")
    parser.add_argument("--benchmark_embeddings_folder", type=str, default="msrvtt_personalization_embeddings")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize CLIP model
    clip_model, _ = clip.load("ViT-L/14")
    clip_model = clip_model.to(device)
    
    # Initialize DINO model
    dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    dino_model = dino_model.to(device)
    
    # Initialize arcface model
    arcface_model = build_arcface_model(
        network = "r100",
        weight = "../models/arcface/weight/backbone.pth"
    ).to(device)

    # Read video list
    video_list = [os.path.basename(i) for i in sorted(glob.glob(os.path.join(args.benchmark_data_folder, "*")))]
    
    for video in tqdm(video_list):
        # Create output folder
        output_folder = os.path.join(args.benchmark_embeddings_folder, video)
        os.makedirs(output_folder, exist_ok=True)
        
        # Extract clip embeddings for video caption
        summary_text = os.path.join(args.benchmark_data_folder, video, "summary_text.txt")
        summary_text = open(summary_text).read()

        with torch.no_grad():
            encoded_text = clip.tokenize([summary_text], truncate=True).to(device)
            text_embeddings = clip_model.encode_text(encoded_text).cpu()[0]
        
        output_file = os.path.join(output_folder, "text_embeddings.pt")
        torch.save(text_embeddings, output_file)
        
        # Extract clip embeddings for video frames
        video_frames = sorted(glob.glob(os.path.join(args.benchmark_data_folder, video, "frame*.jpg")))
        video_frames = [Image.open(video_frame) for video_frame in video_frames]
        video_frames = [pil_to_tensor(video_frame, square_mode="cropping", normalization_mode="clip") for video_frame in video_frames] # center crop video frames to fairly evaluate videos with different aspect ratio
        video_frames = torch.stack(video_frames)
        
        with torch.no_grad():
            video_frames = video_frames.to(device)
            video_embeddings = clip_model.encode_image(video_frames).cpu()
        
        output_file = os.path.join(output_folder, "video_embeddings.pt")
        torch.save(video_embeddings, output_file)

        # Extract dino embeddings for subject and object images
        all_word_tags = os.path.join(args.benchmark_data_folder, video, "word_tags.json")
        all_word_tags = json.load(open(all_word_tags))
        word_tags = all_word_tags["subject"] + all_word_tags["object"]
        subject_image_embeddings = {}
        
        for word_tag in word_tags:
            subject_images = sorted(glob.glob(os.path.join(args.benchmark_data_folder, video, word_tag.replace(" ", "_").replace("/", "_") + ".frame*.png")))
            subject_images = [Image.open(subject_image) for subject_image in subject_images]
            subject_images = [add_white_background(subject_image) for subject_image in subject_images] # add white background for the subject subject_images (with transparency channel)
            subject_images = [pil_to_tensor(subject_image, square_mode="padding", normalization_mode="imagenet") for subject_image in subject_images]
            subject_images = torch.stack(subject_images)
            
            with torch.no_grad():
                subject_images = subject_images.to(device)
                subject_image_embeddings[word_tag] = dino_model(subject_images).cpu()

        output_file = os.path.join(output_folder, "subject_embeddings.pkl.gz")
        with gzip.open(output_file, "wb") as f:
            pickle.dump(subject_image_embeddings, f)

        # Extract arcface embeddings for face crops (if the video has exactly one subject with one or more face crops)
        if len(all_word_tags["subject_with_face"]) == 1:
            face_word_tag = all_word_tags["subject_with_face"][0]
            
            word_tags_masks = os.path.join(args.benchmark_annotation_folder, video + ".word_tags_masks.pkl.gz")
            word_tags_masks = pickle.load(gzip.open(word_tags_masks, "rb"))
            masks = word_tags_masks[face_word_tag]
            
            # For face embeddings, use face crop without masking (arcface embeddings should be invariant with background)
            face_crops = []
            for mask in masks:
                if mask["is_face_crop"]:
                    filename = os.path.join(args.benchmark_data_folder, video, mask["filename"])
                    box_xyxy = mask["box_xyxy"]
                    frame = Image.open(filename).convert("RGB")
                    face_crops.append(frame.crop(box_xyxy))
                    
            face_embeddings = extract_arcface_embeddings(face_crops, arcface_model)
            output_file = os.path.join(output_folder, "face_embeddings.pt")
            torch.save(face_embeddings, output_file)