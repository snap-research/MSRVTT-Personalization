from PIL import Image
from tqdm import tqdm
import numpy as np
import yaml
import os
import argparse

def process_image(image_path):
    pil_image = Image.open(image_path)
    np_image = np.array(pil_image)
    background_mask = np_image[:, :, 3] == 0
    np_image[background_mask] = [255, 255, 255, 255]
    np_image = np_image[:, :, :3]
    image = Image.fromarray(np_image, mode="RGB")

    foreground_mask = 1 - background_mask
    foreground_mask = (foreground_mask * 255).astype(np.uint8)
    foreground_mask = Image.fromarray(foreground_mask, mode='P')
    foreground_mask.putpalette([0, 0, 0, 255, 255, 255]) # Black for 0, White for 255

    return image, foreground_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_folder", type=str, default="../../msrvtt_personalization")
    parser.add_argument("--video_list", type=str, default="../../msrvtt_personalization/subject_mode_videos.txt")
    parser.add_argument("--prompt_list", type=str, default="../../msrvtt_personalization/subject_mode_prompts.txt")
    parser.add_argument("--word_tag_image_list", type=str, default="../../msrvtt_personalization/subject_mode_word_tag_single_subject_image.txt")
    parser.add_argument("--input_data_folder", type=str, default="msrvtt_personalization_subject_mode/inputs")
    args = parser.parse_args()

    if not os.path.exists(args.input_data_folder):
        os.makedirs(args.input_data_folder)
        
    video_list = open(args.video_list, "r").read().splitlines()
    prompt_list = open(args.prompt_list, "r").read().splitlines()
    word_tag_image_list = open(args.word_tag_image_list, "r").read().splitlines()
    assert len(video_list) == len(prompt_list) == len(word_tag_image_list)
    
    for i in tqdm(range(len(video_list))):
        video = video_list[i]
        prompt = prompt_list[i].lower()
        word_tag_image = eval(word_tag_image_list[i])
        word_tag = list(word_tag_image.keys())[0]
        image_path = os.path.join(args.benchmark_folder, word_tag_image[word_tag][0])
        
        image, mask = process_image(image_path)
        width, height = image.size
        
        video_folder = os.path.join(args.input_data_folder, video)
        os.makedirs(video_folder, exist_ok=True)
        
        config_path = os.path.join(video_folder, "config.yaml")
        image_path = os.path.join(video_folder, "image.png")
        mask_path = os.path.join(video_folder, "mask.png")
        
        image.save(image_path)
        mask.save(mask_path)
        
        data = {
            "img_path" : image_path,
            "mask_path" : mask_path,
            "text_prompt": prompt,
            "replace_word": word_tag,
            "seed": 0,
            "bbox": [0, 0, float(width), float(height)]
        }

        with open(config_path, "w") as f:
            yaml.dump(data, f, width=1000, default_flow_style=False)
