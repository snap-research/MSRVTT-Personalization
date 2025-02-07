from collections import defaultdict
from tqdm import tqdm
import random
import gzip
import pickle
import json
import glob
import os
import argparse

random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_data_folder", type=str, default="msrvtt_personalization_data")
    parser.add_argument("--benchmark_mode", choices=["subject_mode", "face_mode", "all"], default="all",
                        help="Currently support subject mode and face mode. You can also customize your own mode.")
    args = parser.parse_args()
    
    video_paths = sorted(glob.glob(os.path.join(args.benchmark_data_folder, "*")))
    
    # Prepare the lists for the subject mode
    if args.benchmark_mode == "all" or args.benchmark_mode == "subject_mode":
        video_list = []
        summary_text_list = []
        word_tag_single_subject_image = []
        word_tag_multi_subject_images = []
        word_tag_multi_subject_images_with_background = []
        
        for video_path in tqdm(video_paths):
            word_tags_file = os.path.join(video_path, "word_tags.json")
            word_tags = json.load(open(word_tags_file, "r"))
            
            summary_text_file = os.path.join(video_path, "summary_text.txt")
            summary_text = open(summary_text_file, "r").read()
            
            # only sample the videos with exactly one subject
            if len(word_tags["subject"]) == 1: 
                subject_word_tag = word_tags["subject"][0]
                background_word_tag = word_tags["background"][0]
                
                multi_images = sorted(glob.glob(os.path.join(video_path, subject_word_tag.replace(" ", "_").replace("/", "_") + ".frame*.png")))
                single_image = [multi_images[len(multi_images)//2]]
                background_image = [os.path.join(video_path, "background.jpg")]
                
                video_list.append(os.path.basename(video_path))
                summary_text_list.append(summary_text)
                word_tag_single_subject_image.append(str({subject_word_tag : single_image}))
                word_tag_multi_subject_images.append(str({subject_word_tag : multi_images}))
                word_tag_multi_subject_images_with_background.append(str({subject_word_tag : multi_images, background_word_tag : background_image}))
                
        with open("subject_mode_videos.txt", "w") as f:
            f.write("\n".join(video_list))
        with open("subject_mode_prompts.txt", "w") as f:
            f.write("\n".join(summary_text_list))
        with open("subject_mode_word_tag_single_subject_image.txt", "w") as f:
            f.write("\n".join(word_tag_single_subject_image))
        with open("subject_mode_word_tag_multi_subject_images.txt", "w") as f:
            f.write("\n".join(word_tag_multi_subject_images))
        with open("subject_mode_word_tag_multi_subject_images_with_background.txt", "w") as f:
            f.write("\n".join(word_tag_multi_subject_images_with_background))

    # Prepare the lists for the face mode
    if args.benchmark_mode == "all" or args.benchmark_mode == "face_mode":
        video_list = []
        summary_text_list = []
        word_tag_single_face_image = []
        word_tag_multi_face_images = []
        
        for video_path in tqdm(video_paths):
            word_tags_file = os.path.join(video_path, "word_tags.json")
            word_tags = json.load(open(word_tags_file, "r"))
            
            summary_text_file = os.path.join(video_path, "summary_text.txt")
            summary_text = open(summary_text_file, "r").read()
            
            if len(word_tags["subject_with_face"]) == 1: 
                subject_word_tag = word_tags["subject_with_face"][0]
                
                multi_images = sorted(glob.glob(os.path.join(video_path, subject_word_tag.replace(" ", "_").replace("/", "_") + ".face.frame*.png")))
                single_image = [multi_images[len(multi_images)//2]]
                
                video_list.append(os.path.basename(video_path))
                summary_text_list.append(summary_text)
                word_tag_single_face_image.append(str({subject_word_tag : single_image}))
                word_tag_multi_face_images.append(str({subject_word_tag : multi_images}))
                
        with open("face_mode_videos.txt", "w") as f:
            f.write("\n".join(video_list))
        with open("face_mode_prompts.txt", "w") as f:
            f.write("\n".join(summary_text_list))
        with open("face_mode_word_tag_single_face_crop.txt", "w") as f:
            f.write("\n".join(word_tag_single_face_image))
        with open("face_mode_word_tag_multi_face_crops.txt", "w") as f:
            f.write("\n".join(word_tag_multi_face_images))