from tqdm import tqdm
import glob
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_video_folder", type=str, default="msrvtt_personalization_subject_mode/outputs")
    parser.add_argument("--benchmark_embeddings_folder", type=str, default="../../msrvtt_personalization/msrvtt_personalization_embeddings")
    args = parser.parse_args()
    
    videos = sorted(glob.glob(os.path.join(args.output_video_folder, "*.mp4")))
    
    for video in tqdm(videos):
        video_name = os.path.splitext(os.path.basename(video))[0]
        embeddings_files = ["text_embeddings.pt", "video_embeddings.pt", "subject_embeddings.pkl.gz"]
        
        for embeddings_file in embeddings_files:
            source_path = os.path.join(args.benchmark_embeddings_folder, video_name, embeddings_file)
            target_path = os.path.join(args.output_video_folder, video_name + "." + embeddings_file)
            os.system(f"cp {source_path} {target_path}")
