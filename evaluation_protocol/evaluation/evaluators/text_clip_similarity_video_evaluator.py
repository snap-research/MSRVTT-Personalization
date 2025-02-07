from typing import Dict

import os
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils
from torch.utils.data.dataloader import DataLoader
import tqdm

from dataset.batching import flexible_batch_elements_collate_fn
from dataset.simple_map_video_folder_dataset import SimpleMapVideoFolderDataset
from dataset.transforms import pil_to_tensor_clip, pil_to_tensor_clip_longcrop, stack_video_frames

from evaluation.metrics.distributed_text_clip_similarity import DistributedTextClipSimilarity
from utils.distributed.distributed_utils import print_r0

class TextClipSimilarityVideoEvaluator():
    """
    Class representing an evaluator
    """
    def __init__(self, evaluator_config: Dict):
        # Root where the videos are saved
        self.generated_video_root = evaluator_config["generated_video_root"]
        # The type of crop to perform on the videos. Short crops to shortsize x shortside px,
        # long crops to longsize x longside px introducing padding where necessary
        self.crop_type = evaluator_config.get("crop_type", "short")
        if self.crop_type not in ["short", "long"]:
            raise Exception("Invalid crop type: {}".format(self.crop_type))
        
        self.frames_count = evaluator_config["frames_count"]
        self.batch_size = evaluator_config["batch_size"]
        self.num_workers = evaluator_config["num_workers"]

    def get_dataset(self, directory: str) -> SimpleMapVideoFolderDataset:
        """
        Builds the dataset for loading the generated videos
        :param directory: directory where the generated videos are saved
        """
        dataset_config = {
            "name": "simple_map_video_folder_dataset",
            "directory": directory,
            "frames_count": self.frames_count,
        }
        
        if self.crop_type == "short":
            frame_transform = pil_to_tensor_clip
        elif self.crop_type == "long":
            frame_transform = pil_to_tensor_clip_longcrop
        else:
            raise Exception("Invalid crop type: {}".format(self.crop_type))
        
        transforms = {
            "frame": frame_transform,
            "video": stack_video_frames
        }

        dataset = SimpleMapVideoFolderDataset(dataset_config, transforms)
        return dataset

    def get_dataloader(self, dataset: SimpleMapVideoFolderDataset) -> DataLoader:
        images_count = len(dataset)
        if self.batch_size == 1:
            images_count = images_count // dist.get_world_size() * dist.get_world_size() # when batch_size is 1, drop last and make number of batches is dividable by number of rank
            
        # Divide images into batches.
        num_batches = ((images_count - 1) // (self.batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
        all_batches = torch.arange(images_count).tensor_split(num_batches)
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=rank_batches, num_workers=self.num_workers, collate_fn=flexible_batch_elements_collate_fn)
        return dataloader

    def evaluate(self):
        # Gets the inception backbone from which to compute the features
        distributed_similarity = DistributedTextClipSimilarity({
            "visual_encoder": "ViT-L/14",
        })

        # Waits that each process loads the feature extractor
        dist.barrier()

        # Instantiates the dataloder
        dataset = self.get_dataset(self.generated_video_root)
        dataloader = self.get_dataloader(dataset)

        # Extracts statistics from each batch
        print_r0(" - Computing statistics from generated videos")
        for current_batch in tqdm.tqdm(dataloader, disable=(dist.get_rank() != 0)):
            video = current_batch.data["video"].to("cuda")
            text_embeddings = current_batch.data["text_embeddings"]

            # Accumulates the statistics
            distributed_similarity.accumulate_stats(video, text_embeddings)
            dist.barrier()

        # Gathers the final similarity score
        clip_similarity = distributed_similarity.get_statistics()

        results = {
            "text_clip_similarity": clip_similarity.item(),
        }

        dist.barrier()

        return results
