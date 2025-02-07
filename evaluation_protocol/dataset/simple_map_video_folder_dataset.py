import bisect
import collections
import time
import os
from typing import Dict
from pathlib import Path

import numpy as np
import torch
import json
import gzip
import pickle
from PIL import Image

from dataset.batching import BatchElement, flexible_batch_elements_collate_fn

from dataset.transforms import identity
from utils.video.video_decoder import VideoDecoder
from utils.video.video_utils import find_video_files


class SimpleMapVideoFolderDataset(torch.utils.data.Dataset):
    """
    Class representing a map-style dataset of videos contained in a folder.
    Assumes all videos have a constant length in frames
    """
    def __init__(self, dataset_config: Dict, transforms: Dict):
        """
        Initializes the dataset
        :param dataset_config: configuration of the dataset
        :param transforms: transforms to apply to the data
        """
        self.dataset_config = dataset_config
        self.transforms = transforms
        
        self.name = dataset_config["name"]
        self.directory = dataset_config["directory"]
        self.frames_count = dataset_config["frames_count"]

        self.video_paths = list(sorted(find_video_files(self.directory)))

        # Transformation for when the batch element is constructed
        self.video_transform = transforms.get("video", identity)
        self.frame_transform = transforms.get("frame", identity)

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        :return: length of the dataset
        """
        return len(self.video_paths)

    def __getitem__(self, dataset_element_id: int) -> BatchElement:
        """
        Gets an item from the dataset
        :param dataset_element_id: the id of the element to get
        :return: Batch element with the key "image" containing the image and "metadata" containing a possibly empty metadata dictionary
        """
        current_video_path = self.video_paths[dataset_element_id]
        video_decoder = VideoDecoder(current_video_path)
        decoded_frames = video_decoder.decode_frames_at_indexes(list(range(self.frames_count)))
        transformed_frames = [self.frame_transform(frame) for frame in decoded_frames]
        transformed_video = self.video_transform(transformed_frames)

        metadata_path = os.path.splitext(current_video_path)[0] + ".metadata.json"
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)
        else:
            metadata = {}

        text_embeddings_path = os.path.splitext(current_video_path)[0] + ".text_embeddings.pt"
        if os.path.isfile(text_embeddings_path):
            text_embeddings = torch.load(text_embeddings_path)
        else:
            text_embeddings = torch.zeros((0, 512), dtype=torch.float16)
            
        video_embeddings_path = os.path.splitext(current_video_path)[0] + ".video_embeddings.pt"
        if os.path.isfile(video_embeddings_path):
            video_embeddings = torch.load(video_embeddings_path)
        else:
            video_embeddings = torch.zeros((0, 512), dtype=torch.float16)
    
        subject_embeddings_path = os.path.splitext(current_video_path)[0] + ".subject_embeddings.pkl.gz"
        if os.path.isfile(subject_embeddings_path):
            subject_embeddings = pickle.load(gzip.open(subject_embeddings_path, "rb"))
        else:
            subject_embeddings = {}
            
        face_embeddings_path = os.path.splitext(current_video_path)[0] + ".face_embeddings.pt"
        if os.path.isfile(face_embeddings_path):
            face_embeddings = torch.load(face_embeddings_path)
        else:
            face_embeddings = torch.zeros((0, 512), dtype=torch.float32)

        data = {}
        # Adds additional meta-information to the dataset
        data["video"] = transformed_video
        data["video_framerate"] = video_decoder.framerate
        data["metadata"] = metadata
        data["text_embeddings"] = text_embeddings
        data["video_embeddings"] = video_embeddings
        data["subject_embeddings"] = subject_embeddings
        data["face_embeddings"] = face_embeddings

        # Creates the batch element and transforms it
        batch_transforms = {}
        batch_element = BatchElement(data, batch_transforms)

        return batch_element