from typing import Dict, Tuple, List, Any, Union
from einops import rearrange

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as dist
import clip
from sklearn.metrics.pairwise import cosine_similarity

from utils.tensors.tensor_folder import TensorFolder

class DistributedVideoClipSimilarity(nn.Module):
    """
    Class for distributed Video CLIP similarity computation
    """
    def __init__(self, clip_config: Dict):
        super().__init__()
        
        # Gets the type of visual encoder to use
        visual_encoder_type = clip_config.get("visual_encoder", "ViT-L/14")

        # Loads the pretrained CLIP model
        self.clip_model, _ = clip.load(visual_encoder_type)
        self.clip_model.to("cuda")

        # Instantiates the mu and sigma tensors
        self.similarity_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        self.accumulated_elements = 0

        # Waits that each process loads the feature extractor
        dist.barrier()

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Computes the video embeddings for the given input
        :param video: (batch_size, frames_count, 3, height, width) tensor with videos to embed
        :return: (batch_size, frames_count, clip_features_count) tensor with embedded videos
        """
        with torch.no_grad():
            flat_images, initial_dimensions = TensorFolder.flatten(video, -3)
            flat_clip_embedded_images = self.clip_model.encode_image(flat_images)
            folded_clip_embedded_images = TensorFolder.fold(flat_clip_embedded_images, initial_dimensions)

        return folded_clip_embedded_images

    def accumulate_stats(self, video: torch.Tensor, target_video_embeddings: torch.Tensor):
        """
        Accumulates the statistics
        :param video: (batch_size, frames_count, 3, height, width) tensor with videos.
        :param target_video_embeddings: (batch_size, frames_count, clip_features_count) tensor with ground truth video embeddings.
        """
        # Extract the CLIP embeddings
        video_embeddings = self.encode_video(video).cpu()

        # Computes the cosine similarity score
        current_similarity_sum = 0
        for current_video_embeddings, current_target_video_embeddings in zip(video_embeddings, target_video_embeddings):
            similarity = cosine_similarity(
                current_video_embeddings.numpy(),
                current_target_video_embeddings.numpy()
            ).mean()
            current_similarity_sum = current_similarity_sum + similarity

        # Updates the total similarity
        self.similarity_sum = self.similarity_sum + current_similarity_sum

        # Updates the number of accumulated elements
        elements_count = video.shape[0]
        elements_count = torch.as_tensor(elements_count, dtype=torch.int64, device="cuda")
        # Uses distirbuted to get the current images from all processes
        torch.distributed.all_reduce(elements_count)
        self.accumulated_elements += int(elements_count.item())

    def get_statistics(self) -> torch.Tensor:
        """
        Performs computation of mu and sigma based on the accumulated statistics
        :return: () scalar tensor with the average cosine similarity
        """
        similarity_sum = self.similarity_sum
        torch.distributed.all_reduce(similarity_sum)
        
        # Computes mean and covariance estimates
        similarity_sum = similarity_sum / self.accumulated_elements

        return similarity_sum
