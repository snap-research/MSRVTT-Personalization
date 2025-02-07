from typing import Dict, Tuple, List, Any, Union
from einops import rearrange
from PIL import Image
import sys
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed as dist
import torchvision.transforms as transforms
import clip
from sklearn.metrics.pairwise import cosine_similarity

from utils.tensors.tensor_folder import TensorFolder

sys.path.append("../models/Grounded-Segment-Anything")
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def load_image_grounding(image: np.ndarray):
    image = Image.fromarray(image)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image, None)
    return image

def load_subject_image_dino(image: np.ndarray, box_xyxy: np.ndarray, mask: np.ndarray):
    # image cropping
    cropped_image = image[
        box_xyxy[1] : box_xyxy[3],
        box_xyxy[0] : box_xyxy[2],
    ]

    # image masking
    mask = mask[..., None]
    masked_image = mask * cropped_image + \
                   (1-mask) * np.ones_like(cropped_image, dtype=np.uint8) * 255
    masked_image = Image.fromarray(masked_image.astype(np.uint8))

    # pad to square
    width, height = masked_image.size
    if width == height:
        padded_image = masked_image
    elif width > height:
        padded_image = Image.new(masked_image.mode, (width, width), color=(255, 255, 255))
        padded_image.paste(masked_image, (0, (width - height) // 2))
    else:
        padded_image = Image.new(masked_image.mode, (height, height), color=(255, 255, 255))
        padded_image.paste(masked_image, ((height - width) // 2, 0))

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transformed_image = transform(padded_image)

    return transformed_image

def load_grounding_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args).to(device)
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def find_sublist_indices(main_list, sublist):
    sublist_length = len(sublist)
    mask = [False] * len(main_list)
    
    for i in range(len(main_list) - sublist_length + 1):
        if main_list[i:i+sublist_length] == sublist:
            mask[i:i+sublist_length] = [True] * sublist_length
    
    return mask

def get_grounding_output(model, video_frames, word_tags, box_threshold, device="cpu"):
    _, H, W, _ = video_frames.shape # (frames_count, height, width, 3)

    # prepare images
    video_frame_batch = [load_image_grounding(video_frame).to(device) for video_frame in video_frames]
    video_frame_batch = torch.stack(video_frame_batch)

    # prepare prompts
    prompt = " . ".join(word_tags)
    prompt = prompt.lower().strip()
    if not prompt.endswith("."):
        prompt += "."
    prompt_batch = [prompt] * len(video_frames)

    # prepare mask for each word_tag
    tokenlizer = model.tokenizer
    prompt_token_ids = tokenlizer(prompt)["input_ids"]

    word_tag_token_mask = {}
    max_seq_length = 256
    for word_tag in word_tags:
        word_tag_token_ids = tokenlizer(word_tag)["input_ids"][1:-1] # remove <bos> and <eos>
        token_mask = find_sublist_indices(prompt_token_ids, word_tag_token_ids)
        token_mask = torch.tensor(token_mask + [False] * (max_seq_length - len(token_mask)))
        word_tag_token_mask[word_tag] = token_mask

    # model inference
    with torch.no_grad():
        outputs = model(video_frame_batch, captions=prompt_batch)
    
    # parse results
    word_tag_box_xyxy_score_batch = []
    for i in range(len(video_frames)):
        logits = outputs["pred_logits"].cpu().sigmoid()[i]  # (num_query: 900, max_seq_length: 256)
        boxes = outputs["pred_boxes"].cpu()[i]  # (num_query: 900, xyxy: 4)
        num_query, max_seq_length = logits.shape

        # compute the score for each (box, word_tag) pair
        score_box_word_tag = []
        for word_tag, token_mask in word_tag_token_mask.items():
            masked_logits = logits * token_mask
            score_max = torch.max(masked_logits, axis=1).values # (num_query)
            score_avg = torch.sum(masked_logits, axis=1) / torch.sum(token_mask) # (num_query)
            score_box_word_tag.append((score_max + score_avg) / 2)
        score_box_word_tag = torch.stack(score_box_word_tag).T # (num_query, num_word_tag)

        # for each box, find the corresponding word tag 
        map_box_word_tag = torch.max(score_box_word_tag, axis=1).indices # (num_query)

        # for each word_tag, use the box with highest score as to represent this word_tag
        word_tag_box_id_score = {word_tag : (None, 0) for word_tag in word_tags}
        for box_id, word_tag_id in enumerate(map_box_word_tag):
            word_tag = word_tags[word_tag_id]
            score = score_box_word_tag[box_id, word_tag_id]
            if word_tag_box_id_score[word_tag][1] < score:
                word_tag_box_id_score[word_tag] = (box_id, score)

        # read and process box_xyxy
        word_tag_box_xyxy_score = {}
        for word_tag, (box_id, score) in word_tag_box_id_score.items():
            if box_id != None and score > box_threshold:
                box_xyxy = boxes[box_id]
                box_xyxy = box_xyxy * torch.Tensor([W, H, W, H])
                box_xyxy[:2] -= box_xyxy[2:] / 2
                box_xyxy[2:] += box_xyxy[:2]
                word_tag_box_xyxy_score[word_tag] = (box_xyxy.cpu().int().tolist(), float(score))

        word_tag_box_xyxy_score_batch.append(word_tag_box_xyxy_score)

    return word_tag_box_xyxy_score_batch

def get_sam_output(sam_predictor, image, word_tag_box_xyxy_score, mask_threshold, device="cpu"):
    sam_predictor.set_image(image)

    boxes_xyxy = [box_xyxy for box_xyxy, _ in word_tag_box_xyxy_score.values()]
    boxes_xyxy = torch.tensor(boxes_xyxy)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(device)

    masks, scores, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    word_tag_mask_score = {}
    for i, word_tag in enumerate(word_tag_box_xyxy_score.keys()):
        mask, score = masks[i], scores[i]
        if score > mask_threshold:
            word_tag_mask_score[word_tag] = (mask.cpu().numpy().squeeze().astype(bool), float(score))

    return word_tag_mask_score


class DistributedSubjectDinoSimilarity(nn.Module):
    """
    Class for distributed Subject DINO similarity computation
    """
    def __init__(self, grounding_config: Dict, sam_config: Dict, dino_config: Dict):
        super().__init__()

        # Loads the pretrained grounding model
        config = grounding_config["config"]
        checkpoint = grounding_config["checkpoint"]
        self.grounding_model = load_grounding_model(config, checkpoint, device="cuda")
        self.box_threshold = grounding_config["box_threshold"]

        # Loads the pretrained SAM model
        network = sam_config["network"]
        checkpoint = sam_config["checkpoint"]
        self.sam_predictor = SamPredictor(sam_model_registry[network](checkpoint=checkpoint).to("cuda"))
        self.mask_threshold = sam_config["mask_threshold"]

        # Loads the pretrained DINO model
        visual_encoder_type = dino_config["visual_encoder"]
        self.dino_model = torch.hub.load("facebookresearch/dino:main", visual_encoder_type).to("cuda")
        
        # Instantiates the mu and sigma tensors
        self.similarity_sum = torch.zeros((), dtype=torch.float64, device="cuda")
        self.accumulated_elements = 0

        # Waits that each process loads the feature extractor
        dist.barrier()

    def accumulate_stats(self, video: np.ndarray, target_subject_embeddings: Dict):
        """
        Accumulates the statistics
        :param video: (batch_size, frames_count, height, width, 3) numpy array with videos.
        :param target_subject_embeddings: Dictionay with the key of the subject word tag and the value of CLIP embeddings of the subject images
        """

        # only support batch_size 1
        assert video.shape[0] == 1
        video = video[0]
        word_tags = list(target_subject_embeddings.keys())
        
        # Initialize the subject count and the scores of cosine similarity and recall rate
        current_similarity_sum = 0
        current_target_subject_count = video.shape[0] * len(word_tags) # each video frame should have "len(word_tags)" subject
        
        # Runs the grounding model
        word_tag_box_xyxy_score_batch = get_grounding_output(self.grounding_model, video, word_tags, self.box_threshold, device="cuda")

        # Runs the SAM model
        word_tag_mask_score_batch = []
        for video_frame, word_tag_box_xyxy_score in zip(video, word_tag_box_xyxy_score_batch):
            if len(word_tag_box_xyxy_score) == 0:
                word_tag_mask_score_batch.append({})
            else:
                word_tag_mask_score_batch.append(
                    get_sam_output(self.sam_predictor, video_frame, word_tag_box_xyxy_score, self.mask_threshold, device="cuda")
                )
                
        # Compute the DINO similarity
        for i, video_frame in enumerate(video):
            word_tag_box_xyxy_score = word_tag_box_xyxy_score_batch[i]
            word_tag_mask_score = word_tag_mask_score_batch[i]
            
            # Compute the metric for each subject
            for word_tag in word_tags:                       
                # Cannot detect the subject from the grounding or SAM model
                if not (word_tag in word_tag_box_xyxy_score and word_tag in word_tag_mask_score):
                    continue
                
                # Parse box and mask infornmation
                box_xyxy = word_tag_box_xyxy_score[word_tag][0]
                box_h = box_xyxy[3] - box_xyxy[1]
                box_w = box_xyxy[2] - box_xyxy[0]
                
                mask = word_tag_mask_score[word_tag][0]
                mask = mask[box_xyxy[1]:box_xyxy[3], box_xyxy[0]:box_xyxy[2]]
                mask_h, mask_w = mask.shape

                # Invalid detection results from the grounding or SAM model (the mask is sometimes smaller than the box on one side. It is acceptable if the difference is within 3 pixels)
                if not ((mask_h == box_h and mask_w >= box_w-3 and mask_w <= box_w) or \
                        (mask_w == box_w and mask_h >= box_h-3 and mask_h <= box_h)):
                    continue
                
                # Prepare the subject image
                subject_image = load_subject_image_dino(video_frame, box_xyxy, mask)
                subject_image = subject_image.unsqueeze(0).to("cuda")

                # Extract the DINO embeddings
                with torch.no_grad():
                    subject_embeddings = self.dino_model(subject_image)
            
                # Compute cosine similarity
                subject_embeddings = subject_embeddings.cpu()
                current_target_subject_embeddings = target_subject_embeddings[word_tag][0]
                similarity = cosine_similarity(
                    subject_embeddings.numpy(),
                    current_target_subject_embeddings.numpy()
                ).mean()
                current_similarity_sum = current_similarity_sum + similarity
            
        self.similarity_sum = self.similarity_sum + current_similarity_sum / current_target_subject_count
        
        # Updates the number of accumulated elements (only have 1 video each time)
        elements_count = 1
        elements_count = torch.as_tensor(elements_count, dtype=torch.int64, device="cuda")
        # Uses distirbuted to get the current images from all processes
        torch.distributed.all_reduce(elements_count)
        self.accumulated_elements += int(elements_count.item())

    def get_statistics(self) -> torch.Tensor:
        """
        Performs computation of mu and sigma based on the accumulated statistics
        :return: () scalar tensor with the average clip similarity and recall rate
        """

        similarity_sum = self.similarity_sum
        torch.distributed.all_reduce(similarity_sum)
        
        # Computes mean and covariance estimates
        similarity_sum = similarity_sum / self.accumulated_elements

        return similarity_sum
