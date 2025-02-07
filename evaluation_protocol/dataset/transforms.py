from typing import List
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def identity(x):
    return x

def pil_maybe_remove_transparency(image):
    image = image.convert("RGB")
    return image

def pil_to_numpy(pil_image: Image.Image):
    pil_image = pil_maybe_remove_transparency(pil_image)
    return np.array(pil_image)

def pil_to_numpy_chw(pil_image: Image.Image):
    numpy_image = pil_to_numpy(pil_image)
    if numpy_image.ndim == 2:
        numpy_image = numpy_image[:, :, np.newaxis]
    numpy_image = numpy_image.transpose(2, 0, 1)
    return numpy_image

def numpy_to_tensor(array: np.ndarray):
    return torch.from_numpy(array)

def pil_to_tensor(pil_image: Image.Image):
    numpy_image = pil_to_numpy(pil_image)
    return numpy_to_tensor(numpy_image)

def pil_to_tensor_chw(pil_image: Image.Image):
    numpy_image = pil_to_numpy_chw(pil_image)
    return numpy_to_tensor(numpy_image)

def pil_to_tensor_clip(pil_image: Image.Image):
    pil_image = pil_maybe_remove_transparency(pil_image)

    width, height = pil_image.size
    square_size = min(width, height)
    pil_image = TF.center_crop(pil_image, (square_size, square_size))

    # See https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L79
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), # Deforms the image to make it fit in the expected size
        transforms.CenterCrop(size=(224, 224)),
        pil_maybe_remove_transparency,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    transformed_image = transform(pil_image)

    return transformed_image

def pil_to_tensor_clip_longcrop(pil_image: Image.Image):
    """
    Transforms a PIL image into a tensor that can be used as input for the clip model
    Does not perform center crop, rather adds external black bars
    """
    pil_image = pil_maybe_remove_transparency(pil_image)

    width, height = pil_image.size
    square_size = max(width, height)
    pil_image = TF.center_crop(pil_image, (square_size, square_size))

    # See https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L79
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), # Deforms the image to make it fit in the expected size
        transforms.CenterCrop(size=(224, 224)),
        pil_maybe_remove_transparency,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    transformed_image = transform(pil_image)

    return transformed_image

def stack_video_frames(frames: List[torch.Tensor], *args, **kwargs):
    """
    Stacks the given frames into a single tensor
    :param frames: list of frames to stack
    :return: (frames_count, channels, height, width) tensor with the stacked frames
    """
    if torch.is_tensor(frames[0]):
        return torch.stack(frames, dim=0)
    else:
        return np.stack(frames, axis=0)