import glob
import os
from typing import List

def find_video_files(video_directory: str, extensions=["mp4", "mkv", "webm", "avi", "webp", "mov"]) -> List[str]:
    """
    Gets all the videos in a directory
    :param video_directory:
    :param extensions:
    :return:
    """
    video_files = glob.glob(os.path.join(video_directory, "*"))

    # Filters the files
    selected_files = []
    for video_file in video_files:
        if is_video_file(video_file, extensions):
            selected_files.append(video_file)

    return selected_files

def is_video_file(filename: str, extensions=["mp4", "mkv", "webm", "avi", "webp", "mov"]) -> bool:
    """
    Checks if the file is a video file
    :param filename:
    :param extensions:
    :return:
    """
    for extension in extensions:
        if filename.endswith("." + extension):
            return True

    return False
