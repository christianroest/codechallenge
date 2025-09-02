from os import path
from glob import glob
from typing import List

import cv2
import numpy as np


def find_image_paths(base_dir: str) -> List[str]:
    """
    Finds all image file paths with a given extension in the specified base directory and its subdirectories.

    Args:
        base_dir (str): The base directory to search for image files.

    Returns:
        list: A list of file paths matching the specified extension within the directory structure.
    """
    # Find all images in the directory and its subdirectories
    matches = sorted(glob(path.join(base_dir, "**", "rgb", f"*.png"), recursive=True))

    return matches


def find_segmentation_paths(base_dir: str) -> List[str]:
    """
    Finds all segmentation file paths with a given extension in the specified base directory and its subdirectories.

    Args:
        base_dir (str): The base directory to search for segmentation files.

    Returns:
        list: A list of file paths matching the specified extension within the directory structure.
    """
    # Find all segmentations in the directory and its subdirectories
    matches = sorted(glob(path.join(base_dir, "**", "segmentation", f"*.png"), recursive=True))

    return matches


def read_images(image_paths: List[str]) -> np.ndarray:
    """
    Reads images from the provided file paths and stacks them into a single numpy array.

    Args:
        image_paths (list): A list of file paths to the images.

    Returns:
        numpy.ndarray: A numpy array containing all the images stacked along a new axis.
    """

    images = [cv2.imread(str(p)) for p in image_paths]
    images = np.stack(images, axis=0)
    return images
