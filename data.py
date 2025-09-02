from os import path
from glob import glob
from typing import List, Tuple

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
    matches = sorted(
        glob(path.join(base_dir, "**", "segmentation", f"*.png"), recursive=True)
    )

    return matches


def read_images_to_numpy(image_paths: List[str]) -> np.ndarray:
    """
    Reads images from the provided file paths and stacks them into a single numpy array.

    Args:
        image_paths (list): A list of file paths to the images.

    Returns:
        numpy.ndarray: A numpy array containing all the images stacked along a new axis.
    """

    images = []
    for i, p in enumerate(image_paths):
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Image at path {p} could not be read.")
        images.append(img)
        print(i, "/", len(image_paths))
    images = np.stack(images, axis=0)
    return images


def load_dataset(base_dir: str, limit: int = 99999999) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a dataset of images and their corresponding segmentations from subdirectories within a base directory.
    The function searches for subdirectories matching the pattern "video_*/" inside the specified base directory.
    For each subdirectory, it attempts to find image and segmentation files, reads them into NumPy arrays.

    Args:
        base_dir (str): The base directory containing subdirectories with image and segmentation data.
        limit (int, optional): The maximum number of images and segmentations to load from each subdirectory, for testing.
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - images: Array of loaded images.
            - segmentations: Array of loaded segmentation masks.
    """

    # Look for subdirectories containing the data
    subdirs = sorted(glob(path.join(base_dir, "video_*/")))

    for subdir in subdirs:
        print(f"Loading data from {subdir}...")
        try:
            image_paths = find_image_paths(subdir)[:limit]
            segmentation_paths = find_segmentation_paths(subdir)[:limit]

            images = read_images_to_numpy(image_paths)
            segmentations = read_images_to_numpy(segmentation_paths)

            print(
                f"Loaded {images.shape[0]} images and {segmentations.shape[0]} segmentations from {subdir}."
            )

        except Exception as e:
            print(f"Error loading data from {subdir}: {e}, skipping this directory.")
            continue

    return images, segmentations
