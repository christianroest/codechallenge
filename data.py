from os import path
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3)

    # Resize the image to the desired size
    return img


def process_path(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given a file path, reads the image and its corresponding segmentation mask,
    decodes them, and returns them as tensors.
    Args:
        file_path (str): Path to the image file.
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the image tensor and the segmentation mask
        tensor.
    """
    # Convert the file path to a string
    file_path = file_path.numpy().decode("utf-8")

    # Derive the segmentation file path from the image file path
    seg_path = file_path.replace("/rgb/", "/segmentation/")

    # Read and decode the image
    img_contents = tf.io.read_file(file_path)
    img = tf.io.decode_png(img_contents, channels=3)

    # Read and decode the segmentation mask
    seg_contents = tf.io.read_file(seg_path)
    seg = tf.io.decode_png(seg_contents, channels=3)

    return img, seg


def make_tf_dataset(base_dir: str) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset from image files located in the specified base directory.

    Args:
        base_dir (str): The base directory containing image files.

    Returns:
        tf.data.Dataset: A TensorFlow dataset containing the image file paths.
    """

    img_files = tf.data.Dataset.list_files(
        path.join(base_dir, "video_01/rgb/*.png"), shuffle=False
    )
    img_files = img_files.shuffle(buffer_size=5000, reshuffle_each_iteration=False)
    dataset = img_files.map(
        lambda x: tf.py_function(
            func=process_path, inp=[x], Tout=(tf.uint8, tf.uint8)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    
    return dataset
