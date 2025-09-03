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

    # Take only one channel for segmentation (assuming all channels are identical)
    seg = tf.cast(seg, tf.uint8)
    seg = seg[..., 0]

    # Convert segmentation to one-hot encoding
    seg = tf.one_hot(seg, depth=10, axis=-1, dtype=tf.uint8)

    return img, seg


def random_crop_img_and_seg(img, seg, crop_size):
    combined = tf.concat([img, seg], axis=-1)
    combined_cropped = tf.image.random_crop(combined, size=[*crop_size, 13])
    return combined_cropped[:, :, :3], combined_cropped[:, :, 3:]

def central_crop(img, size):
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    top = (h - size[0]) // 2
    left = (w - size[1]) // 2
    return img[top:top+size[0], left:left+size[1]]

def make_tf_dataset(
    dir_names: List[str], random_crop=None, center_crop=None, num_classes=10
) -> tf.data.Dataset:
    img_files = []

    assert not (
        random_crop is not None and center_crop is not None
    ), "Only one of random_crop or center_crop can be specified"

    for base_dir in dir_names:
        img_files += glob(path.join(base_dir, "rgb/*.png"))

    img_files = tf.data.Dataset.from_tensor_slices(np.array(img_files).flatten())
    img_files = img_files.shuffle(buffer_size=500, reshuffle_each_iteration=False)
    dataset = img_files.map(
        lambda x: tf.py_function(func=process_path, inp=[x], Tout=(tf.uint8, tf.uint8)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if random_crop is not None:
        dataset = dataset.map(
            lambda img, seg: random_crop_img_and_seg(img, seg, random_crop),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if center_crop is not None:
        dataset = dataset.map(
            lambda img, seg: (central_crop(img, center_crop), central_crop(seg, center_crop)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset
