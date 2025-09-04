import os
from os import path
from glob import glob
import argparse
import cv2
import numpy as np
import tensorflow as tf

def predict_in_four_tiles(model: tf.keras.Model, img: np.ndarray):
    """
    Run inference on a large image by splitting it into four overlapping tiles.

    This function assumes the model has a fixed spatial input size
    (height, width, channels). It extracts the four corner crops of the
    input image, each matching the model input size, runs prediction on
    each tile, applies softmax to the logits, and then places the
    predictions back into the corresponding region of the output array.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras/TensorFlow model with a 2D image input of fixed size.
        The model must output per-pixel class scores (logits).
    img : np.ndarray
        Input image as a NumPy array of shape (H, W, C) with channel-last
        layout. The height and width must be at least as large as the model's
        required input size.

    Returns
    -------
    preds : np.ndarray
        Array of shape (H, W, num_classes) containing the per-pixel softmax
        probabilities from combining the four tiles.

    Notes
    -----
    - The four tiles are taken as the top-left, top-right, bottom-left,
      and bottom-right corners of the input image.
    - If H or W does not exactly match the tile size, the function crops
      to fit, so central regions may not be predicted.
    - For full coverage on arbitrary image sizes, a more general tiling
      strategy is required.
    """
    H, W, C = img.shape
    tile_size = model.input_shape[1:3]
    th, tw = tile_size
    preds = np.zeros((H, W, model.output_shape[-1]), dtype=np.float32)

    # Define 4 tiles (top-left, top-right, bottom-left, bottom-right)
    coords = [
        (0, 0, th, tw),               # top-left
        (0, W - tw, th, W),           # top-right
        (H - th, 0, H, tw),           # bottom-left
        (H - th, W - tw, H, W)        # bottom-right
    ]

    for (y0, x0, y1, x1) in coords:
        tile = img[y0:y1, x0:x1, :]                  # crop
        tile_batch = np.expand_dims(tile, axis=0)    # (1,H,W,C)
        pred_tile = model.predict(tile_batch)        # (1,h,w,num_classes)
        pred_tile_soft = tf.nn.softmax(pred_tile, axis=-1).numpy()
        preds[y0:y1, x0:x1, :] += pred_tile_soft[0]

    return preds

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Inference using trained models')
    parser.add_argument('-r','--data-root', help='Path to the root of the inference dataset', required=True)
    parser.add_argument('-m','--model-paths', help='Paths to one or more trained model (.h5)', nargs="+", required=True)
    args = parser.parse_args()
    print(args.data_root, args.model_paths)
    
    # Look for subdirs named "rgb"
    image_dirs = sorted(glob(path.join(args.data_root, "**/rgb/"), recursive=True))
    print(image_dirs)
    
    models = [tf.keras.models.load_model(model_path) for model_path in args.model_paths]
    for image_dir in image_dirs:
        # Create the predictions folder adjacent to the rgb folder if not exists
        pred_dir = path.join(image_dir, "..", "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        image_paths = sorted(glob(path.join(image_dir, "*.png")))
        
        for image_path in image_paths:
            # Read the image to numpy
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(img.shape)
            
            # Reset the prediction for each new image
            img_pred = None
            for model in models:
                # Predict the image in four tiles, using the model's input shape
                pred = predict_in_four_tiles(model, img)
                print(pred.shape, np.amax(pred))
            
                # If this is the first prediction, use it as the base, otherwise add to it
                if img_pred is None:
                    img_pred = pred
                else:
                    img_pred += pred
            
            # Perform argmax only after all softmax predictions from models and tiles have
            # been combined, for an ensembled segmentation
            img_pred = np.argmax(img_pred, axis=-1)
            print(pred_dir)
            
            # Export the segmentation to PNG
            cv2.imwrite(path.join(pred_dir, path.basename(image_path)), img_pred.astype(np.uint8))
        
