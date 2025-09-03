import os
from os import path
from glob import glob
import argparse
import cv2
import numpy as np
import tensorflow as tf

def predict_in_four_tiles(model, img):
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
    
    for image_dir in image_dirs:
        pred_dir = path.join(image_dir, "..", "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        image_paths = sorted(glob(path.join(image_dir, "*.png")))
        for image_path in image_paths:
            # Read the image to numpy
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(img.shape)
            
            img_pred = None
            
            for model_path in args.model_paths:
                model = tf.keras.models.load_model(model_path)
                model.summary()
                
                pred = predict_in_four_tiles(model, img)
                print(pred.shape, np.amax(pred))
            
                if img_pred is None:
                    img_pred = pred
                else:
                    img_pred += pred
                
                del model
            img_pred = np.argmax(img_pred, axis=-1)
            print(pred_dir)
            cv2.imwrite(path.join(pred_dir, path.basename(image_path)), img_pred.astype(np.uint8))
        