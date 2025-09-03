from os import path
from glob import glob
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

from data import make_tf_dataset
from model import build_model
from viz import plot_sample

# Define model input shape
IN_SHAPE = 1024, 1024, 3

# Define output classes
OUT_CLASSES = {
    0: "Background",
    1: "Tool clasper",
    2: "Tool wrist",
    3: "Tool shaft",
    4: "Suturing needle",
    5: "Thread",
    6: "Suction tool",
    7: "Needle Holder",
    8: "Clamps",
    9: "Catheter",
}

# Define colors to use in visualizations
COLORS = [
    (0, 0, 0),       # Background - Black
    (255, 0, 0),     # Tool clasper - Blue
    (0, 255, 0),     # Tool wrist - Green
    (0, 0, 255),     # Tool shaft - Red
    (255, 255, 0),   # Suturing needle - Cyan
    (255, 0, 255),   # Thread - Magenta
    (0, 255, 255),   # Suction tool - Yellow
    (128, 0, 128),   # Needle Holder - Purple
    (128, 128, 0),   # Clamps - Olive
    (0, 128, 128),   # Catheter - Teal
]

# Location of the data
DATA_ROOT = "/scratch/p286425/challenge/data/code_data/"

# Location to output prediction and data samples
SAMPLE_DIR = "samples/"

# Location to store model output
MODEL_DIR = "models/"

# Number of epochs to train
NUM_EPOCHS = 150

# Batch size for training
TRAIN_BATCH_SIZE = 10

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Model training')
    parser.add_argument('-r','--data-root', help='Path to the root of the training dataset', required=True)
    parser.add_argument('-w','--work-dir', help='Path to the work dir for the training run', required=True)
    args = parser.parse_args()
    print(args.data_root, args.work_dir)
    
    # Create the work dir if it does not exist
    sample_dir = path.join(args.work_dir, SAMPLE_DIR)
    model_dir = path.join(args.work_dir, MODEL_DIR)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Get all training directories
    all_train_dirs = sorted(glob(path.join(args.data_root, "video_*/")))
    
    # Shuffle the training directories
    rng = np.random.default_rng(12345)
    rng.shuffle(all_train_dirs)
    
    # Determine how many videos to use for training and validation
    num_train_dirs = round(len(all_train_dirs) * 0.9)
    num_val_dirs = len(all_train_dirs) - num_train_dirs
    print(f"Using {num_train_dirs} directories for train and {num_val_dirs} for val")
    
    # Split videos into training and validation
    train_dirs = all_train_dirs[: num_train_dirs]
    valid_dirs = all_train_dirs[num_train_dirs :]
    print("Train dirs:", train_dirs)
    print("Val dirs:", valid_dirs)

    # Create tf.data dataset for the training data
    # Shuffle all filenames to make sure varied data is shown during training
    train_dataset = make_tf_dataset(train_dirs, random_crop=IN_SHAPE[:2], shuffle_before_load=True)
    train_dataset = train_dataset.batch(TRAIN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.repeat()

    # Make a static validation set (non repeating)
    valid_dataset = make_tf_dataset(valid_dirs, center_crop=IN_SHAPE[:2])
    valid_dataset = valid_dataset.batch(1).prefetch(tf.data.AUTOTUNE)

    # Extract a smaller set to inspect predictions during training
    sample_set = valid_dataset.take(10)
    
    # Take the first 10 batches as a list of tensors
    sample_set_list = list(sample_set)

    # If dataset yields (x, y), extract only x
    sample_inputs = [x for x, y in sample_set_list]
    
    # Concatenate along batch axis
    sample_inputs = tf.concat(sample_inputs, axis=0)
    
    # Export images and segmentations once
    for i, (imgs, segs) in enumerate(sample_set):
        os.makedirs(sample_dir, exist_ok=True)
        
        # Debugging
        print(f"Sample {i:02d}:")
        print(imgs.numpy().shape)
        print(segs.numpy().shape)

        # Export a few samples to the workdir
        img = imgs.numpy().astype(np.uint8)
        seg = segs.numpy().astype(np.uint8)  # Scale segmentation for visibility

        print(img.shape, seg.shape)

        # Remove batch dimension and export as PNG
        img = img[0]
        seg = seg[0]
        
        # Argmax the segmentation 
        seg = np.argmax(seg, axis=-1)

        # Export the segmentation as a color image
        # Map each class to a color for better visualization
        seg_color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for cls, color in enumerate(COLORS):
            seg_color[seg == cls] = color
        
        # Export as PNG
        cv2.imwrite(f"{sample_dir}/sample_{i:02d}_img.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{sample_dir}/sample_{i:02d}_seg.png", cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
        plot_sample(img, seg, f"{sample_dir}/sample_{i:02d}_plot.png")

    # Build the U-Net model and show summary
    m = build_model(IN_SHAPE, len(OUT_CLASSES))
    m.summary()
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Define variables to keep track of best epoch and loss
    best_loss = 999999.
    best_epoch = -1
    
    # Start training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Fit the model for one training epoch
        m.fit(train_dataset, epochs=1, steps_per_epoch=200)
        
        # Predict on the smaller subset and export the predicted segmentations
        preds = m.predict(sample_inputs)
        int_preds = np.argmax(preds, axis=-1).astype(np.uint8)
        print("Prediction shape and max predicted class:", int_preds.shape, np.amax(int_preds))

        # Iterate over the prediction array
        for i, p in enumerate(int_preds):

            # Export the segmentation as a color image
            # Map each class to a color for better visualization
            pred_color = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
            
            for cls, color in enumerate(COLORS):
                pred_color[p == cls] = color
            
            # Export as PNG
            cv2.imwrite(f"{sample_dir}/sample_{i:02d}_pred.png", cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
        
        # Evaluate the performance on the full validation set to gauge performance
        loss, accuracy = m.evaluate(valid_dataset)
        
        print(f"Validation loss: {loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")
        
        # If the validation loss improves, store the model as best_loss.h5
        if loss < best_loss:
            print(f"[I] Validation loss improved from {best_loss:.4f} (epoch {best_epoch}) to {loss:.4f} (epoch {epoch})")
            best_epoch = epoch
            best_loss = loss
            print(f"Saving to {model_dir}/best_loss.h5")
            m.save(f"{model_dir}/best_loss.h5")
        
        # Export the latest available model
        print(f"Saving to {model_dir}/latest.h5")
        m.save(f"{model_dir}/latest.h5")