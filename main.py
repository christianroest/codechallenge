from os import path
from glob import glob
import os
import cv2
import numpy as np
import tensorflow as tf

from data import make_tf_dataset
from model import build_model
from viz import plot_sample
from loss import CategoricalFocalCrossentropy, CategoricalDiceLoss

in_shape = 1024, 1024, 3
out_classes = {
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
data_root = "/scratch/p286425/challenge/data/code_data/"
SAMPLE_DIR = "./samples/"
NUM_EPOCHS = 25
TRAIN_BATCH_SIZE = 10

if __name__ == "__main__":

    # Get all training directories
    all_train_dirs = sorted(glob(path.join(data_root, "video_*/")))
    rng = np.random.default_rng(12345)
    rng.shuffle(all_train_dirs)
    num_train_dirs = round(len(all_train_dirs) * 0.8)
    num_val_dirs = len(all_train_dirs) - num_train_dirs
    print(f"Using {num_train_dirs} directories for train and {num_val_dirs} for val")

    train_dirs = all_train_dirs[: int(0.9 * len(all_train_dirs))]
    valid_dirs = all_train_dirs[int(0.9 * len(all_train_dirs)) :]
    print("Train dirs:", train_dirs)
    print("Val dirs:", valid_dirs)

    # First create the training and validation datasets
    train_dataset = make_tf_dataset(train_dirs, random_crop=in_shape[:2])
    train_dataset = train_dataset.batch(TRAIN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.repeat()

    valid_dataset = make_tf_dataset(valid_dirs, center_crop=in_shape[:2])
    valid_dataset = valid_dataset.batch(1).prefetch(tf.data.AUTOTUNE)

    sample_set = valid_dataset.take(10)
    
    # Take the first 10 batches as a list of tensors
    sample_set_list = list(sample_set)

    # If dataset yields (x, y), extract only x
    sample_inputs = [x for x, y in sample_set_list]
    
    # Concatenate along batch axis
    sample_inputs = tf.concat(sample_inputs, axis=0)

    for i, (imgs, segs) in enumerate(sample_set):
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        
        print(f"Sample {i:02d}:")
        print(imgs.numpy().shape)
        print(segs.numpy().shape)

        # Export a few samples to the workdir
        img = imgs.numpy().astype(np.uint8)
        seg = segs.numpy().astype(np.uint8)  # Scale segmentation for visibility

        print(img.shape, seg.shape)

        img = img[0]  # Remove batch dim
        seg = seg[0] # Remove batch dim
        seg = np.argmax(seg, axis=-1)  # Convert one-hot to int
        print("SASDADS", seg.shape)
        cv2.imwrite(f"{SAMPLE_DIR}/sample_img_{i:02d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Export the segmentation as a color image
        # Map each class to a color for better visualization
        seg_color = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        colors = [
            (0, 0, 0),        # Background - Black
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
        for cls, color in enumerate(colors):
            seg_color[seg == cls] = color
        
        cv2.imwrite(f"{SAMPLE_DIR}/sample_seg_{i:02d}.png", cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
        plot_sample(img, seg, f"{SAMPLE_DIR}/sample_plot_{i:02d}.png")

    # Build the model
    m = build_model(in_shape, len(out_classes))
    m.summary()
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        # loss=SparseDiceLoss(num_classes=len(out_classes)),
        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        # loss=CategoricalFocalCrossentropy(from_logits=True),
        loss=CategoricalDiceLoss(from_logits=True),
        metrics=["accuracy"],
    )

    print(train_dataset.take(1))

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        m.fit(train_dataset, epochs=1, steps_per_epoch=100)
        preds = m.predict(sample_inputs)  # Run a prediction to test the model
        int_preds = np.argmax(preds, axis=-1).astype(np.uint8)
        print("INT_PREDS:", int_preds.shape, np.amax(int_preds))

        # Export the predictions
        for i, p in enumerate(int_preds):

            # Export the segmentation as a color image
            # Map each class to a color for better visualization
            pred_color = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
            colors = [
                (0, 0, 0),        # Background - Black
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
            for cls, color in enumerate(colors):
                pred_color[p == cls] = color
            
            cv2.imwrite(f"{SAMPLE_DIR}/sample_pred_{i:02d}.png", cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

        loss, accuracy = m.evaluate(valid_dataset)
        
        print(f"Validation loss: {loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")