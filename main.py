import cv2
import numpy as np
from data import make_tf_dataset
from model import build_model


if __name__ == "__main__":
    in_shape = 1024, 1024, 3
    m = build_model(in_shape)
    m.summary()

    dataset = make_tf_dataset('/mnt/d/code_data')
    for i, (imgs, segs) in enumerate(dataset.take(5)):
        print(imgs.numpy().shape)
        print(segs.numpy().shape)

        # Export a few samples to the workdir 
        img = imgs.numpy().astype(np.uint8)
        seg = segs.numpy().astype(np.uint8) * 20  # Scale segmentation for visibility
        cv2.imwrite(f"sample_img_{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"sample_seg_{i}.png", cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
    