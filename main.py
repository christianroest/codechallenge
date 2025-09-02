import cv2
import numpy as np
from data import make_tf_dataset
from model import build_model

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


if __name__ == "__main__":
    dataset = make_tf_dataset("/mnt/d/code_data", in_shape[:2])
    for i, (imgs, segs) in enumerate(dataset.take(5)):
        print(imgs.numpy().shape)
        print(segs.numpy().shape)

        # Export a few samples to the workdir
        img = imgs.numpy().astype(np.uint8)
        seg = segs.numpy().astype(np.uint8) * 20  # Scale segmentation for visibility
        cv2.imwrite(f"sample_img_{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"sample_seg_{i}.png", cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))

    m = build_model(in_shape, len(out_classes))
    m.summary()

    sample_input = dataset.take(5).batch(5)
    preds = m.predict(sample_input)  # Run a prediction to test the model

    print(preds.shape)  # Should be (5, 1024, 1024, 3)
