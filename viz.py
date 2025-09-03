import matplotlib.pyplot as plt

def plot_sample(image, segmentation, filename):
    """
    Plots a sample image and its corresponding segmentation mask side by side.
    
    The segmentation mask is assumed to have integer class labels, and is plotted as an overlay with 0 being transparant, and each class having a distinct color
    """

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(segmentation, alpha=0.5, cmap='jet', vmin=0, vmax=segmentation.max())
    plt.title("Segmentation Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.savefig(filename)
    plt.close()