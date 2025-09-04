import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    concatenate,
    MaxPooling2D,
    Conv2DTranspose,
)
from tensorflow.keras.models import Model


def build_model(
    input_shape: tuple,
    num_classes: int,
    num_down_steps: int = 7,
    num_filters_start: int = 16,
    num_filters_max: int = 512,
) -> tf.keras.models.Model:
    """
    Basic model class that implements a very simple model to test the rest of the pipeline

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels).

        Returns:
        tensorflow.keras.models.Model: Keras model.
    """

    def conv_block(x, filters, kernel_size=(3, 3), activation="relu", padding="same"):
        """
        Build a convolutional block with two stacked Conv2D layers.
    
        This utility function applies two consecutive 2D convolutions with the same
        number of filters, kernel size, activation, and padding.
        """
        x = Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding=padding,
        )(x)
        x = Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding=padding,
        )(x)
        return x

    i = Input(input_shape)

    # Add a layer that rescales input images to the [0, 1] range
    i = tf.keras.layers.Rescaling(1.0 / 255)(i)

    # Downsampling path
    skips = []
    x = i
    num_filters = num_filters_start
    
    for _ in range(num_down_steps):
        # Apply two convolutions (conv_block)
        x = conv_block(x, min(num_filters, num_filters_max))
        
        # Save feature map for later concatenation
        skips.append(x) 
        
        # Downsample by a factor of 2
        x = MaxPooling2D((2, 2))(x)
        
        # Double the number of feature maps at each deeper level
        num_filters *= 2
    
    # Bottom bottleneck layer    
    x = conv_block(x, min(num_filters, num_filters_max))
    
    # Reverse the skip list and halve the number of filters
    skips = reversed(skips)
    num_filters //= 2
    
    # Upsampling path
    for skip in skips:
        # Upsample using transposed convolutions
        x = Conv2DTranspose(min(num_filters, num_filters_max), (2, 2), strides=(2, 2), padding="same")(x)
        
        # Skip connection to downsampling path
        x = concatenate([x, skip])
        
        # Apply convolution after merging
        x = conv_block(x, min(num_filters, num_filters_max))
        num_filters //= 2

    # Final output layer with num_classes channels (logit output)
    o = Conv2D(num_classes, (3, 3), padding="same")(x)
    return Model(i, o)
