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
    num_down_steps: int = 4,
    num_filters_start: int = 32,
) -> tf.keras.models.Model:
    """
    Basic model class that implements a very simple model to test the rest of the pipeline

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels).

        Returns:
        tensorflow.keras.models.Model: Keras model.
    """

    def conv_block(x, filters, kernel_size=(3, 3), activation="relu", padding="same"):
        x = Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding=padding,
            kernel_regularizer="l2",
        )(x)
        x = Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding=padding,
            kernel_regularizer="l2",
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
        x = conv_block(x, num_filters)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        x = MaxPooling2D((2, 2))(x)
        num_filters *= 2
    x = conv_block(x, num_filters)
    skips = reversed(skips)
    num_filters //= 2
    # Upsampling path
    for skip in skips:
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(x)
        x = concatenate([x, skip])
        x = conv_block(x, num_filters)
        num_filters //= 2

    o = Conv2D(num_classes, (3, 3), padding="same")(x)
    return Model(i, o)


if __name__ == "__main__":
    in_shape = 1024, 1024, 3
    m = build_model(in_shape, 1)
    m.summary()
